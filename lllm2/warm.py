"""Bounded token-prefix conversation measurements, separate from cold probes."""
import copy
import hashlib
import json
from pathlib import Path
import threading
import time

from .discovery import hardware
from .engine import Cancelled
from .settings import batch_settings, execution_settings, cache_settings


def host_memory(process):
    result = dict(pid=process.pid if process else None, rss_mib=None, anonymous_mib=None,
                  swap_mib=None, lifetime_peak_rss_mib=None, available_mib=None)
    try:
        if process is None or process.poll() is not None:
            raise OSError('Owned engine is not running')
        fields = {}
        for line in (Path('/proc') / str(process.pid) / 'status').read_text().splitlines():
            key, _, value = line.partition(':')
            if key in ['VmRSS', 'RssAnon', 'VmSwap', 'VmHWM']:
                fields[key] = int(value.split()[0]) / 1024
        if process.poll() is not None:
            raise OSError('Owned engine exited during memory sampling')
        for name, key in [('rss_mib', 'VmRSS'), ('anonymous_mib', 'RssAnon'),
                          ('swap_mib', 'VmSwap'), ('lifetime_peak_rss_mib', 'VmHWM')]:
            result[name] = fields.get(key)
        for line in Path('/proc/meminfo').read_text().splitlines():
            if line.startswith('MemAvailable:'):
                result['available_mib'] = int(line.split()[1]) / 1024
    except (OSError, ValueError) as e:
        result['error'] = str(e)
    return result


def measure_turn(bench, s, opts, sample, prompt, cache_prompt):
    engine = bench.engine
    with engine.guard:
        process = engine.process
    memory = []
    finished = threading.Event()
    lock = threading.Lock()
    progress = dict(output='', output_token_ids=[], first_token_event_seconds=None,
                    first_text_seconds=None, completion_seconds=None, events=0)
    started = time.monotonic()

    def observe(event):
        elapsed = time.monotonic() - started
        with lock:
            progress['events'] += 1
            # Progress events can contain placeholder tokens. They are not output.
            if 'prompt_progress' not in event and event.get('stop') is not True:
                tokens = event.get('tokens', [])
                if not isinstance(tokens, list) or any(type(t) is not int or t < 0 for t in tokens):
                    raise RuntimeError('Invalid generated token IDs in stream.')
                text = event.get('content', '')
                if not isinstance(text, str):
                    raise RuntimeError('Invalid generated text in stream.')
                if tokens and progress['first_token_event_seconds'] is None:
                    progress['first_token_event_seconds'] = elapsed
                if text and progress['first_text_seconds'] is None:
                    progress['first_text_seconds'] = elapsed
                progress['output_token_ids'].extend(tokens)
                progress['output'] += text
                if len(progress['output_token_ids']) > opts['output_tokens'] + 256 or len(progress['output']) > 2 * 1024 * 1024:
                    raise RuntimeError('Generated stream exceeded bounded output budget.')
            if event.get('stop') is True:
                progress['completion_seconds'] = elapsed

    def sample_memory():
        while not finished.is_set():
            h = hardware()
            memory.append(dict(elapsed_seconds=time.monotonic() - started, host=host_memory(process),
                               gpus=h['gpus'], error=h['error']))
            finished.wait(.5)

    sample['host_before'] = host_memory(process)
    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    try:
        response = engine.stream_completion(dict(prompt=prompt, n_predict=opts['output_tokens'],
            temperature=0, seed=42, cache_prompt=cache_prompt, stream=True, return_tokens=True,
            ignore_eos=True), bench.cancel, opts['timeout'], observe)
        sample['response'] = {k: v for k, v in response.items() if k not in ['prompt', 'content', 'tokens']}
        timings = response.get('timings', {})
        processed, reused = timings.get('prompt_n'), timings.get('cache_n')
        predicted = response.get('tokens_predicted')
        sample.update(timings=timings, processed_tokens=processed, reused_tokens=reused,
                      output_tokens=predicted, decode_tok_s=timings.get('predicted_per_second'),
                      processed_prefill_tok_s=timings.get('prompt_per_second'),
                      engine_cache_occupancy_tokens=response.get('tokens_cached'),
                      execution_settings=execution_settings(s, engine.execution_environment, engine.state()['logs'], response))
        if any(type(n) is not int or n < 0 for n in [processed, reused]) or processed + reused != len(prompt):
            raise RuntimeError('Missing or inconsistent processed/reused prompt accounting; cannot establish reuse.')
        if response.get('tokens_evaluated') != len(prompt) or response.get('truncated') or predicted != opts['output_tokens']:
            raise RuntimeError('Incomplete warm workload or context truncation.')
        with lock:
            if len(progress['output_token_ids']) != predicted or progress['first_token_event_seconds'] is None:
                raise RuntimeError('Stream token IDs do not match the completed output budget (this engine may omit byte tokens while buffering UTF-8); exact history cannot be reconstructed.')
        if not cache_prompt and reused != 0:
            raise RuntimeError('Uncached replay unexpectedly reused prompt tokens.')
        if sample['turn'] == 'cold' and not sample['control'] and reused != 0:
            raise RuntimeError('First conversation turn was not cold.')
        sample.update(status='complete', reuse_observed=reused > 0)
    except BaseException as e:
        sample.update(status='cancelled' if isinstance(e, Cancelled) else 'failed', error=str(e) or type(e).__name__)
        raise
    finally:
        finished.set()
        sampler.join(timeout=5)
        with lock:
            sample.update(copy.deepcopy(progress))
        sample.update(wall_seconds=time.monotonic() - started, memory=list(memory), host_after=host_memory(process))
        host_points = [sample['host_before'], sample['host_after']] + [m['host'] for m in memory]
        sample['peak_engine_rss_mib'] = max((h['rss_mib'] for h in host_points if h.get('rss_mib') is not None), default=None)
        sample['peak_total_gpu_used_mib'] = max((sum(g['used_mib'] for g in m['gpus']) for m in memory), default=None)


def run_conversation(bench, s, opts, result):
    def save():
        bench.store.put('result', result['id'], result)

    for repetition in range(1, opts['repeats'] + 1):
        if bench.cancel.is_set():
            raise Cancelled()
        bench.update(phase=f'Warm conversation {repetition}/{opts["repeats"]}: loading')
        bench.engine.start(s, bench.cancel, opts['timeout'])
        result['argv'] = bench.engine.argv
        base, provenance = bench.prompt(s, 'long-code', opts['prompt_tokens'], opts['timeout'])
        continuation = bench.req('/tokenize', dict(content='\nContinue the review. Add concrete validation for negative deposits and pagination limits.\n',
                                                  add_special=False, parse_special=False), opts['timeout'])['tokens'][:128]
        replacement = bench.req('/tokenize', dict(content='\nReview a completely unrelated image processing pipeline with resizing, rotation and color conversion.\n',
                                                 add_special=False, parse_special=False), opts['timeout'])['tokens']
        if not continuation or not replacement:
            raise RuntimeError('Could not tokenize conversation continuation.')
        batches = batch_settings(s, bench.engine.state()['logs'])
        turns = []

        def run(turn, prompt, control=False):
            if bench.cancel.is_set():
                raise Cancelled()
            if len(prompt) + opts['output_tokens'] + 32 > s.context:
                raise ValueError('Conversation turn exceeds its bounded context budget.')
            sample = dict(workload='warm-conversation', turn=turn, control=control, repetition=repetition,
                          status='running', input_tokens=len(prompt), output_budget=opts['output_tokens'],
                          slots=1, context_per_slot=s.context, batch_settings=batches, cache_settings=cache_settings(s),
                          prompt=dict(generator='token-conversation-v1', token_ids=list(prompt),
                                      token_sha256=hashlib.sha256(json.dumps(prompt).encode()).hexdigest(),
                                      initial_source_sha256=provenance['source_sha256']),
                          sampling=dict(temperature=0, seed=42, ignore_eos=True, cache_prompt=not control, stream=True, return_tokens=True))
            result['samples'].append(sample)
            save()
            bench.update(phase=f'Warm conversation {repetition}/{opts["repeats"]}: {"uncached replay " if control else ""}{turn}')
            try:
                measure_turn(bench, s, opts, sample, prompt, not control)
            finally:
                if sample.get('execution_settings'):
                    result['execution_settings'] = sample['execution_settings']
                with bench.engine.log_lock:
                    logs = list(bench.engine.lines)
                starts = [i for i, line in enumerate(logs) if line.startswith('Launching: ')]
                # If the bounded log ring lost the launch marker, provenance is
                # unknown; do not attribute retained lines to this launch.
                sample['engine_log_tail'] = logs[starts[-1] + 1:][-80:] if starts else []
                save()
            return sample

        first = run('cold', base)
        turns.append(('cold', base))
        appended = base + first['output_token_ids'] + continuation
        run('append', appended)
        turns.append(('append', appended))
        # Change source near the previous prompt ending, retaining a long exact prefix.
        suffix = list(appended)
        at = max(16, len(base) - 64)
        suffix[at:at + min(16, len(replacement))] = replacement[:16]
        run('suffix-edit', suffix)
        turns.append(('suffix-edit', suffix))
        away = list(base)
        away[8:-64] = (replacement * (len(base) // len(replacement) + 1))[:len(base) - 72]
        run('switch-away', away)
        turns.append(('switch-away', away))
        run('switch-back', suffix)
        turns.append(('switch-back', suffix))
        early = list(suffix)
        early[16:16 + min(16, len(replacement))] = replacement[:16]
        run('history-edit', early)
        turns.append(('history-edit', early))
        for turn, prompt in turns:
            run(turn, prompt, control=True)
        result['reuse_summary'] = [dict(repetition=x['repetition'], turn=x['turn'], control=x['control'],
                                       processed_tokens=x.get('processed_tokens'), reused_tokens=x.get('reused_tokens'),
                                       status=x['status']) for x in result['samples']]
        save()
        bench.engine.stop()

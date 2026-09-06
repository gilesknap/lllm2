import copy
import hashlib
import json
import math
import threading
import time
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from .discovery import hardware, identity, metadata, probe
from .engine import Cancelled, GPUUnavailable
from .settings import Settings, capabilities, launch_args
from . import config

WORKLOADS = {
    'generate': 'Implement a Python LRU cache with get, put, bounded capacity and O(1) operations. Explain edge cases briefly and provide the code.',
    'edit': 'Edit the supplied Python modules to reject negative amounts and use decimal.Decimal for monetary arithmetic. Return the changed code and explain the changes.',
    'long-code': 'Review the supplied Python modules. Identify shared design problems, then implement a reusable ledger service with validated transactions and pagination. Return Python code.',
}


def stamp():
    return datetime.now(timezone.utc).isoformat()


def source_text(index):
    return f'''# module ledger_{index}.py
class Ledger{index}:
    def __init__(self):
        self.entries = []
    def deposit(self, account, amount):
        self.entries.append((account, amount))
    def balance(self, account):
        return sum(amount for owner, amount in self.entries if owner == account)
    def page(self, start, count):
        return self.entries[start:start + count]

'''


def suite(s):
    # The selected (normally inherited or saved) configuration is the baseline.
    # Every candidate changes exactly one setting, and invalid combinations skip.
    base = replace(s)
    caps = capabilities(base)
    variants = [('baseline',base)]
    skipped = []
    def candidate(label, **changes):
        v = replace(base, **changes)
        if v == base:
            return
        try:
            launch_args(v, config.ENGINE_PORT)
        except ValueError as e:
            skipped.append(dict(option=label,reason=str(e)))
        else:
            variants.append((label,v))
    for mode in ['none','draft-mtp','draft-dflash','ngram-simple']:
        if mode == base.speculation:
            continue
        if mode == 'none' or caps[mode]['status'] == 'available':
            candidate('speculation-' + mode,speculation=mode)
        else:
            skipped.append(dict(option=mode,reason=caps[mode]['reason']))
    if caps['cache']['status'] == 'available':
        for cache in ['f16','q8_0','q4_0']:
            candidate('cache-' + cache,cache=cache)
    if caps['flash']['status'] == 'available':
        candidate('flash-' + ('off' if base.flash == 'on' else 'on'),flash='off' if base.flash == 'on' else 'on')
    if caps['effort']['status'] == 'available':
        for effort in ['default','low','high']:
            candidate('effort-' + effort,effort=effort)
    return variants, skipped


class Bench:
    def __init__(self, engine, store):
        self.engine, self.store = engine, store
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.active = False
        self.progress = dict(status='idle')

    def snapshot(self):
        with self.lock:
            return copy.deepcopy(self.progress) | dict(active=self.active)

    def update(self, **values):
        with self.lock:
            self.progress.update(values)

    def submit(self, data):
        s = Settings.parse(data['settings'])
        opts = dict(workloads=data.get('workloads',['generate','edit','long-code']),
                    prompt_tokens=data.get('prompt_tokens',1024), output_tokens=data.get('output_tokens',256),
                    search_context=data.get('search_context',True), max_context=data.get('max_context',metadata(s.model)['context'] or 131072),
                    timeout=data.get('timeout',180), context_timeout=data.get('context_timeout',900), repeats=data.get('repeats',1))
        if not opts['workloads'] or not isinstance(opts['workloads'],list) or any(w not in WORKLOADS for w in opts['workloads']):
            raise ValueError('Select coding workloads.')
        for k, lo, hi in [('prompt_tokens',128,131072),('output_tokens',16,4096),('max_context',512,1048576),('timeout',10,1800),('context_timeout',10,3600),('repeats',1,5)]:
            if type(opts[k]) is not int or not lo <= opts[k] <= hi:
                raise ValueError(f'{k} must be in {lo}..{hi}')
        if type(opts['search_context']) is not bool:
            raise ValueError('search_context must be boolean')
        if opts['search_context'] and opts['max_context'] < opts['output_tokens'] + 160:
            raise ValueError('Context ceiling must fit the output budget plus a real prompt.')
        if opts['prompt_tokens'] + opts['output_tokens'] + 32 > s.context // s.slots:
            raise ValueError('Shared prompt and output budgets must fit context per slot, with 32 tokens of margin.')
        mode = data.get('mode','baseline')
        variants, skipped = suite(s) if mode == 'suite' else ([('baseline' if mode == 'baseline' else 'custom',replace(s))],[])
        if mode not in ['baseline','suite','custom','combinations']:
            raise ValueError('Unknown benchmark mode')
        if mode == 'combinations':
            choices = data.get('combinations',[])
            if not choices or len(choices)>20:
                raise ValueError('Choose 1..20 combinations.')
            variants = [(f'combination-{i+1}',Settings.parse(v)) for i,v in enumerate(choices)]
            if any((v.model,v.engine,v.backend,v.device,v.context,v.slots) != (s.model,s.engine,s.backend,s.device,s.context,s.slots) for _,v in variants):
                raise ValueError('Combinations must share model, engine, backend, device, context and slots for comparison.')
        for _,v in variants:
            launch_args(v,config.ENGINE_PORT)
        with self.lock:
            if self.active:
                raise ValueError('An operation is already running.')
            if self.engine.state()['running']:
                raise ValueError('Stop normal serving before starting an experiment.')
            self.active = True
            self.cancel.clear()
            self.progress = dict(status='queued', total=len(variants), completed=0, skipped=skipped)
        threading.Thread(target=self._run,args=(variants,opts),daemon=True).start()
        return self.snapshot()

    def _run(self, variants, opts):
        group = str(uuid.uuid4())
        cancelled = False
        fatal = None
        try:
            for i,(label,s) in enumerate(variants):
                if self.cancel.is_set():
                    raise Cancelled()
                r = dict(id=str(uuid.uuid4()),group=group,label=label,started=stamp(),status='running',
                         settings=s.dict(),options=opts,model=identity(s.model),engine={k:v for k,v in probe(s.engine).items() if k not in ['help','flags']},
                         hardware=hardware(),samples=[],probes=[],largest_observed_context=None,recommended_context=None,
                         note='Cold, uncached single-request coding probes; no quality or long-term stability claim. Context numbers are per slot.')
                if s.drafter and s.speculation == 'draft-dflash':
                    r['drafter'] = identity(s.drafter)
                    r['pair_evidence'] = 'User-declared target-specific ordinary DFlash pair; successful runs verify execution only.'
                self.store.put('result',r['id'],r)
                self.update(status='running', current=label, completed=i, result_id=r['id'], phase='loading',context_probe_started=None)
                try:
                    # Restart each workload/repeat to make cold prefill unambiguous.
                    for workload in opts['workloads']:
                        for repetition in range(opts['repeats']):
                            self.update(phase=f'{workload}: cold run {repetition+1}')
                            self.engine.start(s,self.cancel,opts['timeout'])
                            r['argv'] = self.engine.argv
                            sample = self.measure(s,workload,opts['prompt_tokens'],opts['output_tokens'],opts['timeout'])
                            sample['repetition'] = repetition + 1
                            r['samples'].append(sample)
                            self.store.put('result',r['id'],r)
                            self.engine.stop()
                    if opts['search_context']:
                        self.context_search(s,opts,r)
                    r['status'] = 'complete'
                except Cancelled:
                    r['status'] = 'cancelled'
                    cancelled = True
                except Exception as e:
                    r.update(status='failed',error=str(e),logs=self.engine.state()['logs'])
                    if isinstance(e,GPUUnavailable) or not hardware()['gpus']:
                        fatal = str(e)
                finally:
                    r['finished'] = stamp()
                    self.store.put('result',r['id'],r)
                    self.engine.stop()
                if cancelled or fatal:
                    break
            self.update(status='cancelled' if cancelled else 'halted' if fatal else 'complete', completed=i+1,error=fatal)
        except Cancelled:
            self.update(status='cancelled')
        except Exception as e:
            self.update(status='halted',error=str(e))
        finally:
            try:
                self.engine.stop()
            finally:
                with self.lock:
                    self.active = False

    def req(self,path,body,timeout):
        return self.engine.guarded_request(path,body,self.cancel,timeout)

    def prompt(self,s,workload,tokens,timeout):
        # Tokenize the actual checkpoint's formatted chat prompt, preserving both
        # the complete instruction and template suffix while filling with code.
        def format_prompt(n):
            text = ''.join(source_text(i) for i in range(n)) + '\n' + WORKLOADS[workload]
            body = dict(messages=[dict(role='user',content=text)],add_generation_prompt=True)
            if s.effort != 'default':
                body['reasoning_effort'] = s.effort
            formatted = self.req('/apply-template',body,timeout)['prompt']
            tok = self.req('/tokenize',dict(content=formatted,add_special=True,parse_special=True),timeout)['tokens']
            return text,formatted,tok
        # First estimate is deliberately oversized; truncate only source code tokens.
        text,formatted,tok = format_prompt(max(2, math.ceil(tokens/90)))
        if len(tok) < tokens:
            text,formatted,tok = format_prompt(math.ceil(tokens/max(len(tok),1) * max(2,math.ceil(tokens/90))) + 10)
        if len(tok) < tokens:
            raise RuntimeError('Could not construct requested prompt length.')
        # Keep template start and the final instruction/template ending. Synthetic
        # modules may end partway through a function; recorded prompt is exact.
        keep_tail = min(256,tokens//2)
        actual = tok[:tokens-keep_tail] + tok[-keep_tail:]
        return actual,dict(generator='ledger-v1',workload=workload,source_sha256=hashlib.sha256(text.encode()).hexdigest(),
                           formatted_sha256=hashlib.sha256(formatted.encode()).hexdigest(),token_ids=actual)

    def measure(self,s,workload,tokens,output,timeout):
        prompt, provenance = self.prompt(s,workload,tokens,timeout)
        memory = []
        finished = threading.Event()
        def sample_memory():
            while not finished.is_set():
                h = hardware()
                memory.append(dict(time=stamp(),gpus=h['gpus'],error=h['error']))
                finished.wait(.5)
        sampler = threading.Thread(target=sample_memory,daemon=True)
        sampler.start()
        started = time.monotonic()
        try:
            response = self.req('/completion',dict(prompt=prompt,n_predict=output,temperature=0,seed=42,
                                cache_prompt=False,stream=False,ignore_eos=True),timeout)
            elapsed = time.monotonic()-started
        finally:
            finished.set()
            sampler.join(timeout=5)
        timings = response.get('timings',{})
        predicted = response.get('tokens_predicted',timings.get('predicted_n',0))
        evaluated = response.get('tokens_evaluated',timings.get('prompt_n',0))
        if response.get('truncated') or predicted < output or evaluated < tokens:
            raise RuntimeError(f'Incomplete workload: requested {tokens}+{output} tokens, evaluated {evaluated}, generated {predicted}; truncated={response.get("truncated")}.')
        if timings.get('prompt_n',0) < tokens - 1:
            raise RuntimeError('Prompt cache reuse or incomplete timing detected; cannot report cold prefill.')
        return dict(workload=workload,input_tokens=evaluated,output_tokens=predicted,output_budget=output,
                    context_per_slot=s.context//s.slots,slots=s.slots,wall_seconds=elapsed,
                    prefill_tok_s=timings.get('prompt_per_second'),decode_tok_s=timings.get('predicted_per_second'),
                    timings=timings,memory=memory,peak_total_gpu_used_mib=max((sum(g['used_mib'] for g in m['gpus']) for m in memory),default=None),
                    prompt=provenance,output=response.get('content',''),sampling=dict(temperature=0,seed=42,ignore_eos=True,cache_prompt=False))

    def context_search(self,s,opts,r):
        ceiling = min(opts['max_context'], metadata(s.model)['context'] or opts['max_context'],1048576//s.slots)
        floor = max(512,opts['output_tokens'] + 160)
        floor = math.ceil(floor/256)*256
        ceiling = ceiling//256*256
        good, bad = 0, ceiling+256
        # Probe the selected launch context first, then bisect the remaining
        # interval. The ceiling+256 sentinel keeps the ceiling itself testable.
        attempt = min(max(floor,s.context//s.slots),ceiling)
        attempts = 0
        context_timeout = opts.get('context_timeout',900)
        r['context_search_status'] = 'running'
        while floor <= attempt <= ceiling and attempts < 24:
            if self.cancel.is_set():
                raise Cancelled()
            attempts += 1
            self.update(phase=f'Context probe {attempts} (up to 24): {attempt:,} tokens per slot; {context_timeout}s timeout per operation',context_probe_started=time.time())
            entry = dict(context_per_slot=attempt,status='running',started=stamp(),timeout_seconds=context_timeout)
            r['probes'].append(entry)
            self.store.put('result',r['id'],r)
            candidate = replace(s,context=attempt*s.slots)
            try:
                self.engine.start(candidate,self.cancel,context_timeout)
                entry['sample'] = self.measure(candidate,'long-code',attempt-opts['output_tokens']-32,opts['output_tokens'],context_timeout)
                entry['status'] = 'success'
                good = attempt
            except Cancelled:
                entry['status'] = 'cancelled'
                raise
            except TimeoutError as e:
                entry.update(status='timed_out',error=str(e),logs=self.engine.state()['logs'])
                r['context_search_status'] = 'inconclusive_timeout'
                r['context_search_stop_reason'] = 'Context probe timed out; usable-context limit is unknown. Increase the context-probe timeout to continue testing. Earlier successful probes remain valid.'
                self.update(phase=r['context_search_stop_reason'])
                break
            except Exception as e:
                entry.update(status='failed',error=str(e),logs=self.engine.state()['logs'])
                bad = attempt
                if isinstance(e,GPUUnavailable) or not hardware()['gpus']:
                    raise GPUUnavailable('GPU unavailable after context probe; no further restarts.') from e
            finally:
                self.engine.stop()
                entry['finished'] = stamp()
                r['largest_observed_context'] = good or None
                # Suggested headroom is an estimate, not another measured limit.
                recommended = math.floor(good*.9/256)*256
                r['recommended_context'] = recommended if recommended >= floor else None
                r['recommended_context_is_estimate'] = True
                r['context_ceiling'] = ceiling
                self.store.put('result',r['id'],r)
            lower = max(good//256*256,floor-256)
            upper = math.ceil(bad/256)*256
            if good >= ceiling or upper-lower <= 256:
                break
            attempt = ((lower+upper)//2)//256*256
            if attempt < floor:
                break
        if r['context_search_status'] == 'running':
            resolved = math.ceil(bad/256)*256 - max(good//256*256,floor-256) <= 256
            r['context_search_status'] = 'complete' if good >= ceiling or resolved else 'inconclusive_probe_limit'
        r['context_search_resolution'] = 256
        r['context_ceiling_reached'] = good == ceiling

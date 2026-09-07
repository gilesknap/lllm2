from dataclasses import asdict, dataclass, fields
from pathlib import Path
import re
from .discovery import batch_defaults, metadata, probe, cuda_graph_support, cache_kernel_support, engine_environment, EXECUTION_ENV_KEYS


MTP_MODES = ('draft-mtp', 'draft-mtp,ngram-simple')
LOOKUP_MODES = ('ngram-simple', 'draft-mtp,ngram-simple')


@dataclass
class Settings:
    model: str = ''
    engine: str = ''
    backend: str = 'CUDA'
    device: str = 'CUDA0'
    context: int = 4096
    slots: int = 1
    gpu_layers: int | None = None
    flash: str = 'auto'
    cache: str = 'f16'
    cache_k: str | None = None
    cache_v: str | None = None
    speculation: str = 'none'
    drafter: str = ''
    pair_confirmed: bool = False
    draft_length: int = 15
    effort: str = 'default'
    draft_cache: str = 'default'
    chat_template: str = ''
    batch_size: int | None = None
    ubatch_size: int | None = None
    backend_sampling: bool = False
    cuda_graph_opt: str = 'default'
    cache_ram_mib: int | None = None
    context_checkpoints: int | None = None
    lookup_ngram_n: int | None = None
    lookup_ngram_m: int | None = None

    @classmethod
    def parse(cls, data):
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown launch setting.')
        s = cls(**data)
        for key in ('cache_k', 'cache_v'):
            if getattr(s, key) == '':
                setattr(s, key, None)
            if getattr(s, key) not in (None, 'f16', 'q8_0', 'q4_0'):
                raise ValueError(f'Invalid {key}')
        if type(s.backend_sampling) is not bool:
            raise ValueError('backend_sampling must be boolean')
        if s.cuda_graph_opt not in ('default', 'on', 'off'):
            raise ValueError('Invalid cuda_graph_opt')
        for key, upper in [('cache_ram_mib', 8192), ('context_checkpoints', 32), ('lookup_ngram_n', 256), ('lookup_ngram_m', 256)]:
            if getattr(s, key) == '':
                setattr(s, key, None)
            value = getattr(s, key)
            if value is not None and (type(value) is not int or not 0 <= value <= upper):
                raise ValueError(f'{key} must be blank or an integer in 0..{upper}')
        if (s.lookup_ngram_n is None) != (s.lookup_ngram_m is None):
            raise ValueError('Set both lookup N and M, or leave both blank.')
        if s.lookup_ngram_n is not None and not 1 <= s.lookup_ngram_n <= s.lookup_ngram_m:
            raise ValueError('Lookup requires 1 <= N <= M; N greater than M produces no drafts.')
        for key in ('batch_size', 'ubatch_size'):
            if getattr(s, key) == '':
                setattr(s, key, None)
            value = getattr(s, key)
            if value is not None and (type(value) is not int or not 1 <= value <= 1048576):
                raise ValueError(f'{key} must be blank or an integer in 1..1048576')
        if s.batch_size is not None and s.ubatch_size is not None and s.ubatch_size > s.batch_size:
            raise ValueError('Microbatch must not exceed logical batch size.')
        if s.gpu_layers == '':
            s.gpu_layers = None
        if s.gpu_layers is not None and (type(s.gpu_layers) is not int or not 0 <= s.gpu_layers <= 999):
            raise ValueError('GPU layers must be blank (automatic) or an integer in 0..999')
        for key, low, high in [('context',512,1048576),('slots',1,16),('draft_length',1,256)]:
            value = getattr(s, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in {low}..{high}')
        if s.context % s.slots or s.context // s.slots < 512:
            raise ValueError('Total context must divide evenly into slots, each at least 512 tokens.')
        for key, choices in dict(draft_cache=['default','f16','q8_0','q4_0'],backend=['CUDA','Vulkan'],flash=['auto','on','off'],cache=['f16','q8_0','q4_0'],speculation=['none','draft-mtp','draft-dflash','ngram-simple','draft-mtp,ngram-simple'],effort=['default','minimal','low','medium','high','xhigh','max']).items():
            if getattr(s,key) not in choices:
                raise ValueError(f'Invalid {key}')
        if type(s.pair_confirmed) is not bool:
            raise ValueError('pair_confirmed must be boolean')
        return s

    def dict(self):
        return asdict(self)

    def cache_pair(self):
        return self.cache_k or self.cache, self.cache_v or self.cache


def cache_settings(s):
    k, v = s.cache_pair()
    return dict(requested=dict(cache=s.cache, cache_k=s.cache_k, cache_v=s.cache_v),
                resolved=dict(k=k, v=v), kernel=cache_kernel_support(s.engine, s.backend),
                context_evidence='Inherited planner is calibrated for q8_0/q8_0 only; it is not a measured capacity for this pair.',
                state_scope='K/V choices apply to attention cache; hybrid recurrent states retain engine-selected precision. Draft cache is unchanged.')


def capabilities(s):
    p = probe(s.engine)
    m = metadata(s.model)
    flags, help_text = p['flags'], p['help']
    def flag_status(flag):
        return 'unknown' if p['error'] else 'available' if flag in flags else 'unsupported'
    out = {}
    sampling_status = flag_status('--backend-sampling') if s.backend == 'CUDA' else 'unsupported'
    out['backend_sampling'] = dict(status=sampling_status, reason='Experimental target GPU sampling requires CUDA and --backend-sampling. Requests may fall back to CPU; no measured benefit implied. Draft sampling is unchanged.')
    graph = cuda_graph_support(s.engine)
    reason = graph['reason'] + ' Requires ordinary CUDA Graphs and one visible CUDA device; performance needs measurement.'
    status = 'available' if s.backend == 'CUDA' and graph['supported'] else 'unsupported'
    env = engine_environment(s.engine)
    if env.get('GGML_CUDA_DISABLE_GRAPHS') is not None:
        status, reason = 'missing prerequisites', 'Inherited GGML_CUDA_DISABLE_GRAPHS disables ordinary CUDA Graphs; concurrent streams cannot run.'
    if len([d for d in p['devices'] if d.startswith('CUDA')]) > 1:
        status, reason = 'missing prerequisites', 'Concurrent streams require exactly one visible CUDA device.'
    out['cuda_graph_opt'] = dict(status=status, reason=reason, evidence=graph)
    for name, flag in [('flash','--flash-attn'),('cache','--cache-type-k'),('effort','--reasoning-effort')]:
        status = flag_status(flag)
        if name == 'cache' and '--cache-type-v' not in flags:
            status = 'unsupported'
        out[name] = dict(status=status, reason=f'Binary probe: {flag}. Actual model/backend behavior is validated on launch.')
    pair = cache_settings(s)
    k, v = s.cache_pair()
    out['cache_pair'] = dict(status=out['cache']['status'] if k == v or out['cache']['status'] != 'available' else 'experimental',
                            reason=pair['kernel']['reason'] if k != v else 'Resolved attention cache: K ' + k + ', V ' + v + '. Equal types still require a successful engine/model/backend launch.',
                            evidence=pair)
    defaults = batch_defaults(help_text) if not p['error'] else {}
    for name, flag in [('cache_ram_mib', '--cache-ram'), ('context_checkpoints', '--ctx-checkpoints')]:
        block = re.search(r'(?m)^[^\n]*(?<!\S)' + re.escape(flag) +
                          r'(?=[,\s])[^\n]*(?:\n(?![ \t]*-{1,2}[A-Za-z])[ \t]{10,}[^\n]*)*', help_text)
        match = re.search(r'\(default:\s*(\d+)(?=[,\s)])', block[0]) if block else None
        out[name] = dict(status=flag_status(flag), advertised_default=int(match[1]) if match else None,
                         reason='Blank preserves engine defaults for normal launches. Zero disables this cache component; finite values do not cap total process memory.')
    for name, flag in [('batch_size', '--batch-size'), ('ubatch_size', '--ubatch-size')]:
        default = defaults.get(name)
        out[name] = dict(status=flag_status(flag), advertised_default=default,
                         reason=f'Blank leaves the engine default unchanged: {default if default is not None else "unknown"}. Effective size and performance require a launch measurement.')
    template = Path(s.chat_template).expanduser().read_text() if s.chat_template else m['template']
    if out['effort']['status'] == 'available' and 'reasoning_effort' not in template:
        out['effort'] = dict(status='unknown', reason='Binary accepts effort; checkpoint template does not explicitly reference reasoning_effort. Verify with a launch.')
    for mode in ['draft-mtp','draft-dflash','ngram-simple']:
        status = flag_status('--spec-type')
        reason = 'Advertised by this binary; successful workload run still required.'
        if status == 'available' and mode not in help_text:
            status, reason = 'unsupported', 'This speculative mode is not advertised by the selected binary.'
        if status == 'available' and mode == 'draft-mtp' and not m['mtp']:
            status, reason = ('unknown','GGUF inspection failed.') if m['error'] else ('missing prerequisites','Checkpoint contains no MTP tensors.')
        if status == 'available' and mode == 'draft-dflash':
            if s.backend != 'CUDA':
                status, reason = 'unsupported', 'Initial experimental DFlash integration is CUDA only.'
            elif not s.drafter or not Path(s.drafter).expanduser().is_file():
                status, reason = 'missing prerequisites', 'Provide a target-specific ordinary DFlash GGUF (DFlash 2 is not integrated).'
            elif not s.pair_confirmed:
                status, reason = 'missing prerequisites', 'Confirm this drafter was trained and converted for this exact target; filenames alone are insufficient.'
            elif metadata(s.drafter)['error']:
                status, reason = 'missing prerequisites', 'Drafter is not a readable GGUF.'
            elif '--model-draft' not in flags or '--spec-draft-n-max' not in flags or '--flash-attn' not in flags:
                status, reason = 'unsupported', 'Binary lacks required draft model, draft length or flash-attention controls.'
            else:
                reason = 'Pair declared by user, not yet performance validated. Requires flash attention on. No MTP combination.'
        out[mode] = dict(status=status, reason=reason)
    required = {'--spec-type', '--spec-draft-n-max', '--spec-ngram-simple-size-n', '--spec-ngram-simple-size-m'}
    c = dict(out['draft-mtp'])
    if c['status'] == 'available':
        if out['ngram-simple']['status'] != 'available':
            c = dict(out['ngram-simple'])
        elif not required.issubset(flags):
            c = dict(status='unsupported', reason='Combination requires MTP draft width and both lookup N/M controls.')
        else:
            c['reason'] = 'Explicit draft-mtp,ngram-simple combination. Build 662a0b0 tries lookup first, then MTP on misses; other builds need execution verification. Blank N/M uses 3/3 for this combination only. No measured benefit implied.'
    out['draft-mtp,ngram-simple'] = c
    for key, flag in [('lookup_ngram_n', '--spec-ngram-simple-size-n'), ('lookup_ngram_m', '--spec-ngram-simple-size-m')]:
        out[key] = dict(status=flag_status(flag), reason='Optional paired lookup controls; N is match length, M is lookup draft length. Require 1 <= N <= M. Blank omits flags except explicit MTP+lookup uses 3/3.')
    return out


def speculative_settings(s, timings=None):
    combined = s.speculation == 'draft-mtp,ngram-simple'
    p = probe(s.engine)
    timings = timings or {}
    return dict(mode=s.speculation, mtp_draft_length=s.draft_length if s.speculation in MTP_MODES else None,
                lookup_n=(s.lookup_ngram_n if s.lookup_ngram_n is not None else 3 if combined else None) if s.speculation in LOOKUP_MODES else None,
                lookup_m=(s.lookup_ngram_m if s.lookup_ngram_m is not None else 3 if combined else None) if s.speculation in LOOKUP_MODES else None,
                order='lookup first, MTP fallback' if combined and '662a0b0' in p.get('version', '') else 'not verified for this engine',
                order_evidence='llama.cpp 662a0b0 common/speculative.cpp',
                draft_n=timings.get('draft_n'), draft_n_accepted=timings.get('draft_n_accepted'),
                counter_scope='HTTP counters aggregate all speculative methods; per-method counts unavailable without trace instrumentation')


def launch_environment(s):
    env = engine_environment(s.engine)
    if s.gpu_layers is None:
        # An inherited explicit layer count prevents llama.cpp's fitter from
        # choosing placement, just as an explicit --gpu-layers argument does.
        env.pop('LLAMA_ARG_N_GPU_LAYERS', None)
    if s.cuda_graph_opt != 'default':
        env['GGML_CUDA_GRAPH_OPT'] = '1' if s.cuda_graph_opt == 'on' else '0'
    return env


def execution_settings(s, environment=None, logs=(), response=None):
    """Requested options and observed diagnostics; never infer GPU execution from a flag."""
    lines = list(logs)
    starts = [i for i, line in enumerate(lines) if line.startswith('Launching: ')]
    current = lines[starts[-1] + 1:] if starts else []
    diagnostics = [line for line in current if re.search(r'backend sampl|sampler chain|sampl.*(?:fallback|disabl|not compatible|not supported)', line, re.I)]
    generation = (response or {}).get('generation_settings', {})
    return dict(requested=dict(backend_sampling=s.backend_sampling, cuda_graph_opt=s.cuda_graph_opt),
                child_environment={k: environment.get(k) for k in EXECUTION_ENV_KEYS} if environment is not None else None,
                target_sampling=dict(response_requested=generation.get('backend_sampling'),
                                     offload='unknown; normal logs do not prove complete target GPU sampling'),
                draft_sampling=dict(requested=None, policy='engine default unchanged; enablement not observed'),
                diagnostics=diagnostics[-40:])


def batch_settings(s, logs=()):
    """Keep requests and help defaults separate from observed target-context values."""
    p = probe(s.engine)
    defaults = batch_defaults(p['help']) if not p['error'] else {}
    # Engine logs span restarts. Never attribute a previous launch's values to this one.
    lines = list(logs)
    starts = [i for i, line in enumerate(lines) if line.startswith('Launching: ')]
    current = lines[starts[-1] + 1:] if starts else []
    out = {}
    for key, flag, runtime in [('batch_size', '--batch-size', 'n_batch'), ('ubatch_size', '--ubatch-size', 'n_ubatch')]:
        # The target context is initialized before speculative/draft contexts.
        match = next((m for line in current if (m := re.search(r'\bllama_context:\s+' + runtime + r'\s*=\s*(\d+)\b', line))), None)
        out[key] = dict(requested=getattr(s, key), supported=flag in p['flags'] if not p['error'] else None,
                        advertised_default=defaults.get(key), effective=int(match[1]) if match else None,
                        effective_source='target context startup log' if match else 'unknown; not observed')
    return out


def launch_args(s, port):
    p = probe(s.engine)
    if s.batch_size is not None or s.ubatch_size is not None:
        defaults = batch_defaults(p['help']) if not p['error'] else {}
        batch = s.batch_size if s.batch_size is not None else defaults.get('batch_size')
        micro = s.ubatch_size if s.ubatch_size is not None else defaults.get('ubatch_size')
        if batch is not None and micro is not None and micro > batch:
            raise ValueError('Microbatch must not exceed logical batch size (including advertised engine defaults). Set both values explicitly to override defaults.')
    if s.device not in p['devices'] or not s.device.startswith(s.backend):
        raise ValueError('Choose a detected GPU device matching the backend. Check binary --list-devices output.')
    m = metadata(s.model)
    if m['error']:
        raise ValueError('Cannot read checkpoint: ' + m['error'])
    if m['context'] and s.context // s.slots > m['context']:
        raise ValueError('Requested context per slot exceeds checkpoint context metadata.')
    caps = capabilities(s)
    if s.backend_sampling and caps['backend_sampling']['status'] != 'available':
        raise ValueError(caps['backend_sampling']['reason'])
    if s.cuda_graph_opt != 'default':
        if s.backend != 'CUDA' or not cuda_graph_support(s.engine)['supported']:
            raise ValueError('Explicit CUDA streams settings require a CUDA build with the compiled switch.')
        if s.cuda_graph_opt == 'on' and caps['cuda_graph_opt']['status'] != 'available':
            raise ValueError(caps['cuda_graph_opt']['reason'])
    args = [p['path']]
    def add(flag, value=None):
        if flag not in p['flags']:
            raise ValueError(f'This binary does not advertise {flag}. Select a compatible existing build.')
        args.append(flag)
        if value is not None:
            args.append(str(value))
    for flag, value in [('--model',str(Path(s.model).expanduser().resolve())),('--host','127.0.0.1'),('--port',port),('--ctx-size',s.context),('--parallel',s.slots),('--device',s.device)]:
        add(flag,value)
    if s.gpu_layers is None:
        if not {'--fit', '--fit-target'}.issubset(p['flags']):
            raise ValueError('Automatic GPU layers require an engine with --fit and --fit-target. Select a compatible engine or enter a GPU layer count.')
        # Keep context/slots explicit: fit placement to the requested window,
        # including KV/compute buffers, rather than silently shrinking it.
        add('--fit', 'on')
        add('--fit-target', 1024)
    else:
        add('--gpu-layers', s.gpu_layers)
    add('--jinja')
    if s.backend_sampling:
        add('--backend-sampling')
    for flag, value in [('--batch-size', s.batch_size), ('--ubatch-size', s.ubatch_size),
                        ('--cache-ram', s.cache_ram_mib), ('--ctx-checkpoints', s.context_checkpoints)]:
        if value is not None:
            add(flag, value)
    if s.chat_template:
        template = Path(s.chat_template).expanduser().resolve()
        if not template.is_file():
            raise ValueError('Chat template file does not exist.')
        add('--chat-template-file', template)
    if '--perf' in p['flags']:
        add('--perf')
    if '--no-context-shift' in p['flags']:
        add('--no-context-shift')
    if s.flash != 'auto':
        add('--flash-attn',s.flash)
    cache_k, cache_v = s.cache_pair()
    if (cache_k, cache_v) != ('f16', 'f16'):
        if s.flash != 'on':
            raise ValueError('Quantized KV experiments require flash attention on.')
        add('--cache-type-k', cache_k)
        add('--cache-type-v', cache_v)
    elif '--cache-type-k' in p['flags'] and '--cache-type-v' in p['flags']:
        add('--cache-type-k','f16'); add('--cache-type-v','f16')
    if s.effort != 'default':
        add('--reasoning-effort', s.effort)
    if s.speculation != 'none':
        c = caps[s.speculation]
        if c['status'] != 'available':
            raise ValueError(c['reason'])
        add('--spec-type',s.speculation)
        if s.draft_cache != 'default' and s.speculation in (*MTP_MODES, 'draft-dflash'):
            add('--spec-draft-type-k',s.draft_cache)
            add('--spec-draft-type-v',s.draft_cache)
        if '--spec-draft-n-max' in p['flags']:
            add('--spec-draft-n-max',s.draft_length)
        if s.speculation in LOOKUP_MODES:
            for flag, value in [('--spec-ngram-simple-size-n', s.lookup_ngram_n), ('--spec-ngram-simple-size-m', s.lookup_ngram_m)]:
                if value is not None or s.speculation == 'draft-mtp,ngram-simple':
                    add(flag, 3 if value is None else value)
        if s.speculation == 'draft-dflash':
            if s.flash != 'on':
                raise ValueError('DFlash requires flash attention on.')
            add('--model-draft',str(Path(s.drafter).expanduser().resolve()))
            if '--spec-draft-device' in p['flags']:
                add('--spec-draft-device',s.device)
    elif '--spec-type' in p['flags']:
        add('--spec-type','none')
    return args

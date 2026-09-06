from dataclasses import asdict, dataclass, fields
from pathlib import Path
from .discovery import metadata, probe


@dataclass
class Settings:
    model: str = ''
    engine: str = ''
    backend: str = 'CUDA'
    device: str = 'CUDA0'
    context: int = 4096
    slots: int = 1
    gpu_layers: int = 999
    flash: str = 'auto'
    cache: str = 'f16'
    speculation: str = 'none'
    drafter: str = ''
    pair_confirmed: bool = False
    draft_length: int = 15
    effort: str = 'default'

    @classmethod
    def parse(cls, data):
        if set(data) - {f.name for f in fields(cls)}:
            raise ValueError('Unknown launch setting.')
        s = cls(**data)
        for key, low, high in [('context',512,1048576),('slots',1,16),('gpu_layers',0,999),('draft_length',1,256)]:
            value = getattr(s, key)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f'{key} must be an integer in {low}..{high}')
        if s.context % s.slots or s.context // s.slots < 512:
            raise ValueError('Total context must divide evenly into slots, each at least 512 tokens.')
        for key, choices in dict(backend=['CUDA','Vulkan'],flash=['auto','on','off'],cache=['f16','q8_0','q4_0'],speculation=['none','draft-mtp','draft-dflash','ngram-simple'],effort=['default','minimal','low','medium','high','xhigh','max']).items():
            if getattr(s,key) not in choices:
                raise ValueError(f'Invalid {key}')
        if type(s.pair_confirmed) is not bool:
            raise ValueError('pair_confirmed must be boolean')
        return s

    def dict(self):
        return asdict(self)


def capabilities(s):
    p = probe(s.engine)
    m = metadata(s.model)
    flags, help_text = p['flags'], p['help']
    def flag_status(flag):
        return 'unknown' if p['error'] else 'available' if flag in flags else 'unsupported'
    out = {}
    for name, flag in [('flash','--flash-attn'),('cache','--cache-type-k'),('effort','--reasoning-effort')]:
        status = flag_status(flag)
        if name == 'cache' and '--cache-type-v' not in flags:
            status = 'unsupported'
        out[name] = dict(status=status, reason=f'Binary probe: {flag}. Actual model/backend behavior is validated on launch.')
    if out['effort']['status'] == 'available' and 'reasoning_effort' not in m['template']:
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
    return out


def launch_args(s, port):
    p = probe(s.engine)
    if s.device not in p['devices'] or not s.device.startswith(s.backend):
        raise ValueError('Choose a detected GPU device matching the backend. Check binary --list-devices output.')
    m = metadata(s.model)
    if m['error']:
        raise ValueError('Cannot read checkpoint: ' + m['error'])
    if m['context'] and s.context // s.slots > m['context']:
        raise ValueError('Requested context per slot exceeds checkpoint context metadata.')
    caps = capabilities(s)
    args = [p['path']]
    def add(flag, value=None):
        if flag not in p['flags']:
            raise ValueError(f'This binary does not advertise {flag}. Select a compatible existing build.')
        args.append(flag)
        if value is not None:
            args.append(str(value))
    for flag, value in [('--model',str(Path(s.model).expanduser().resolve())),('--host','127.0.0.1'),('--port',port),('--ctx-size',s.context),('--parallel',s.slots),('--gpu-layers',s.gpu_layers),('--device',s.device)]:
        add(flag,value)
    add('--jinja')
    if '--perf' in p['flags']:
        add('--perf')
    if '--no-context-shift' in p['flags']:
        add('--no-context-shift')
    if s.flash != 'auto':
        add('--flash-attn',s.flash)
    if s.cache != 'f16':
        if s.flash != 'on':
            raise ValueError('Quantized KV experiments require flash attention on.')
        for flag in ['--cache-type-k','--cache-type-v']:
            add(flag,s.cache)
    elif '--cache-type-k' in p['flags'] and '--cache-type-v' in p['flags']:
        add('--cache-type-k','f16'); add('--cache-type-v','f16')
    if s.effort != 'default':
        add('--reasoning-effort', s.effort)
    if s.speculation != 'none':
        c = caps[s.speculation]
        if c['status'] != 'available':
            raise ValueError(c['reason'])
        add('--spec-type',s.speculation)
        if '--spec-draft-n-max' in p['flags']:
            add('--spec-draft-n-max',s.draft_length)
        if s.speculation == 'draft-dflash':
            if s.flash != 'on':
                raise ValueError('DFlash requires flash attention on.')
            add('--model-draft',str(Path(s.drafter).expanduser().resolve()))
            if '--spec-draft-device' in p['flags']:
                add('--spec-draft-device',s.device)
    elif '--spec-type' in p['flags']:
        add('--spec-type','none')
    return args

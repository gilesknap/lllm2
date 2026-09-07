"""Starting settings adapted from lllm3090 e56c8d0 (Apache-2.0).

The cache and MTP choices are inherited tuning; context is the old calibrated
planner's estimate on the detected hardware, not a new measured maximum.
"""
from dataclasses import replace
from pathlib import Path
from .discovery import CATALOG, command, hardware, metadata, probe
from .settings import Settings, capabilities, MTP_MODES
from .recommendations import measured_defaults

# Built-in tuning was previously benchmarked on an RTX 3090.
SOURCE = 'Inherited defaults · estimates'


def catalogue_entry(path):
    p = Path(path).expanduser()
    candidates = [m for m in CATALOG if m['file'] == p.name]
    return next((m for m in candidates if m['name'] == p.parent.name), candidates[0] if len(candidates) == 1 else None)


def context_plan(entry, total_mib, backend, ceiling, *, mtp=False, draft_cache='q8_0', desktop=True, driver_reserve=512):
    # Same reserves, q8 block cost, backend factors and slot policy as the
    # source planner. Retain its conservative vision allowance for these entries.
    budget = total_mib - driver_reserve - 1024 - (2400 if desktop else 0)
    budget -= 230 if backend == 'CUDA' else 0
    budget -= 1024 if entry.get('mmproj') else 0
    budget -= (entry['size_gb'] + entry.get('mmproj_gb', 0)) * 1e9 / 2**20
    cost = entry['kv_kib_per_token'] * (34 / 64)
    if mtp and entry.get('full_attention_layers'):
        cost += entry['kv_kib_per_token'] / entry['full_attention_layers'] * 1.25 * (34 / 64 if draft_cache == 'q8_0' else 1)
    cost *= (1.12 / 2 / (34 / 64)) * (1.136 if backend == 'CUDA' else 1)
    pool = min(1048576, max(0, int(budget * 1024 / cost) // 1024 * 1024))
    if pool < 1024:
        raise ValueError('The inherited memory budget leaves no usable context on this GPU.')
    ceiling = min(ceiling, 1048576)
    def window(slots):
        return min(ceiling, pool // slots) // 1024 * 1024
    slots = 1
    if pool >= 1.5 * window(1):
        slots = 2
        while slots < 4:
            here, further = window(slots), window(slots + 1)
            if further <= 1024 or ((slots + 1) * further) / (slots * here) <= here / further:
                break
            slots += 1
    share = window(slots)
    if share < 512:
        raise ValueError('Checkpoint context metadata is too small for inherited planning.')
    return share * slots, slots


def starting_defaults(selection):
    measured, qualifications = measured_defaults(selection)
    if measured:
        return measured
    fallback = inherited_defaults(selection)
    fallback['notes'] = qualifications + fallback['notes']
    return fallback


def inherited_defaults(selection):
    # Never carry experiments or another model's drafter/effort into a new model.
    s = Settings(model=selection.model, engine=selection.engine, backend=selection.backend, device=selection.device)
    notes = []
    if not s.model or not s.engine:
        return dict(settings=s.dict(), source='generic fallback', notes=['Select a checkpoint and engine to load inherited tuning.'])
    p = probe(s.engine)
    automatic_layers = {'--fit', '--fit-target'}.issubset(p['flags'])
    if automatic_layers:
        notes.append('GPU layers: automatic. At startup the engine fits weights and buffers to available VRAM with a 1024 MiB margin, offloading to system RAM when needed. The engine log records actual placement.')
    else:
        s.gpu_layers = 999
        notes.append('This engine lacks automatic memory fitting. GPU layers request full offload (999); enter a smaller count if the model does not fit.')
    devices = [d for d in p['devices'] if d.startswith(s.backend)]
    if s.device not in devices:
        s.device = devices[0] if devices else ''
    caps = capabilities(s)
    entry = catalogue_entry(s.model)
    m = metadata(s.model)
    s.flash = 'on' if caps['flash']['status'] == 'available' else 'auto'
    if caps['cache']['status'] == 'available' and s.flash == 'on':
        s.cache = 'q8_0'
    else:
        notes.append('This binary cannot apply the inherited q8_0/flash settings; generic cache/context retained.')
    s.draft_length = 3
    if caps['draft-mtp']['status'] == 'available':
        s.speculation = 'draft-mtp'
    else:
        notes.append('MTP not enabled: ' + caps['draft-mtp']['reason'])
    draft_flags = {'--spec-draft-type-k', '--spec-draft-type-v'}
    s.draft_cache = 'q8_0' if draft_flags.issubset(p['flags']) else 'default'
    if s.speculation in MTP_MODES and s.draft_cache == 'default':
        notes.append('Draft-cache controls unavailable; context budget accounts for f16 draft cache.')
    if entry and entry.get('chat_template'):
        if '--chat-template-file' in p['flags']:
            s.chat_template = str(Path(__file__).with_name('templates') / entry['chat_template'])
        else:
            notes.append('Binary lacks the inherited chat-template override flag.')
    cards = hardware()['gpus']
    if entry and len(cards) == 1 and s.cache == 'q8_0':
        rc, state = command(['systemctl', 'is-active', 'graphical.target'], 5)
        desktop = state.strip() != 'inactive'
        rc, reserved = command(['nvidia-smi', '--id=' + cards[0]['uuid'], '--query-gpu=memory.reserved', '--format=csv,noheader,nounits'], 4)
        try:
            reserve = max(0, int(reserved.strip())) if rc == 0 else 512
        except ValueError:
            reserve = 512
        ceiling = min(entry['max_ctx'], m['context'] or entry['max_ctx'])
        try:
            s.context, s.slots = context_plan(entry, cards[0]['total_mib'], s.backend, ceiling,
                                            mtp=s.speculation in MTP_MODES, draft_cache=s.draft_cache,
                                            desktop=desktop, driver_reserve=reserve)
            notes.append('Context/slots use the inherited calibrated planner on this GPU; verify with a workload. They are estimates, not new measurements.')
        except ValueError as e:
            if automatic_layers:
                s.context = min(32768, ceiling) // 512 * 512
                s.context = max(512, s.context)
                s.slots = 1
                s.speculation = 'none'
                notes.append('The full-GPU context estimate does not fit. Starting with one conversation, up to 32K context and speculation off; automatic placement may use system RAM. This is an unmeasured starting point; CPU offloading can slow prompt processing.')
            else:
                notes.append(str(e) + ' Set a smaller model or explicit settings before launch.')
    else:
        notes.append('No calibrated context recommendation: requires a recognised catalogue checkpoint, one detected NVIDIA GPU and q8_0 support.')
    return dict(settings=s.dict(), source=SOURCE, notes=notes)

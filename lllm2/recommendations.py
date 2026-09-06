"""Portable measured starting profiles and compact saved-result provenance."""
import hashlib
import json
import threading
from functools import lru_cache
from pathlib import Path

from .discovery import hardware, identity, metadata, probe, engine_environment, cache_kernel_support
from .settings import Settings, launch_args

PROFILES = json.loads(Path(__file__).with_name('recommendations.json').read_text())
_hash_lock = threading.Lock()


def _stat_key(path):
    s = path.stat()
    return (str(path), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


@lru_cache(maxsize=32)
def _digest(key):
    with open(key[0], 'rb') as f:
        value = hashlib.file_digest(f, 'sha256').hexdigest()
    if _stat_key(Path(key[0])) != key:
        raise ValueError('File changed while checking recommendation identity; retry after the write completes.')
    return value


def fingerprint(path):
    path = Path(path).expanduser().resolve()
    # The lock also prevents concurrent form requests from hashing the same GGUF twice.
    with _hash_lock:
        return _digest(_stat_key(path))


def measured_defaults(selection):
    """Return (resolved profile or None, explicit fallback qualifications)."""
    if not selection.model or not selection.engine:
        return None, []
    path = Path(selection.model).expanduser()
    notes = []
    try:
        candidates = [r for r in PROFILES if r['backend'] == selection.backend
                      and r['model']['size'] == path.stat().st_size]
        if not candidates:
            return None, ['No measured built-in profile for this checkpoint/backend.']
        m = metadata(str(path))
        candidates = [r for r in candidates if not m['error'] and all(
            m.get(k) == r['model'][k] for k in ('architecture', 'mtp'))]
        cards = hardware()['gpus']
        candidates = [r for r in candidates if len(cards) == 1
                      and cards[0]['name'] == r['gpu']['name']
                      and cards[0]['total_mib'] == r['gpu']['total_mib']]
        if not candidates:
            return None, ['Measured built-ins require the exact checkpoint and a single tested RTX 3090; using fallback guidance.']
        digest = fingerprint(path)
        record = next((r for r in candidates if r['model']['sha256'] == digest), None)
        if not record:
            return None, ['Checkpoint fingerprint differs from measured built-ins; using fallback guidance.']
        p = probe(selection.engine)
        values = dict(record['settings'], model=selection.model, engine=selection.engine,
                      backend=selection.backend, device=selection.device)
        devices = [d for d in p['devices'] if d.startswith(selection.backend)]
        if values['device'] not in devices:
            values['device'] = devices[0] if devices else ''
        template = record['template']
        if template.get('resource'):
            resource = Path(template['resource'])
            if resource.name != str(resource):
                raise ValueError('Invalid portable template resource.')
            values['chat_template'] = str(Path(__file__).with_name('templates') / resource)
            template_hash = fingerprint(values['chat_template'])
        else:
            values['chat_template'] = ''
            template_hash = hashlib.sha256(m['template'].encode()).hexdigest()
        changed = []
        if cards[0].get('driver') != record['gpu'].get('driver'):
            changed.append('GPU driver')
        if p['sha256'] != record['engine']['sha256']:
            changed.append('engine build')
        library = record['engine'].get('adjacent_cuda_library')
        if library and selection.backend == 'CUDA':
            current_library = cache_kernel_support(selection.engine, selection.backend).get('library') or {}
            if current_library.get('sha256') != library['sha256']:
                changed.append('adjacent CUDA library')
        if template_hash != template['sha256']:
            changed.append('chat template')
        settings = Settings.parse(values)
        if settings.backend == 'CUDA':
            env = engine_environment(settings.engine)
            inherited = [key for key in ('GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_DISABLE_GRAPHS')
                         if key in env and (key != 'GGML_CUDA_GRAPH_OPT' or settings.cuda_graph_opt == 'default')]
            if inherited:
                changed.append('inherited CUDA execution environment')
                notes.append('Inherited ' + ', '.join(inherited) + ' is preserved. Built-in evidence does not verify these inherited values; execution behavior and performance require revalidation.')
        # Includes device, metadata, speculative prerequisites and every launch flag.
        if settings.speculation != 'none' and '--spec-draft-n-max' not in p['flags']:
            raise ValueError('Binary cannot reproduce the measured draft length.')
        launch_args(settings, 1920)
        if changed:
            notes.append('Changed ' + ' and '.join(changed) + ': settings pass capability checks, but performance and runtime behavior need revalidation.')
            if 'engine build' in changed and (settings.batch_size is None or settings.ubatch_size is None):
                notes.append('Omitted batch controls use the selected engine defaults, which may differ from the measured build.')
        if record.get('summary'):
            notes.append(record['summary'])
        notes.extend(record['limitations'])
        context = record['context']
        reserve = context.get('minimum_observed_free_gpu_mib')
        if reserve is not None and context.get('validated_recommendation'):
            notes.append(f"Tested recommendation: {settings.context:,} total tokens / {settings.slots} slot(s); at least {reserve:,} MiB GPU memory remained free in these probes, including the running desktop. This is an observed reserve, not a searched maximum or a guarantee for other workloads.")
        else:
            notes.append(f"Tested starting allocation: {settings.context:,} total tokens / {settings.slots} slot(s). This is not a searched maximum or a headroom recommendation.")
        source = 'Measured built-in recommendation' if context.get('validated_recommendation') else 'Measured built-in baseline'
        return dict(settings=settings.dict(), source=source + (' · qualified' if changed else ''),
                    notes=notes, evidence=record, qualification=dict(changed=changed, capability_check='passed')), []
    except (OSError, ValueError, KeyError) as e:
        return None, ['Measured built-in cannot be applied: ' + str(e) + ' Using inherited/generic fallback.']


def promotion_provenance(result, settings, use_context):
    """Keep source evidence without copying raw prompts, logs or GPU traces."""
    return dict(kind='benchmark', settings=settings.dict(), result_id=result['id'],
                started=result.get('started'), engine=result.get('engine'), model=result.get('model'),
                hardware=result.get('hardware'), options=result.get('options'),
                measured_settings=result['settings'], template_identity=result.get('template_identity'),
                batch_settings=result.get('batch_settings'),
                execution_settings=result.get('execution_settings'),
                cache_settings=result.get('cache_settings'),
                samples=[{k: s.get(k) for k in ('workload', 'input_tokens', 'output_tokens',
                    'requested_output_budget', 'output_budget', 'wall_seconds', 'peak_engine_rss_mib',
                    'context_per_slot', 'slots', 'prefill_tok_s', 'decode_tok_s', 'peak_total_gpu_used_mib',
                    'batch_settings', 'execution_settings', 'speculative_settings', 'cache_settings')} |
                    dict(adherence={k: s['adherence'].get(k) for k in
                        ('status', 'exact_payload_match', 'output_limit_reached', 'expected_sha256',
                         'actual_sha256', 'thinking_characters', 'policy')} if s.get('adherence') else None)
                    for s in result['samples']],
                context=dict(allocated_total=result['settings']['context'],
                    largest_observed_context=result.get('largest_observed_context'),
                    recommended_context=result.get('recommended_context'),
                    search_status=result.get('context_search_status'),
                    used_headroom_estimate=bool(use_context)),
                note='A completed sample is execution evidence, not a measured gain. Headroom context, if selected, is an estimate.')


def saved_qualifications(provenance, settings):
    """Qualify historical evidence without modifying or rejecting preferences."""
    notes = []
    try:
        measured = provenance.get('model') or {}
        current = identity(settings.model)
        if not all(k in measured for k in ('size', 'mtime_ns')):
            notes.append('Saved result has no checkpoint identity; current checkpoint compatibility is unknown.')
        elif any(current[k] != measured[k] for k in ('size', 'mtime_ns')):
            notes.append('Checkpoint changed since the saved measurement; that result does not validate the current file.')
        else:
            notes.append('Checkpoint size/time match the saved result; legacy results have no full checkpoint fingerprint.')
        engine = provenance.get('engine') or {}
        if not engine.get('sha256') or probe(settings.engine).get('sha256') != engine['sha256']:
            notes.append('Selected engine differs from the saved measurement or its identity is unknown; runtime and performance need revalidation.')
        old_cards = (provenance.get('hardware') or {}).get('gpus', [])
        new_cards = hardware()['gpus']
        card_keys = lambda cards: [(c.get('name'), c.get('total_mib'), c.get('driver')) for c in cards]
        if not old_cards or card_keys(old_cards) != card_keys(new_cards):
            notes.append('Current GPU/driver differs from the saved measurement or is unknown; performance needs revalidation.')
        template = provenance.get('template_identity')
        if not template or not template.get('sha256'):
            notes.append('Saved result did not fingerprint its chat template; current template equivalence is unknown.')
        else:
            digest = fingerprint(settings.chat_template) if settings.chat_template else hashlib.sha256(metadata(settings.model)['template'].encode()).hexdigest()
            if digest != template['sha256']:
                notes.append('Chat template changed since the saved measurement; behavior and performance need revalidation.')
    except (OSError, ValueError, KeyError) as e:
        notes.append('Saved evidence compatibility could not be checked: ' + str(e))
    return notes

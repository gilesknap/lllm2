import csv
import functools
import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from . import config, gguf

CATALOG = json.loads(Path(__file__).with_name('models.json').read_text())


def engine_environment(binary):
    env = {k:v for k,v in os.environ.items() if not k.startswith('LLAMA_ARG_')}
    directory = str(Path(binary).expanduser().resolve().parent)
    existing = env.get('LD_LIBRARY_PATH')
    env['LD_LIBRARY_PATH'] = directory + (os.pathsep + existing if existing else '')
    return env


def command(args, timeout=10, env=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def hardware():
    rc, out = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,driver_version', '--format=csv,noheader,nounits'], 4)
    cards = []
    if rc == 0:
        for row in csv.reader(out.splitlines()):
            if len(row) == 6:
                i, uuid, name, total, used, driver = [x.strip() for x in row]
                cards.append(dict(index=i, uuid=uuid, name=name, total_mib=int(total), used_mib=int(used), driver=driver))
    return dict(gpus=cards, error=None if cards else out[:500], ram_gib=round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1))


def identity(path):
    p = Path(path).expanduser().resolve()
    s = p.stat()
    return dict(path=str(p), size=s.st_size, mtime_ns=s.st_mtime_ns)


@functools.lru_cache(maxsize=128)
def _metadata(path, size, mtime_ns):
    try:
        meta, tensors = gguf.header(path)
        return dict(architecture=meta.get('general.architecture'), name=meta.get('general.name'),
                    context=next((v for k, v in meta.items() if k.endswith('.context_length')), None),
                    mtp=any(m in n.lower() for n in tensors for m in gguf.MTP_MARKERS),
                    template=meta.get('tokenizer.chat_template', ''), error=None)
    except Exception as e:
        return dict(architecture=None, context=None, mtp=None, template='', error=str(e))


def metadata(path):
    return _metadata(**identity(path))


def models():
    found = []
    if config.MODELS_DIR.exists():
        for p in sorted(config.MODELS_DIR.rglob('*.gguf')):
            shard = re.search(r'-(\d{5})-of-(\d{5})\.gguf$', p.name)
            if p.name.startswith('mmproj') or (shard and int(shard.group(1)) != 1):
                continue
            info = identity(p)
            m = metadata(p)
            found.append(dict(**info, name=p.parent.name + '/' + p.name, metadata={k:v for k,v in m.items() if k != 'template'}))
    return found


@functools.lru_cache(maxsize=32)
def _probe(path, size, mtime_ns):
    env = engine_environment(path)
    rc, help_text = command([path, '--help'], 15, env=env)
    _, version = command([path, '--version'], env=env)
    _, devices = command([path, '--list-devices'], env=env) if '--list-devices' in help_text else (1, '')
    flags = sorted(set(re.findall(r'--[a-z][a-z0-9-]+', help_text))) if rc == 0 else []
    available = []
    for line in devices.splitlines():
        match = re.match(r'\s*((?:CUDA|Vulkan)\d+)\s*:', line)
        if match:
            available.append(match.group(1))
    with open(path, 'rb') as binary:
        digest = hashlib.file_digest(binary, 'sha256').hexdigest()
    return dict(path=path, version=version.strip()[:1000], flags=flags, help=help_text,
                devices=available, device_output=devices[:2000], error=None if rc == 0 else help_text[:500],
                sha256=digest)


def probe(path):
    p = Path(path).expanduser().resolve()
    if not p.is_file() or not os.access(p, os.X_OK):
        raise ValueError('Select an executable llama-server binary.')
    return _probe(**identity(p))


def engines():
    _probe.cache_clear()
    paths = set()
    for root in config.ENGINE_ROOTS:
        if root.is_file():
            paths.add(root.resolve())
        elif root.exists():
            paths.update(p.resolve() for p in root.rglob('llama-server') if p.is_file() and os.access(p, os.X_OK))
    if shutil.which('llama-server'):
        paths.add(Path(shutil.which('llama-server')).resolve())
    return [{k:v for k,v in probe(p).items() if k != 'help'} for p in sorted(paths)]

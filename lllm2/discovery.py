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


EXECUTION_ENV_KEYS = ('GGML_CUDA_GRAPH_OPT', 'GGML_CUDA_DISABLE_GRAPHS', 'CUDA_VISIBLE_DEVICES')


@functools.lru_cache(maxsize=32)
def _library_digest(path, device, inode, size, mtime_ns, ctime_ns):
    with open(path, 'rb') as handle:
        digest = hashlib.file_digest(handle, 'sha256').hexdigest()
    stat = Path(path).stat()
    if (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns) != (device, inode, size, mtime_ns, ctime_ns):
        raise ValueError('CUDA library changed during fingerprinting; retry inspection.')
    return digest


def cache_kernel_support(binary, backend):
    """Narrow compiled-library evidence; this does not observe runtime dispatch."""
    out = dict(mixed_gpu_kernel='unknown', runtime_dispatch='unknown; not observed', library=None,
               reason='Independent cache types need a runtime check for this engine/backend; advertised types alone do not prove attention-kernel support.')
    if backend != 'CUDA':
        return out
    library = Path(binary).expanduser().resolve().with_name('libggml-cuda.so')
    try:
        stat = library.stat()
        digest = _library_digest(str(library), stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
        out['library'] = dict(path=str(library), size=stat.st_size, sha256=digest)
        if digest == 'b81f5da083d4c25345569268819bce5f7e14b45de19d055f634db3e22bbcef61':
            out.update(mixed_gpu_kernel='unsupported',
                       reason='This exact adjacent CUDA library rejects mixed K/V types in its flash-attention kernel selector. An explicit mixed-pair launch is experimental: it may fail or use another backend; runtime dispatch and performance remain unverified.',
                       evidence='llama.cpp 662a0b0 fattn.cu; verified binary get_best_fattn_kernel type comparison at 0x216454 returns NONE for unequal types. Adjacent library identity does not prove which library a custom executable loads.')
    except (OSError, ValueError) as error:
        out['reason'] += ' CUDA library identity unavailable: ' + str(error)
    return out


@functools.lru_cache(maxsize=32)
def _cuda_graph_marker(path, size, mtime_ns):
    # Adjacent shared library evidence, independent of command-line help.
    with open(path, 'rb') as handle:
        tail = b''
        while chunk := handle.read(1024 * 1024):
            data = tail + chunk
            if b'GGML_CUDA_GRAPH_OPT\x00' in data:
                return True
            tail = data[-32:]
    return False


def cuda_graph_support(binary):
    library = Path(binary).expanduser().resolve().with_name('libggml-cuda.so')
    try:
        info = identity(library)
        supported = _cuda_graph_marker(**info)
        return dict(supported=supported, library=info,
                    reason='Compiled CUDA streams switch found.' if supported else 'Adjacent CUDA library does not contain the streams switch.')
    except OSError:
        return dict(supported=False, library=None, reason='Cannot confirm streams support in an adjacent readable CUDA library.')


def command(args, timeout=10, env=None):
    try:
        r = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=env)
        return r.returncode, r.stdout + r.stderr
    except (OSError, subprocess.TimeoutExpired) as e:
        return -1, str(e)


def host_memory():
    """System usage excludes reclaimable cache via Linux MemAvailable."""
    try:
        values = {}
        for line in Path('/proc/meminfo').read_text().splitlines():
            key, value = line.split(':', 1)
            if key in ('MemTotal', 'MemAvailable'):
                values[key] = int(value.split()[0])
        total, available = values['MemTotal'], values['MemAvailable']
        if total <= 0 or not 0 <= available <= total:
            raise ValueError('Invalid host memory counters')
        return dict(total_gib=round(total / 2**20, 1),
                    used_gib=round((total - available) / 2**20, 1),
                    available_gib=round(available / 2**20, 1))
    except (OSError, ValueError, KeyError, IndexError):
        return dict(total_gib=None, used_gib=None, available_gib=None)


def hardware():
    rc, out = command(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,memory.used,driver_version', '--format=csv,noheader,nounits'], 4)
    cards = []
    if rc == 0:
        for row in csv.reader(out.splitlines()):
            if len(row) == 6:
                i, uuid, name, total, used, driver = [x.strip() for x in row]
                cards.append(dict(index=i, uuid=uuid, name=name, total_mib=int(total), used_mib=int(used), driver=driver))
    ram = host_memory()
    # Retain the legacy total-only field for existing consumers.
    try:
        total = round(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 2**30, 1)
    except (OSError, ValueError, AttributeError):
        total = ram['total_gib']
    return dict(gpus=cards, error=None if cards else out[:500], ram_gib=total, ram=ram)


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


def engines(refresh=True):
    if refresh:
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


def batch_defaults(help_text):
    """Read numeric advertised defaults only, without assuming a build's values."""
    out = {}
    for key, flag in [('batch_size', '--batch-size'), ('ubatch_size', '--ubatch-size')]:
        block = re.search(r'(?m)^.*(?<!\S)' + re.escape(flag) + r'\s+[^\n]*(?:\n[ \t]{10,}[^\n]*)*', help_text)
        match = re.search(r'\(default:\s*(\d+)\)', block[0]) if block else None
        out[key] = int(match[1]) if match and int(match[1]) > 0 else None
    return out

import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get('LLLM2_MODELS_DIR', Path.home() / 'models')).expanduser()
STATE_DIR = Path(os.environ.get('LLLM2_STATE_DIR', Path.home() / '.local/state/lllm2')).expanduser()
ENGINE_ROOTS = [Path(p).expanduser() for p in os.environ.get(
    'LLLM2_ENGINE_ROOTS', str(Path.home() / '.local/share/lllm3090')).split(os.pathsep)]
ENGINE_PORT = int(os.environ.get('LLLM2_ENGINE_PORT', '1920'))

#!/usr/bin/env bash
# Run inside nvidia/cuda:<version>-devel-rockylinux8, with repo at /repo.
set -euo pipefail
track=${1:?CUDA major track required}
release=${2:?lllm2 release version required}
jobs=${BUILD_JOBS:-2}
cd /repo
# GCC 13 avoids GCC 8 source compatibility patches and keeps the EL8 ABI.
dnf install -y gcc-toolset-13-gcc gcc-toolset-13-gcc-c++ git cmake python3.11
export PATH="/opt/rh/gcc-toolset-13/root/usr/bin:$PATH"
export CC=gcc CXX=g++
readarray -t contract < <(python3.11 - "$track" <<'PY'
import runpy
import sys
c = runpy.run_path('src/lllm2/engine_release.py')
print(c['LLAMA_CPP_REF'])
print(c['REPOSITORY'])
print(c['asset_name'](sys.argv[1]))
print(c['CUDA_TRACKS'][sys.argv[1]])
PY
)
ref=${contract[0]}
repository=${contract[1]}
asset=${contract[2]}
cuda=${contract[3]}
if [[ ${CUDA_VERSION:-} != "$cuda" ]]; then
    echo "Expected CUDA image $cuda, got ${CUDA_VERSION:-unknown}." >&2
    exit 1
fi
work=/build/cuda$track
mkdir -p "$work" /out
# The toolkit supplies a link-only driver stub, without its SONAME symlink.
# Use it for GPU-less build/probe checks, never ship it in the engine.
mkdir -p "$work/driver-stub"
ln -sf /usr/local/cuda/lib64/stubs/libcuda.so "$work/driver-stub/libcuda.so.1"
export LD_LIBRARY_PATH="$work/driver-stub:${LD_LIBRARY_PATH:-}"
# Export separately for CI smoke tests outside the development image.
mkdir -p /checks
cp -L "$work/driver-stub/libcuda.so.1" /checks/libcuda.so.1
if [ ! -d "$work/source" ]; then
    git clone --depth 1 --branch "$ref" "$repository" "$work/source"
fi
revision=$(git -C "$work/source" rev-parse HEAD)
expected=$(git -C "$work/source" rev-parse --verify "$ref^{commit}")
if [[ "$revision" != "$expected" ]]; then
    echo "Cached source does not match $ref; use a fresh build directory." >&2
    exit 1
fi
cmake -S "$work/source" -B "$work/build" \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,$work/driver-stub" \
    -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF \
    -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_OPENSSL=OFF -DLLAMA_USE_PREBUILT_UI=OFF
cmake --build "$work/build" --target llama-server --parallel "$jobs"
stage=$(mktemp -d "$work/package.XXXXXX")
trap 'rm -r "$stage"' EXIT
cp -a "$work/build/bin/llama-server" "$stage/"
cp -a "$work/build/bin/"*.so* "$stage/"
# Resolve transitive dependencies, including CUDA/NCCL and compiler runtimes.
# Keep glibc and the host NVIDIA driver out of the artifact.
export LD_LIBRARY_PATH="$stage:${LD_LIBRARY_PATH:-}"
python3.11 - "$stage" <<'PY'
import pathlib
import shutil
import subprocess
import sys
stage = pathlib.Path(sys.argv[1])
for binary in list(stage.iterdir()):
    output = subprocess.check_output(['ldd', str(binary)], text=True)
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 3 and fields[1] == '=>':
            name, path = fields[0], fields[2]
            if path == 'not':
                raise SystemExit(f'Unresolved dependency: {line}')
            resolved = pathlib.Path(path).resolve()
            if resolved.parent == stage.resolve():
                continue
            package = subprocess.check_output(
                ['rpm', '-qf', '--qf', '%{NAME}', str(resolved)], text=True,
            )
            if package == 'glibc' or name.startswith(('libcuda.', 'libnvidia-')):
                continue
            shutil.copy2(resolved, stage / name)
            print(f'Bundled {name} from {package}')
PY
"$stage/llama-server" --help > "$work/help.txt"
python3.11 - "$stage" "$ref" "$revision" "$cuda" "$release" "$repository" <<'PY'
import json
import pathlib
import sys
stage, ref, revision, cuda, release, repository = sys.argv[1:]
(pathlib.Path(stage) / 'lllm2-engine.json').write_text(json.dumps(dict(
    requested_ref=ref, revision=revision, cuda_track=cuda, built_for_lllm2_version=release,
    repository=repository, backend='cuda', glibc='2.28', architecture='x86_64', packaging_schema=1,
), indent=2) + '\n')
PY
cp "$work/source/LICENSE" "$stage/LICENSE.llama.cpp"
# NVIDIA runtime redistribution terms accompany the bundled libraries.
find -L /usr/local/cuda -iname '*EULA*' -type f -exec cp '{}' "$stage/" \;
# Preserve notices for dependencies copied from distribution packages.
python3.11 - "$stage" <<'PY'
import pathlib
import subprocess
import sys
stage = pathlib.Path(sys.argv[1])
notices = []
for package in ('libgcc', 'libstdc++', 'libgomp', 'libnccl'):
    result = subprocess.run(['rpm', '-ql', package], capture_output=True, text=True)
    for name in result.stdout.splitlines():
        path = pathlib.Path(name)
        if path.is_file() and any(word in path.name.lower() for word in ('license', 'copying', 'copyright')):
            notices.append(f'\n--- {package}: {path.name} ---\n' + path.read_text(errors='replace'))
(stage / 'RUNTIME-LICENSES.txt').write_text(''.join(notices))
PY
python3.11 /repo/scripts/package-engine.py "$stage" "/out/$asset"
(cd /out && sha256sum "$asset" > "$asset.sha256")

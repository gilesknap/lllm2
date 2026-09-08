"""Release engine contract, shared by the installer and container builder."""

LLAMA_CPP_REF = "b10850"
REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
RELEASE_REPOSITORY = "gilesknap/lllm2"
CUDA_TRACKS = {"13": "13.3.1", "12": "12.9.1"}


def asset_name(track: str) -> str:
    return f"lllm2-engine-{LLAMA_CPP_REF}-cuda{CUDA_TRACKS[track]}-el8-x64.tar.gz"

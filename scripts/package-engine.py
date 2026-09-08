"""Package flat engine files, storing identical shared libraries only once."""

import hashlib
import sys
import tarfile
from pathlib import Path


def package_engine(directory: Path, destination: Path) -> None:
    libraries: dict[str, str] = {}
    with tarfile.open(destination, "w:gz", dereference=True) as archive:
        # Prefer versioned filenames as the real file; aliases point straight to it.
        for path in sorted(directory.iterdir(), key=lambda p: (-len(p.name), p.name)):
            if not path.is_file() or path.resolve().parent != directory.resolve():
                raise ValueError(f"Engine package requires flat local files: {path}")
            info = archive.gettarinfo(str(path), arcname=path.name)
            if ".so" in path.name:
                with path.open("rb") as source:
                    digest = hashlib.file_digest(source, "sha256").hexdigest()
                if digest in libraries:
                    info.type = tarfile.SYMTYPE
                    info.linkname = libraries[digest]
                    info.size = 0
                    archive.addfile(info)
                    continue
                libraries[digest] = path.name
            with path.open("rb") as source:
                archive.addfile(info, source)


if __name__ == "__main__":
    package_engine(Path(sys.argv[1]), Path(sys.argv[2]))

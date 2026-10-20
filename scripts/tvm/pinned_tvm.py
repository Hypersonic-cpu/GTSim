"""Import the source checkout and verify both source and binary provenance."""

import os
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TVM_COMMIT = "4d0ac168c78c1e20bf05a6ec5926760bb1c52b05"


def load_tvm():
    source = ROOT / "3rdparty/tvm"
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual != TVM_COMMIT:
        raise RuntimeError(f"expected pinned TVM {TVM_COMMIT}, found {actual}")
    sys.path.insert(0, str(source / "python"))
    os.environ.setdefault("TVM_LIBRARY_PATH", str(ROOT / "build-tvm/lib"))
    import tvm

    if not Path(tvm.__file__).resolve().is_relative_to(source):
        raise RuntimeError("TVM was imported from another checkout")
    # This pin's support.libinfo() is a Python fallback with no native commit
    # metadata. Verify the actual loaded libraries against our build manifest.
    build = Path(os.environ["TVM_LIBRARY_PATH"]).resolve().parent
    manifest = json.loads((build / "gtsim-tvm-build.json").read_text())
    if manifest["tvm_commit"] != TVM_COMMIT or Path(manifest["source"]).resolve() != source:
        raise RuntimeError("TVM build provenance does not match the pinned checkout")
    for key in ("tvm_runtime", "tvm_compiler"):
        library = Path(tvm.base._LOADED_LIBS[key]._name).resolve()
        if library.parent != build / "lib" or hashlib.sha256(library.read_bytes()).hexdigest() != manifest["libraries"].get(library.name):
            raise RuntimeError(f"TVM loaded an unverified library: {library}")
    return tvm

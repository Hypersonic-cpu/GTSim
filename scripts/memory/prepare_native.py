"""Apply integration patches to a build-local copy, preserving unchanged mtimes."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('--source', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--patch', type=Path, action='append', required=True)
args = parser.parse_args()
source, output = args.source.resolve(), args.output.resolve()
if output == source or output.is_relative_to(source) or source.is_relative_to(output):
    parser.error('the integration copy must be outside the upstream source tree')
output.parent.mkdir(parents=True, exist_ok=True)
# Always apply/check the exact patch chain. Only replace files whose contents
# changed, so CMake regeneration does not rebuild every native translation unit.
with tempfile.TemporaryDirectory(prefix='native-staging-', dir=output.parent) as temporary:
    staged = Path(temporary) / 'source'
    shutil.copytree(source, staged, ignore=shutil.ignore_patterns('.git', '__pycache__'))
    for patch in args.patch:
        path = str(patch.resolve())
        subprocess.run(['git', 'apply', '--check', path], cwd=staged, check=True)
        subprocess.run(['git', 'apply', path], cwd=staged, check=True)
    manifest = output / '.gtsim-integration-files.json'
    old_files = json.loads(manifest.read_text()) if manifest.exists() else []
    files = []
    for path in sorted(staged.rglob('*')):
        if not path.is_file():
            continue
        relative = path.relative_to(staged)
        files.append(str(relative))
        destination = output / relative
        if not destination.exists() or destination.read_bytes() != path.read_bytes():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    for relative in set(old_files) - set(files):
        stale = output / relative
        if stale.is_relative_to(output) and stale.is_file():
            stale.unlink()
    manifest.write_text(json.dumps(files, indent=2) + '\n')

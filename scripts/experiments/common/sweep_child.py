"""Join a sweep cgroup and record process resource usage after the driver exits."""

from pathlib import Path
import json
import os
import resource
import subprocess
import sys
import time


def main():
    output, cgroup, *command = sys.argv[1:]
    if cgroup:
        (Path(cgroup) / "cgroup.procs").write_text(str(os.getpid()))
    start = time.monotonic()
    code = subprocess.call(command)
    elapsed = time.monotonic() - start
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = {
        "ru_maxrss_bytes": usage.ru_maxrss * 1024,
        "cpu_seconds": usage.ru_utime + usage.ru_stime,
        "mean_cores": (usage.ru_utime + usage.ru_stime) / elapsed,
    }
    Path(output).write_text(json.dumps(result) + "\n")
    return code if code >= 0 else 128 - code


if __name__ == "__main__":
    sys.exit(main())

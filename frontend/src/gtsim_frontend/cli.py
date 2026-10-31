import argparse
import json
import hashlib
from pathlib import Path
from .serialization.json_codec import loads, digest
from .serialization.printer import print_ir
from .programs.gemm_single import build as gemm
from .programs.fa3_single import build as fa3
from . import driver


def main():
    parser = argparse.ArgumentParser(prog="gtsim-ir")
    parser.add_argument(
        "command", choices=("compile", "run", "inspect", "verify", "validate")
    )
    parser.add_argument("kernel", nargs="?", choices=("gemm", "fa3"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    for name, default in (
        ("batch", 1),
        ("heads-q", 1),
        ("heads-kv", 1),
        ("seq-q", 128),
        ("seq-kv", 128),
        ("head-dim", 128),
    ):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--dtype", default="fp16")
    parser.add_argument("--architecture", default="configs/v0.1.0/h100-sxm.yaml")
    parser.add_argument(
        "--memory-config", default="configs/v0.1.0/memory_full_hbm-0F5M.yaml"
    )
    parser.add_argument("--runner", default="build/gtsim_program")
    parser.add_argument("--out-dir", default="validation/results/ir-framework/run")
    parser.add_argument("--launch-overhead-cycles", type=int, default=0)
    parser.add_argument("--oracle", default="legacy")
    parser.add_argument("--suite", default="h100-v0.1.0")
    args = parser.parse_args()
    try:
        if args.command in ("inspect", "verify"):
            if args.input is None:
                raise ValueError("--input required")
            data = json.loads(args.input.read_text())
            if isinstance(data, dict) and data.get("format") == "gtsim.ir.manifest/1":
                for artifact in data["artifacts"]:
                    path = args.input.parent / artifact["path"]
                    if (
                        hashlib.sha256(path.read_bytes()).hexdigest()
                        != artifact["sha256"]
                    ):
                        raise ValueError(f"artifact hash mismatch: {path}")
                    loads(path.read_text())
                print(
                    json.dumps(data, indent=2)
                    if args.command == "inspect"
                    else f"Manifest: {len(data['artifacts'])} verified artifacts"
                )
            else:
                obj = loads(args.input.read_text())
                print(
                    print_ir(obj)
                    if args.command == "inspect"
                    else f"{obj.symbol}: valid {digest(obj)}"
                )
            return
        if args.command == "validate":
            from .validation import validate

            validate(args)
            return
        if args.input:
            program = loads(args.input.read_text())
        elif args.kernel == "gemm":
            program = gemm(args.m, args.n, args.k, args.dtype)
        elif args.kernel == "fa3":
            program = fa3(
                args.batch,
                args.heads_q,
                args.heads_kv,
                args.seq_q,
                args.seq_kv,
                args.dtype,
                args.head_dim,
            )
        else:
            raise ValueError("kernel or --input program required")
        if args.command == "run":
            result = driver.run(
                program,
                args.out_dir,
                args.runner,
                args.architecture,
                args.memory_config,
                args.launch_overhead_cycles,
            )
            print(
                f"Program cycles: {result['total_cycles']}; device cycles: {[k['device_cycles'] for k in result['per_kernel']]}"
            )
        else:
            profile = driver.hardware(
                args.runner, args.architecture, args.memory_config
            )
            driver.compile_program(
                program, args.out_dir, profile, args.launch_overhead_cycles
            )
    except (ValueError, TypeError, OSError, RuntimeError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()

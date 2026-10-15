#!/usr/bin/env python3
"""Expand task JSON/CSV combinations and run them with bounded parallelism."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PLACEHOLDER = re.compile(r"\$\{([^}]*)\}")
VARIABLE = re.compile(r"[A-Za-z0-9_-]+\Z")
CONFIG_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}
RESERVED_OUTPUT_NAMES = {"inputs", "all_case_names.txt", "all_case_status.log"}
SUCCESS_MARKER = "✅ Simulation completed"


class TaskConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Case:
    name: str
    task_name: str
    argv: tuple[str, ...]


def resolve_from_root(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def template_variables(template: str, label: str) -> set[str]:
    variables: set[str] = set()
    for match in PLACEHOLDER.finditer(template):
        variable = match.group(1)
        if not VARIABLE.fullmatch(variable):
            raise TaskConfigError(
                f"{label}: invalid placeholder ${{{variable}}}; names may contain only letters, digits, '_' and '-'."
            )
        variables.add(variable)
    if "${" in PLACEHOLDER.sub("", template):
        raise TaskConfigError(f"{label}: malformed or unclosed '${{...}}' placeholder.")
    return variables


def substitute(template: str, values: dict[str, str], label: str) -> str:
    variables = template_variables(template, label)
    missing = sorted(variables - values.keys())
    if missing:
        raise TaskConfigError(f"{label}: no value supplied for placeholder(s): {', '.join(missing)}")
    return PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


def read_params_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    records: list[tuple[int, list[str]]] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except OSError as error:
        raise TaskConfigError(f"cannot read parameter CSV {path}: {error}") from error

    for line_number, original in enumerate(lines, start=1):
        if original.lstrip(" \t").startswith("#"):
            continue
        normalized = original.replace(" ", "").replace("\t", "").rstrip(",")
        if not normalized:
            continue
        try:
            row = next(csv.reader([normalized], strict=True))
        except (csv.Error, StopIteration) as error:
            raise TaskConfigError(f"{path}:{line_number}: malformed CSV row: {error}") from error
        records.append((line_number, [field.strip() for field in row]))

    if not records:
        raise TaskConfigError(f"parameter CSV has no header: {path}")

    header_line, header = records[0]
    if not header or any(not item for item in header):
        raise TaskConfigError(f"{path}:{header_line}: CSV header contains an empty field")
    if len(set(header)) != len(header):
        raise TaskConfigError(f"{path}:{header_line}: CSV header contains duplicate names")
    for name in header:
        if not VARIABLE.fullmatch(name):
            raise TaskConfigError(
                f"{path}:{header_line}: invalid CSV header {name!r}; use letters, digits, '_' or '-'."
            )

    rows: list[dict[str, str]] = []
    for line_number, fields in records[1:]:
        if len(fields) != len(header):
            raise TaskConfigError(
                f"{path}:{line_number}: expected {len(header)} fields for header, found {len(fields)}"
            )
        if any(field == "" for field in fields):
            raise TaskConfigError(f"{path}:{line_number}: CSV row contains an empty value")
        rows.append(dict(zip(header, fields)))
    if not rows:
        raise TaskConfigError(f"parameter CSV has a header but no data rows: {path}")
    return header, rows


def validate_output_name(name: str, task_name: str) -> str:
    if not name:
        raise TaskConfigError(f"task {task_name!r}: outfmt expands to an empty name")
    if "\\" in name:
        raise TaskConfigError(f"task {task_name!r}: use '/' rather than backslashes in outfmt paths")
    path = Path(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in path.parts):
        raise TaskConfigError(f"task {task_name!r}: outfmt must be a safe relative output path: {name!r}")
    if path.parts[0] in RESERVED_OUTPUT_NAMES:
        raise TaskConfigError(f"task {task_name!r}: outfmt uses reserved output path {path.parts[0]!r}")
    return path.as_posix()


def validate_command(argv: list[str], task_name: str, case_name: str) -> None:
    if not argv or not argv[0]:
        raise TaskConfigError(f"task {task_name!r}, case {case_name!r}: cmd expands to an empty command")

    executable = Path(argv[0])
    if executable.is_absolute() or "/" in argv[0]:
        executable_path = executable if executable.is_absolute() else ROOT / executable
        if not executable_path.is_file() or not os.access(executable_path, os.X_OK):
            raise TaskConfigError(
                f"task {task_name!r}, case {case_name!r}: executable not found or not executable: {argv[0]}"
            )
    elif shutil.which(argv[0]) is None:
        raise TaskConfigError(f"task {task_name!r}, case {case_name!r}: executable not found in PATH: {argv[0]}")

    for argument in argv[1:]:
        config_argument = argument.split("=", 1)[1] if argument.startswith("-") and "=" in argument else argument
        candidate = Path(config_argument)
        if candidate.suffix.lower() not in CONFIG_SUFFIXES:
            continue
        config_path = candidate if candidate.is_absolute() else ROOT / candidate
        if not config_path.is_file():
            raise TaskConfigError(
                f"task {task_name!r}, case {case_name!r}: configuration file does not exist: {config_argument}"
            )


def expand_tasks(tasks_path: Path, params_path: Path) -> list[Case]:
    def reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise TaskConfigError(f"task JSON contains duplicate key {key!r}")
            result[key] = value
        return result

    try:
        task_data = json.loads(
            tasks_path.read_text(encoding="utf-8"), object_pairs_hook=reject_duplicate_json_keys
        )
    except OSError as error:
        raise TaskConfigError(f"cannot read task JSON {tasks_path}: {error}") from error
    except json.JSONDecodeError as error:
        raise TaskConfigError(f"invalid task JSON {tasks_path}:{error.lineno}:{error.colno}: {error.msg}") from error
    if not isinstance(task_data, dict) or not task_data:
        raise TaskConfigError("task JSON root must be a non-empty object for one workload")
    task_name = task_data.get("name")
    if not isinstance(task_name, str) or not task_name:
        raise TaskConfigError("task JSON must include a non-empty workload 'name'")
    allowed = {"name", "cmd", "arch", "outfmt"}
    unknown = sorted(set(task_data) - allowed)
    missing_fields = sorted({"cmd", "outfmt"} - set(task_data))
    if unknown:
        raise TaskConfigError(f"task {task_name!r}: unknown field(s): {', '.join(unknown)}")
    if missing_fields:
        raise TaskConfigError(f"task {task_name!r}: missing field(s): {', '.join(missing_fields)}")

    command_template = task_data["cmd"]
    output_template = task_data["outfmt"]
    if not isinstance(command_template, str) or not command_template.strip():
        raise TaskConfigError(f"task {task_name!r}: cmd must be a non-empty string")
    if not isinstance(output_template, str) or not output_template:
        raise TaskConfigError(f"task {task_name!r}: outfmt must be a non-empty string")

    arch = task_data.get("arch", {})
    if not isinstance(arch, dict):
        raise TaskConfigError(f"task {task_name!r}: arch must be an object of arrays")
    arch_fields: list[tuple[str, list[str]]] = []
    for field, choices in arch.items():
        if not isinstance(field, str) or not VARIABLE.fullmatch(field):
            raise TaskConfigError(f"task {task_name!r}: invalid arch field name {field!r}")
        if not isinstance(choices, list) or not choices:
            raise TaskConfigError(f"task {task_name!r}: arch field {field!r} must have a non-empty array")
        text_choices: list[str] = []
        for choice in choices:
            if isinstance(choice, (dict, list)) or choice is None:
                raise TaskConfigError(f"task {task_name!r}: arch values must be scalar strings or numbers")
            text = str(choice)
            if not text:
                raise TaskConfigError(f"task {task_name!r}: arch field {field!r} contains an empty value")
            text_choices.append(text)
        arch_fields.append((field, text_choices))

    if not params_path.is_file():
        raise TaskConfigError(f"parameter CSV does not exist: {params_path}")
    csv_header, csv_rows = read_params_csv(params_path)
    duplicate_fields = set(field for field, _ in arch_fields) & set(csv_header)
    if duplicate_fields:
        raise TaskConfigError(
            f"task {task_name!r}: fields appear in both arch and CSV: {', '.join(sorted(duplicate_fields))}"
        )
    available = {field for field, _ in arch_fields} | set(csv_header)
    for label, template in (("cmd", command_template), ("outfmt", output_template)):
        unknown_variables = template_variables(template, f"task {task_name!r} {label}") - available
        if unknown_variables:
            raise TaskConfigError(
                f"task {task_name!r} {label}: placeholder(s) have no arch field or CSV column: "
                f"{', '.join(sorted(unknown_variables))}"
            )

    cases: list[Case] = []
    output_names: set[str] = set()
    arch_value_sets = [choices for _, choices in arch_fields]
    arch_products = itertools.product(*arch_value_sets) if arch_value_sets else [()]
    for arch_values in arch_products:
        arch_record = {field: value for (field, _), value in zip(arch_fields, arch_values)}
        for csv_record in csv_rows:
            values = arch_record | csv_record
            expanded_outfmt = validate_output_name(
                substitute(output_template, values, f"task {task_name!r} outfmt"), task_name
            )
            expanded_name = validate_output_name(f"{task_name}{expanded_outfmt}", task_name)
            collision_key = os.path.normcase(expanded_name).casefold()
            if collision_key in output_names:
                raise TaskConfigError(f"task {task_name!r}: duplicate output name {expanded_name!r}")
            output_names.add(collision_key)

            expanded_command = substitute(command_template, values, f"task {task_name!r} cmd")
            try:
                argv = shlex.split(expanded_command)
            except ValueError as error:
                raise TaskConfigError(
                    f"task {task_name!r}, case {expanded_name!r}: invalid command quoting: {error}"
                ) from error
            validate_command(argv, task_name, expanded_name)
            cases.append(Case(expanded_name, task_name, tuple(argv)))

    # Two distinct names can still collide as a file and a parent directory.
    for case in cases:
        parts = Path(case.name).parts
        for length in range(1, len(parts)):
            parent = os.path.normcase(Path(*parts[:length]).as_posix()).casefold()
            if parent in output_names:
                raise TaskConfigError(
                    f"output path conflict: {case.name!r} is nested under another case output {parent!r}"
                )

    return cases


def append_status(output_root: Path, start_time: float, event: str, case_name: str,
                  lock: threading.Lock) -> None:
    elapsed = time.monotonic() - start_time
    line = f"{elapsed:.3f} [{event}] {case_name}\n"
    with lock:
        with (output_root / "all_case_status.log").open("a", encoding="utf-8") as status_file:
            status_file.write(line)
            status_file.flush()


def validate_output_state(output_root: Path, cases: list[Case], overwrite: bool,
                          tasks_path: Path, params_path: Path) -> set[str]:
    if output_root.exists() and not output_root.is_dir():
        raise TaskConfigError(f"output path exists and is not a directory: {output_root}")

    # Existing ancestors must be directories, otherwise mkdir would fail after
    # the scheduler had started preparing output.
    ancestor = output_root.parent
    while ancestor != ancestor.parent:
        if ancestor.exists() and not ancestor.is_dir():
            raise TaskConfigError(f"output parent is not a directory: {ancestor}")
        ancestor = ancestor.parent

    for filename in ("all_case_names.txt", "all_case_status.log"):
        path = output_root / filename
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise TaskConfigError(f"expected an output file path, found another type: {path}")

    destinations = [
        output_root / "inputs" / "tasks" / tasks_path.name,
        output_root / "inputs" / "params" / params_path.name,
    ]
    for destination in destinations:
        parent = destination.parent
        while parent != output_root:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise TaskConfigError(f"input snapshot parent is not a safe directory: {parent}")
            parent = parent.parent
        if destination.is_symlink() or (destination.exists() and not destination.is_file()):
            raise TaskConfigError(f"input snapshot path is not a regular file: {destination}")

    skipped: set[str] = set()
    for case in cases:
        target = output_root / case.name
        parent = target.parent
        while parent != output_root:
            if parent.is_symlink() or (parent.exists() and not parent.is_dir()):
                raise TaskConfigError(f"case output parent is not a safe directory: {parent}")
            parent = parent.parent
        if target.is_symlink():
            raise TaskConfigError(f"case output path must not be a symlink: {target}")
        if target.exists() and not target.is_dir():
            raise TaskConfigError(f"case output path exists and is not a directory: {target}")
        if target.exists():
            stats_path = target / "stats.txt"
            if stats_path.is_symlink() or (stats_path.exists() and not stats_path.is_file()):
                raise TaskConfigError(f"case stats path is not a regular file: {stats_path}")
            if not overwrite and stats_path.is_file():
                try:
                    completed = SUCCESS_MARKER in stats_path.read_text(
                        encoding="utf-8", errors="replace"
                    )
                except OSError as error:
                    raise TaskConfigError(f"cannot read case stats file {stats_path}: {error}") from error
                if completed:
                    skipped.add(case.name)
    return skipped


def copy_input_snapshot(source: Path, destination: Path, overwrite: bool) -> None:
    if overwrite or not destination.exists():
        shutil.copy2(source, destination)


def run_case(case: Case, output_root: Path, start_time: float,
             status_lock: threading.Lock) -> tuple[str, int]:
    log_path = output_root / case.name / "stats.txt"
    append_status(output_root, start_time, "LAUNCH", case.name, status_lock)
    return_code = 127
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(f"Command: {shlex.join(case.argv)}\n\n")
        log_file.flush()
        try:
            return_code = subprocess.run(
                case.argv,
                cwd=ROOT,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
        except OSError as error:
            log_file.write(f"\nLauncher error: {error}\n")

    event = "SUCCESS" if return_code == 0 else f"FAIL {return_code}"
    append_status(output_root, start_time, event, case.name, status_lock)
    return case.name, return_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", required=True, help="JSON file describing one workload and its command")
    parser.add_argument("--params", required=True, help="CSV file providing workload parameter rows")
    parser.add_argument("--outdir", required=True, help="Output root for logs, case names, and input copies")
    parser.add_argument("-j", "--parallel-jobs", type=int, default=1,
                        help="Maximum number of concurrent cases (default: 1)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace existing case output folders and rerun those cases")
    args = parser.parse_args()
    if args.parallel_jobs < 1:
        parser.error("--parallel-jobs must be at least 1")
    return args


def main() -> int:
    args = parse_args()
    tasks_path = resolve_from_root(args.tasks)
    params_path = resolve_from_root(args.params)
    output_root = resolve_from_root(args.outdir)

    try:
        cases = expand_tasks(tasks_path, params_path)
        if not cases:
            raise TaskConfigError("task configuration expands to zero cases")
        skipped = validate_output_state(output_root, cases, args.overwrite, tasks_path, params_path)
    except TaskConfigError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    # No output directory is created until the entire task matrix passes validation.
    output_root.mkdir(parents=True, exist_ok=True)
    tasks_snapshot = output_root / "inputs" / "tasks"
    params_snapshot = output_root / "inputs" / "params"
    tasks_snapshot.mkdir(parents=True, exist_ok=True)
    params_snapshot.mkdir(parents=True, exist_ok=True)
    copy_input_snapshot(tasks_path, tasks_snapshot / tasks_path.name, args.overwrite)
    copy_input_snapshot(params_path, params_snapshot / params_path.name, args.overwrite)

    (output_root / "all_case_names.txt").write_text(
        "".join(f"{case.name}\n" for case in cases), encoding="utf-8"
    )
    (output_root / "all_case_status.log").touch()
    active_cases: list[Case] = []
    for case in cases:
        target = output_root / case.name
        if case.name in skipped:
            continue
        if args.overwrite and target.exists():
            shutil.rmtree(target)
        target.mkdir(parents=True, exist_ok=True)
        active_cases.append(case)

    start_time = time.monotonic()
    status_lock = threading.Lock()
    for case in cases:
        if case.name in skipped:
            append_status(output_root, start_time, "SKIP", case.name, status_lock)
            print(f"SKIP {case.name}: stats.txt contains the simulation completion marker")

    print(f"Launching {len(active_cases)} cases with {args.parallel_jobs} parallel job(s).")
    failures: list[tuple[str, int]] = []
    with ThreadPoolExecutor(max_workers=args.parallel_jobs) as executor:
        futures = [executor.submit(run_case, case, output_root, start_time, status_lock)
                   for case in active_cases]
        for future in futures:
            case_name, return_code = future.result()
            print(f"{'SUCCESS' if return_code == 0 else f'FAIL {return_code}'} {case_name}", flush=True)
            if return_code:
                failures.append((case_name, return_code))

    print(
        f"Completed {len(active_cases) - len(failures)}/{len(active_cases)} cases; "
        f"skipped {len(skipped)}. Results: {output_root}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

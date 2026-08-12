from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import sys
import tempfile
from pathlib import Path

from .hierarchy import decompose_case
from .models import (
    DemoConfig,
    GraphCase,
    canonical_json_bytes,
    read_demo_config,
    read_graph_cases,
    read_hierarchy_result,
    read_verification_report,
    write_demo_config_once,
    write_graph_cases_once,
    write_hierarchy_result_once,
)
from .samples import build_curated_cases, build_random_cases


_PROJECT_ROOT = Path(os.path.abspath(Path(__file__).parent.parent))
_SCRATCH_ROOT = _PROJECT_ROOT / "runs" / ".tmp"


def _project_path(value: str | Path) -> Path:
    path = value if isinstance(value, Path) else Path(value)
    candidate = path if path.is_absolute() else _PROJECT_ROOT / path
    absolute = Path(os.path.abspath(candidate))
    try:
        relative = absolute.relative_to(_PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(
            "path must remain inside the algorithm directory"
        ) from exc

    current = _PROJECT_ROOT
    parts = relative.parts
    for index, component in enumerate(parts):
        current = current / component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            break
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                f"symlink path components are forbidden: {current}"
            )
        if index < len(parts) - 1 and not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path component is not a directory: {current}")
    return absolute


def _lstat_or_none(path: Path) -> os.stat_result | None:
    try:
        return os.lstat(path)
    except FileNotFoundError:
        return None


def _require_absent(path: Path) -> None:
    if _lstat_or_none(path) is not None:
        raise FileExistsError(f"write-once target already exists: {path}")


def _require_regular_file(path: Path) -> None:
    metadata = _lstat_or_none(path)
    if metadata is None:
        raise FileNotFoundError(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink files are forbidden: {path}")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"expected a regular file: {path}")


def _require_directory(path: Path) -> None:
    metadata = _lstat_or_none(path)
    if metadata is None:
        raise FileNotFoundError(path)
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"symlink directories are forbidden: {path}")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"expected a directory: {path}")


def _ensure_directories(value: Path) -> Path:
    directory = _project_path(value)
    relative = directory.relative_to(_PROJECT_ROOT)
    current = _PROJECT_ROOT
    for component in relative.parts:
        current = current / component
        metadata = _lstat_or_none(current)
        if metadata is None:
            try:
                os.mkdir(current, 0o700)
            except FileExistsError:
                pass
            metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(
                f"symlink path components are forbidden: {current}"
            )
        if not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"path component is not a directory: {current}")
    return directory


def _write_bytes_once(path: Path, payload: bytes) -> None:
    target = _project_path(path)
    _require_directory(target.parent)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    complete = False
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        complete = True
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not complete:
            try:
                os.unlink(target)
            except FileNotFoundError:
                pass


def _new_stage(prefix: str) -> Path:
    scratch = _ensure_directories(_SCRATCH_ROOT)
    return _project_path(Path(tempfile.mkdtemp(prefix=prefix, dir=scratch)))


def _cleanup_stage(stage: Path) -> None:
    absolute = _project_path(stage)
    if absolute.parent != _SCRATCH_ROOT:
        raise ValueError("refusing to clean a directory outside runs/.tmp")
    metadata = _lstat_or_none(absolute)
    if metadata is None:
        return
    if stat.S_ISLNK(metadata.st_mode):
        os.unlink(absolute)
        return
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("staging path is not a directory")
    shutil.rmtree(absolute)


def _publish_directory(stage: Path, destination: Path) -> Path:
    source = _project_path(stage)
    target = _project_path(destination)
    _require_directory(source)
    _require_absent(target)
    _ensure_directories(target.parent)
    _require_absent(target)
    os.rename(source, target)
    return target


def _cases_for_config(config: DemoConfig) -> tuple[GraphCase, ...]:
    curated = build_curated_cases(config.ks)
    random_cases = build_random_cases(
        config.ks,
        config.random_per_k,
        config.seed,
    )
    cases = curated + random_cases
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("sample generation produced duplicate case IDs")
    return cases


def _freeze_engine_inputs(
    stage: Path,
    config: DemoConfig,
    cases: tuple[GraphCase, ...],
) -> None:
    input_dir = _ensure_directories(stage / "input")
    engine_dir = _ensure_directories(stage / "engine")
    write_demo_config_once(input_dir / "config.json", config)
    write_graph_cases_once(input_dir / "cases.jsonl", cases)
    for case in cases:
        candidate = decompose_case(case, config)
        write_hierarchy_result_once(
            engine_dir / f"{case.case_id}.json",
            candidate,
        )


def _generate_samples(config_value: str, output_value: str) -> dict[str, object]:
    config_path = _project_path(config_value)
    output_path = _project_path(output_value)
    _require_regular_file(config_path)
    _require_absent(output_path)
    _ensure_directories(output_path.parent)

    config = read_demo_config(config_path)
    cases = _cases_for_config(config)
    write_graph_cases_once(output_path, cases)
    return {"case_count": len(cases), "output": output_path.relative_to(_PROJECT_ROOT).as_posix()}


def _run_stage(
    input_value: str,
    run_value: str,
    config_value: str,
) -> dict[str, object]:
    cases_path = _project_path(input_value)
    run_dir = _project_path(run_value)
    config_path = _project_path(config_value)
    _require_regular_file(cases_path)
    _require_absent(run_dir)
    _require_regular_file(config_path)

    config = read_demo_config(config_path)
    cases = read_graph_cases(cases_path)
    stage = _new_stage(".run-")
    published = False
    try:
        _freeze_engine_inputs(stage, config, cases)
        _publish_directory(stage, run_dir)
        published = True
    finally:
        if not published:
            _cleanup_stage(stage)
    return {"case_count": len(cases), "run_dir": run_dir.relative_to(_PROJECT_ROOT).as_posix()}


def _verification_api():
    from .verify import publish_verification_reports_once, verify_run

    return publish_verification_reports_once, verify_run


def _verify_stage(run_value: str | Path) -> dict[str, object]:
    run_dir = _project_path(run_value)
    _require_directory(run_dir)
    publish_reports, verify_run = _verification_api()
    publish_reports(run_dir)
    audit = verify_run(run_dir)
    if not audit.get("all_verified", False):
        raise ValueError("run verification did not verify every case")
    return audit


def _safe_tree_bytes(
    root: Path,
    *,
    excluded: frozenset[str] = frozenset(),
) -> dict[str, bytes]:
    directory = _project_path(root)
    _require_directory(directory)
    files: dict[str, bytes] = {}
    for current, directory_names, file_names in os.walk(
        directory, followlinks=False
    ):
        current_path = Path(current)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            path = current_path / name
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"run tree contains a symlink: {path}")
            if not stat.S_ISDIR(metadata.st_mode):
                raise ValueError(f"run tree contains a non-directory: {path}")
        for name in file_names:
            path = current_path / name
            relative = path.relative_to(directory).as_posix()
            if relative in excluded:
                continue
            metadata = os.lstat(path)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"run tree contains a symlink: {path}")
            if not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"run tree contains a non-regular file: {path}")
            files[relative] = path.read_bytes()
    return files


def _merge_expected_files(
    root: Path,
    expected: dict[str, bytes],
) -> tuple[Path, ...]:
    targets: list[Path] = []
    missing: list[tuple[Path, bytes]] = []
    for relative, payload in expected.items():
        target = _project_path(root / relative)
        metadata = _lstat_or_none(target)
        if metadata is not None:
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ValueError(f"existing artifact is not a regular file: {target}")
            if target.read_bytes() != payload:
                raise ValueError(f"existing artifact differs from recomputation: {target}")
        else:
            missing.append((target, payload))
        targets.append(target)
    for target, payload in missing:
        _ensure_directories(target.parent)
        _write_bytes_once(target, payload)
    return tuple(targets)


def _export_or_audit_bundle(
    case: GraphCase,
    hierarchy,
    report,
    config: DemoConfig,
    destination: Path,
) -> None:
    from .export import export_case_bundle

    target = _project_path(destination)
    metadata = _lstat_or_none(target)
    if metadata is None:
        export_case_bundle(case, hierarchy, report, config, target)
        return
    _require_directory(target)

    stage = _new_stage(".bundle-audit-")
    try:
        expected_dir = stage / "bundle"
        export_case_bundle(
            case,
            hierarchy,
            report,
            config,
            expected_dir,
        )
        expected = _safe_tree_bytes(expected_dir)
        current = _safe_tree_bytes(target)
        unknown = set(current).difference(expected)
        if unknown:
            raise ValueError(
                f"existing GPT bundle has unexpected files: {sorted(unknown)!r}"
            )
        _merge_expected_files(target, expected)
    finally:
        _cleanup_stage(stage)


def _expected_summary_manifest(run_dir: Path) -> dict[str, bytes]:
    from .export import export_run_summary

    stage = _new_stage(".summary-audit-")
    try:
        snapshot = _ensure_directories(stage / "run")
        source_files = _safe_tree_bytes(
            run_dir,
            excluded=frozenset({"summary.csv", "manifest.json"}),
        )
        for relative, payload in source_files.items():
            target = snapshot / relative
            _ensure_directories(target.parent)
            _write_bytes_once(target, payload)
        summary_path, manifest_path = export_run_summary(snapshot)
        return {
            "summary.csv": summary_path.read_bytes(),
            "manifest.json": manifest_path.read_bytes(),
        }
    finally:
        _cleanup_stage(stage)


def _publish_or_audit_summary(run_dir: Path) -> None:
    from .export import export_run_summary

    summary = run_dir / "summary.csv"
    manifest = run_dir / "manifest.json"
    summary_exists = _lstat_or_none(summary) is not None
    manifest_exists = _lstat_or_none(manifest) is not None
    if not summary_exists and not manifest_exists:
        export_run_summary(run_dir)
        return
    if summary_exists and manifest_exists:
        return
    _merge_expected_files(run_dir, _expected_summary_manifest(run_dir))


def _require_complete_verification_stage(
    run_dir: Path,
    cases: tuple[GraphCase, ...],
) -> None:
    verification_dir = _project_path(run_dir / "verification")
    _require_directory(verification_dir)
    expected = {f"{case.case_id}.json" for case in cases}
    actual: set[str] = set()
    for entry in verification_dir.iterdir():
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ValueError("verification stage contains a non-regular artifact")
        actual.add(entry.name)
    if actual != expected:
        raise ValueError("verification stage is incomplete or contains extra artifacts")

    publish_reports, _ = _verification_api()
    publish_reports(run_dir)
    for case in cases:
        report = read_verification_report(
            verification_dir / f"{case.case_id}.json"
        )
        if not report.verified:
            raise ValueError(
                f"GPT export requires a verified report for {case.case_id}"
            )


def _require_export_stage_shape(
    run_dir: Path,
    cases: tuple[GraphCase, ...],
) -> None:
    allowed_directories = {"input", "engine", "verification", "gpt"}
    allowed_files = {"summary.csv", "manifest.json"}
    for entry in run_dir.iterdir():
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError("run root contains a symlink artifact")
        if stat.S_ISDIR(metadata.st_mode):
            if entry.name not in allowed_directories:
                raise ValueError(f"run root contains an unexpected directory: {entry.name}")
        elif stat.S_ISREG(metadata.st_mode):
            if entry.name not in allowed_files:
                raise ValueError(f"run root contains an unexpected file: {entry.name}")
        else:
            raise ValueError("run root contains a special filesystem object")

    gpt_dir = run_dir / "gpt"
    if _lstat_or_none(gpt_dir) is None:
        return
    _require_directory(gpt_dir)
    expected_case_ids = {case.case_id for case in cases}
    actual_case_ids: set[str] = set()
    for entry in gpt_dir.iterdir():
        metadata = entry.lstat()
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError("gpt root contains an unexpected artifact")
        actual_case_ids.add(entry.name)
    if not actual_case_ids.issubset(expected_case_ids):
        raise ValueError("gpt root contains an unexpected case directory")


def _export_stage(run_value: str | Path) -> dict[str, object]:
    run_dir = _project_path(run_value)
    _require_directory(run_dir)
    config = read_demo_config(run_dir / "input" / "config.json")
    cases = read_graph_cases(run_dir / "input" / "cases.jsonl")
    _require_complete_verification_stage(run_dir, cases)
    _require_export_stage_shape(run_dir, cases)
    _, verify_run = _verification_api()
    gpt_dir = _ensure_directories(run_dir / "gpt")
    for case in cases:
        hierarchy = read_hierarchy_result(
            run_dir / "engine" / f"{case.case_id}.json"
        )
        report = read_verification_report(
            run_dir / "verification" / f"{case.case_id}.json"
        )
        _export_or_audit_bundle(
            case,
            hierarchy,
            report,
            config,
            gpt_dir / case.case_id,
        )
    _require_export_stage_shape(run_dir, cases)
    _publish_or_audit_summary(run_dir)
    after = verify_run(run_dir)
    if not after.get("all_verified", False):
        raise ValueError("exported run failed its read-only audit")
    return after


def _demo(config_value: str, run_value: str) -> dict[str, object]:
    run_dir = _project_path(run_value)
    config_path = _project_path(config_value)
    _require_absent(run_dir)
    _require_regular_file(config_path)
    config = read_demo_config(config_path)
    cases = _cases_for_config(config)
    if len(cases) != 18:
        raise ValueError("demo requires exactly 18 curated cases")

    stage = _new_stage(".demo-")
    published = False
    try:
        _freeze_engine_inputs(stage, config, cases)
        first_audit = _verify_stage(stage)
        if first_audit.get("case_count") != 18:
            raise ValueError("demo verification did not cover all 18 cases")
        final_audit = _export_stage(stage)
        if (
            final_audit.get("case_count") != 18
            or not final_audit.get("all_verified", False)
        ):
            raise ValueError("demo final audit did not verify 18/18 cases")
        _publish_directory(stage, run_dir)
        published = True
    finally:
        if not published:
            _cleanup_stage(stage)
    return final_audit


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3.13 -m bl_rctn.cli",
        description="BL-RCTN-V_k bounded STRUCTURE_ONLY reference runner",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate-samples")
    generate.add_argument("--config", required=True)
    generate.add_argument("--output", required=True)

    run = commands.add_parser("run")
    run.add_argument("--input", required=True)
    run.add_argument("--run-dir", required=True)
    run.add_argument("--config", required=True)

    verify = commands.add_parser("verify")
    verify.add_argument("--run-dir", required=True)

    export = commands.add_parser("export-gpt")
    export.add_argument("--run-dir", required=True)

    demo = commands.add_parser("demo")
    demo.add_argument("--run-dir", required=True)
    demo.add_argument("--config", required=True)
    return parser


def _dispatch(arguments: argparse.Namespace) -> dict[str, object]:
    if arguments.command == "generate-samples":
        return _generate_samples(arguments.config, arguments.output)
    if arguments.command == "run":
        return _run_stage(arguments.input, arguments.run_dir, arguments.config)
    if arguments.command == "verify":
        return _verify_stage(arguments.run_dir)
    if arguments.command == "export-gpt":
        return _export_stage(arguments.run_dir)
    if arguments.command == "demo":
        return _demo(arguments.config, arguments.run_dir)
    raise ValueError(f"unknown command {arguments.command!r}")


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _dispatch(arguments)
    except (OSError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(result) + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

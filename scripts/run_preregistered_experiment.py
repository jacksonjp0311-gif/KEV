"""Create, verify, and run KEV's fixed frozen-encoder experiment plan.

This runner is deliberately narrower than a general sweep utility.  The seed
order, training steps, retention epsilon, and sole recommendation-eligible
seed are constants.  Each seed calls :func:`kev.evolution.evolve` from the
same hash-pinned public incumbent in a separate state directory.  The
aggregate reports the per-seed ledger decisions; it never replaces them and
never reads training loss as a selection criterion.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from importlib import metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import sys
from typing import Any, Mapping, Sequence

import torch

from kev.artifacts import (
    canonical_json_bytes,
    canonical_json_sha256,
    file_sha256,
    validate_sha256,
    verify_file_sha256,
)
from kev.calibration_evidence import validate_calibration_binding
from kev.encoders import verify_local_encoder_manifest
from kev.evaluation import (
    CHALLENGER_REPORT_SCHEMA,
    COMPARISON_SCHEMA,
    EVALUATION_SCHEMA,
    compare_evaluations,
    evaluate_challenger,
    load_frozen_suite,
)
from kev.evolution import evolve
from kev.model_runtime import CheckpointPredictor
from kev.uc51a2.semantic_breadth import (
    PINNED_HELD_OUT_VOCABULARY,
    _frozen_evaluation_exclusions,
    required_held_out_vocabulary,
)
from kev.uc51a3.alive import AliveStore


ROOT = Path(__file__).resolve().parents[1]
PLAN_SCHEMA = "kev.preregistered-experiment-plan.v1"
AGGREGATE_SCHEMA = "kev.preregistered-experiment-evidence.v1"
ARTIFACT_REF_SCHEMA = "kev.artifact-ref.v1"
ORDERED_SEEDS = (52031, 52047, 52069)
PRIMARY_SEED = 52031
ROBUSTNESS_ONLY_SEEDS = (52047, 52069)
TRAINING_STEPS = 800
RETENTION_EPSILON = 0.0
REQUIRED_ARTIFACT_ROLES = frozenset(
    {
        "public_registry",
        "incumbent_checkpoint",
        "promotion_suite",
        "calibration_fit",
        "held_out_vocabulary",
        "encoder_manifest",
        "reviewed_corpus",
    }
)
TERMINAL_EVENTS = frozenset({"MODEL_QUALIFIED", "MODEL_REJECTED"})
IGNORED_DIRECTORY_NAMES = frozenset({"runs", "state"})
PUBLIC_REGISTRY = (ROOT / "models" / "registry.json").resolve()
FROZEN_EVAL_DIRECTORY = (ROOT / "evals" / "frozen").resolve()
DEFAULT_SUITE = (FROZEN_EVAL_DIRECTORY / "public-audit-v6-260.json").resolve()
DEFAULT_CALIBRATION = (FROZEN_EVAL_DIRECTORY / "calibration-fit-v2.jsonl").resolve()
DEFAULT_HELD_OUT = (FROZEN_EVAL_DIRECTORY / "held-out-vocabulary-v2.txt").resolve()
TRAINER_DISCOVERY_PATTERNS = (
    "manifest-v*.json",
    "held-out-vocabulary-v*.txt",
    "public-audit-v*-260.json",
    "calibration-fit-*.jsonl",
    "*known-failures*.json",
)
EXECUTABLE_SOURCE_PATTERNS = ("kev/**/*.py",)
RUNTIME_PACKAGES = (
    "huggingface-hub",
    "numpy",
    "safetensors",
    "tokenizers",
    "torch",
    "transformers",
)
TRAINING_OBJECTIVE = "L_relation + 0.20 L_cardinality + 0.35 L_contrastive"
FROZEN_SENTENCE_CHECKPOINT_SCHEMA = "kev.frozen-sentence-frames.v1"
TRAINING_SOURCE_SPECS = (
    ("FROZEN_ENCODER_TRAINER", "kev/sentence_training.py"),
    ("ENCODER_RUNTIME", "kev/encoders.py"),
    ("SHARED_SEMANTIC_TRAINING", "kev/uc51a2/semantic_breadth.py"),
)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _current_training_source_manifest() -> list[dict[str, str]]:
    """Recompute the ordered frozen-encoder training source closure."""

    return [
        {
            "role": role,
            "path": relative_path,
            "sha256": file_sha256(ROOT / relative_path),
        }
        for role, relative_path in TRAINING_SOURCE_SPECS
    ]


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _write_exclusive_json(path: Path, value: Any) -> None:
    _write_exclusive(path, _json_bytes(value))


def _object(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _exact_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] | None = None,
    location: str,
) -> None:
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    extra = sorted(set(value) - allowed)
    if missing or extra:
        raise ValueError(f"{location} keys invalid: missing={missing}, extra={extra}")


def _resolve_plan_path(raw: Any, plan_dir: Path, field: str) -> Path:
    if not isinstance(raw, str) or not raw.strip() or "\x00" in raw:
        raise ValueError(f"{field} must be a non-empty local path")
    if "://" in raw:
        raise ValueError(f"{field} must be a local path")
    path = Path(raw).expanduser()
    return (plan_dir / path).resolve() if not path.is_absolute() else path.resolve()


def _artifact_ref(path: str | Path, role: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"artifact is not a file: {source}")
    return {
        "schema": ARTIFACT_REF_SCHEMA,
        "role": role,
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def _plan_relative(path: str | Path, plan_directory: Path) -> str:
    """Prefer relocatable plan paths while retaining a cross-drive fallback."""

    resolved = Path(path).expanduser().resolve()
    try:
        return Path(os.path.relpath(resolved, plan_directory.resolve())).as_posix()
    except ValueError:
        return str(resolved)


def _runtime_environment() -> dict[str, Any]:
    """Return the software environment that can affect experiment execution."""

    packages: dict[str, str] = {}
    for package in RUNTIME_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = "UNAVAILABLE"
    return {
        "schema": "kev.preregistered-runtime-environment.v1",
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
        "packages": packages,
    }


def _discovered_trainer_inputs() -> set[Path]:
    discovered: set[Path] = set()
    for pattern in TRAINER_DISCOVERY_PATTERNS:
        for path in FROZEN_EVAL_DIRECTORY.glob(pattern):
            if path.is_symlink():
                raise ValueError(
                    f"trainer-discovered input cannot be a symbolic link: {path}"
                )
            if path.is_file():
                discovered.add(path.resolve())
    return discovered


def _discovered_executable_sources() -> set[Path]:
    discovered = {Path(__file__).resolve()}
    for pattern in EXECUTABLE_SOURCE_PATTERNS:
        for path in ROOT.glob(pattern):
            if path.is_symlink():
                raise ValueError(
                    f"experiment executable source cannot be a symbolic link: {path}"
                )
            if path.is_file():
                discovered.add(path.resolve())
    return discovered


def _automatic_role(prefix: str, path: Path) -> str:
    try:
        relative = path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        relative = path.name
    readable = "".join(
        character.casefold() if character.isalnum() else "_" for character in relative
    ).strip("_")
    suffix = hashlib.sha256(relative.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}__{readable[:80]}__{suffix}"


def _add_automatic_artifacts(artifact_paths: dict[str, Path]) -> None:
    """Add every implicit trainer input and executable source exactly once."""

    existing = {path.resolve() for path in artifact_paths.values()}
    for prefix, paths in (
        ("trainer_input", _discovered_trainer_inputs()),
        ("executable_source", _discovered_executable_sources()),
    ):
        for path in sorted(paths):
            if path in existing:
                continue
            role = _automatic_role(prefix, path)
            if role in artifact_paths:
                raise ValueError(f"automatic artifact role collision: {role}")
            artifact_paths[role] = path
            existing.add(path)


def _verify_discovered_inventory(
    *,
    name: str,
    discovered: set[Path],
    declared_paths: set[Path],
) -> None:
    def display(path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return str(path)

    missing = sorted(display(path) for path in discovered - declared_paths)
    if missing:
        raise ValueError(
            f"preregistered plan does not close the {name} inventory; "
            "add artifact reference(s): " + ", ".join(missing)
        )


def _validate_plan(plan: Any) -> dict[str, Any]:
    value = _object(plan, "plan")
    _exact_keys(
        value,
        required={
            "schema",
            "frozen",
            "experiment_id",
            "output_root",
            "artifacts",
            "execution",
            "runtime_environment",
            "selection_policy",
        },
        location="plan",
    )
    if value.get("schema") != PLAN_SCHEMA:
        raise ValueError(f"plan schema must be {PLAN_SCHEMA}")
    if value.get("frozen") is not True:
        raise ValueError("experiment plan must declare frozen=true")
    experiment_id = value.get("experiment_id")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string")

    environment = _object(value["runtime_environment"], "runtime_environment")
    _exact_keys(
        environment,
        required={
            "schema",
            "python",
            "python_implementation",
            "platform",
            "byteorder",
            "packages",
        },
        location="runtime_environment",
    )
    if environment.get("schema") != "kev.preregistered-runtime-environment.v1":
        raise ValueError("runtime_environment schema is invalid")
    packages = _object(environment.get("packages"), "runtime_environment.packages")
    if set(packages) != set(RUNTIME_PACKAGES) or not all(
        isinstance(version, str) and version for version in packages.values()
    ):
        raise ValueError("runtime_environment.packages inventory is invalid")
    current_environment = _runtime_environment()
    if environment != current_environment:
        raise ValueError(
            "runtime environment differs from the preregistered software record"
        )

    execution = _object(value["execution"], "execution")
    _exact_keys(
        execution,
        required={"ordered_seeds", "steps", "retention_epsilon"},
        location="execution",
    )
    if execution.get("ordered_seeds") != list(ORDERED_SEEDS):
        raise ValueError(f"ordered_seeds must be exactly {list(ORDERED_SEEDS)}")
    if (
        isinstance(execution.get("steps"), bool)
        or execution.get("steps") != TRAINING_STEPS
    ):
        raise ValueError(f"steps must be exactly {TRAINING_STEPS}")
    epsilon = execution.get("retention_epsilon")
    if isinstance(epsilon, bool) or not isinstance(epsilon, (int, float)):
        raise ValueError("retention_epsilon must be numeric")
    if float(epsilon) != RETENTION_EPSILON:
        raise ValueError(f"retention_epsilon must be {RETENTION_EPSILON}")

    policy = _object(value["selection_policy"], "selection_policy")
    _exact_keys(
        policy,
        required={
            "kind",
            "primary_seed",
            "robustness_only_seeds",
            "requires_all_seed_receipts",
            "training_loss_is_criterion",
        },
        location="selection_policy",
    )
    if policy.get("kind") != "PREDECLARED_PRIMARY_ONLY":
        raise ValueError("selection_policy.kind must be PREDECLARED_PRIMARY_ONLY")
    if policy.get("primary_seed") != PRIMARY_SEED:
        raise ValueError(f"primary_seed must be {PRIMARY_SEED}")
    if policy.get("robustness_only_seeds") != list(ROBUSTNESS_ONLY_SEEDS):
        raise ValueError(
            f"robustness_only_seeds must be exactly {list(ROBUSTNESS_ONLY_SEEDS)}"
        )
    if policy.get("requires_all_seed_receipts") is not True:
        raise ValueError("all seed receipts must be required")
    if policy.get("training_loss_is_criterion") is not False:
        raise ValueError("training loss cannot be an aggregate criterion")

    artifacts = _object(value["artifacts"], "artifacts")
    missing_roles = sorted(REQUIRED_ARTIFACT_ROLES - set(artifacts))
    if missing_roles:
        raise ValueError(f"plan is missing artifact roles: {missing_roles}")
    for role, raw_record in artifacts.items():
        if not isinstance(role, str) or not role:
            raise ValueError("artifact role names must be non-empty strings")
        record = _object(raw_record, f"artifacts.{role}")
        _exact_keys(
            record,
            required={"schema", "role", "path", "size_bytes", "sha256"},
            optional={"canonical_sha256"},
            location=f"artifacts.{role}",
        )
        if record.get("schema") != ARTIFACT_REF_SCHEMA:
            raise ValueError(f"artifacts.{role}.schema is invalid")
        if record.get("role") != role:
            raise ValueError(f"artifacts.{role}.role must match its object key")
        size = record.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError(f"artifacts.{role}.size_bytes must be non-negative")
        validate_sha256(str(record.get("sha256", "")), field=f"{role}.sha256")
        if "canonical_sha256" in record:
            validate_sha256(
                str(record["canonical_sha256"]),
                field=f"{role}.canonical_sha256",
            )
    return value


def _registry_active_checkpoint(registry_path: Path) -> tuple[Path, str]:
    try:
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("public registry must be UTF-8 JSON") from error
    if (
        not isinstance(registry, dict)
        or registry.get("schema") != "kev.model-registry.v1"
    ):
        raise ValueError("invalid public model registry schema")
    active = _object(registry.get("active"), "public registry active")
    active_hash = validate_sha256(
        str(active.get("sha256", "")), field="public registry active.sha256"
    )
    raw_path = active.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise ValueError("public registry active.path must be a non-empty string")
    checkpoint = Path(raw_path).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    return checkpoint.resolve(), active_hash


def _verify_trainer_discovery_closure(declared_paths: set[Path]) -> None:
    """Pin the exact file inventory consumed by trainer-side glob discovery.

    The training boundary deliberately scans every frozen manifest, public
    suite, calibration fit, known-failure audit, and held-out vocabulary. A
    plan that hashes only caller-selected files would not notice a newly added
    matching file. The trainers scan this repository directory even when the
    caller supplies custom suite paths, so closure is unconditional.
    """

    _verify_discovered_inventory(
        name="trainer-discovered input",
        discovered=_discovered_trainer_inputs(),
        declared_paths=declared_paths,
    )


def _verify_executable_source_closure(declared_paths: set[Path]) -> None:
    _verify_discovered_inventory(
        name="experiment executable source",
        discovered=_discovered_executable_sources(),
        declared_paths=declared_paths,
    )


def verify_plan_artifacts(
    plan: Mapping[str, Any], plan_path: str | Path
) -> tuple[dict[str, Path], dict[str, Any]]:
    """Hash-verify every declared input and its structural integrity."""

    validated = _validate_plan(plan)
    plan_file = Path(plan_path).expanduser().resolve()
    plan_dir = plan_file.parent
    resolved: dict[str, Path] = {}
    verified_records: dict[str, Any] = {}
    seen_paths: dict[Path, str] = {}
    for role in sorted(validated["artifacts"]):
        record = validated["artifacts"][role]
        path = _resolve_plan_path(record["path"], plan_dir, f"artifacts.{role}.path")
        if path in seen_paths:
            raise ValueError(
                f"artifact roles {seen_paths[path]!r} and {role!r} use the same path"
            )
        seen_paths[path] = role
        expected = validate_sha256(record["sha256"], field=f"{role}.sha256")
        actual = verify_file_sha256(path, expected)
        actual_size = path.stat().st_size
        if actual_size != record["size_bytes"]:
            raise ValueError(
                f"artifact size mismatch for {role}: expected {record['size_bytes']}, "
                f"got {actual_size}"
            )
        resolved[role] = path
        verified_records[role] = {
            "path": str(path),
            "size_bytes": actual_size,
            "sha256": actual,
        }

    declared_paths = set(seen_paths)
    _verify_trainer_discovery_closure(declared_paths)
    _verify_executable_source_closure(declared_paths)

    registry_checkpoint, registry_checkpoint_hash = _registry_active_checkpoint(
        resolved["public_registry"]
    )
    if resolved["public_registry"] != PUBLIC_REGISTRY:
        raise ValueError(
            "public_registry must be the repository's models/registry.json; "
            "isolated Alive state initializes from that public source of truth"
        )
    canonical_boundaries = {
        "promotion_suite": DEFAULT_SUITE,
        "calibration_fit": DEFAULT_CALIBRATION,
        "held_out_vocabulary": DEFAULT_HELD_OUT,
    }
    for role, canonical_path in canonical_boundaries.items():
        if resolved[role] != canonical_path:
            raise ValueError(
                f"{role} must be the repository's canonical preregistered artifact: "
                f"{canonical_path}"
            )
    planned_checkpoint = resolved["incumbent_checkpoint"]
    planned_checkpoint_hash = validated["artifacts"]["incumbent_checkpoint"]["sha256"]
    if registry_checkpoint != planned_checkpoint:
        raise ValueError(
            "public registry active checkpoint path differs from the pinned incumbent"
        )
    if registry_checkpoint_hash != planned_checkpoint_hash:
        raise ValueError(
            "public registry active checkpoint hash differs from the pinned incumbent"
        )
    verify_file_sha256(registry_checkpoint, registry_checkpoint_hash)

    suite = load_frozen_suite(
        resolved["promotion_suite"],
        expected_file_sha256=validated["artifacts"]["promotion_suite"]["sha256"],
        expected_canonical_sha256=validated["artifacts"]["promotion_suite"].get(
            "canonical_sha256"
        ),
    )
    verified_records["promotion_suite"]["canonical_sha256"] = suite.canonical_sha256
    encoder = verify_local_encoder_manifest(resolved["encoder_manifest"])
    if (
        encoder["manifest_sha256"]
        != validated["artifacts"]["encoder_manifest"]["sha256"]
    ):
        raise ValueError("verified encoder manifest hash differs from the plan")
    verified_records["encoder_manifest"]["bundle"] = {
        "name": encoder["name"],
        "network_policy": encoder["network_policy"],
        "license": encoder["license"],
        "source": encoder["source"],
        "files": encoder["files"],
    }
    for role in ("calibration_fit", "held_out_vocabulary", "reviewed_corpus"):
        if verified_records[role]["size_bytes"] == 0:
            raise ValueError(f"{role} cannot be empty")
    return resolved, verified_records


def _validate_output_root(plan: Mapping[str, Any], plan_path: Path) -> Path:
    output = _resolve_plan_path(plan["output_root"], plan_path.parent, "output_root")
    if output.exists():
        raise FileExistsError(f"experiment output root already exists: {output}")
    if any(part.casefold() in IGNORED_DIRECTORY_NAMES for part in output.parts):
        raise ValueError(
            "output_root cannot be nested under a directory named 'runs' or 'state'"
        )
    if output == ROOT or output == plan_path.parent:
        raise ValueError("output_root must be a new dedicated experiment directory")
    return output


def create_plan(
    *,
    plan_path: str | Path,
    experiment_id: str,
    output_root: str | Path,
    public_registry: str | Path,
    promotion_suite: str | Path,
    calibration_fit: str | Path,
    held_out_vocabulary: str | Path,
    encoder_manifest: str | Path,
    reviewed_corpus: str | Path,
    additional_artifacts: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Write a hash-pinned plan exactly once without running training."""

    destination = Path(plan_path).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"plan already exists: {destination}")
    registry_path = Path(public_registry).expanduser().resolve()
    incumbent_path, _ = _registry_active_checkpoint(registry_path)
    artifact_paths = {
        "public_registry": registry_path,
        "incumbent_checkpoint": incumbent_path,
        "promotion_suite": Path(promotion_suite).expanduser().resolve(),
        "calibration_fit": Path(calibration_fit).expanduser().resolve(),
        "held_out_vocabulary": Path(held_out_vocabulary).expanduser().resolve(),
        "encoder_manifest": Path(encoder_manifest).expanduser().resolve(),
        "reviewed_corpus": Path(reviewed_corpus).expanduser().resolve(),
    }
    for role, path in (additional_artifacts or {}).items():
        if not isinstance(role, str) or not role.strip():
            raise ValueError("additional artifact roles must be non-empty strings")
        if role in artifact_paths:
            raise ValueError(f"additional artifact role duplicates {role!r}")
        artifact_paths[role] = Path(path).expanduser().resolve()
    _add_automatic_artifacts(artifact_paths)
    artifacts = {
        role: _artifact_ref(path, role) for role, path in artifact_paths.items()
    }
    for role, path in artifact_paths.items():
        artifacts[role]["path"] = _plan_relative(path, destination.parent)
    suite = load_frozen_suite(artifact_paths["promotion_suite"])
    artifacts["promotion_suite"]["canonical_sha256"] = suite.canonical_sha256
    plan = {
        "schema": PLAN_SCHEMA,
        "frozen": True,
        "experiment_id": experiment_id,
        "output_root": _plan_relative(output_root, destination.parent),
        "artifacts": artifacts,
        "execution": {
            "ordered_seeds": list(ORDERED_SEEDS),
            "steps": TRAINING_STEPS,
            "retention_epsilon": RETENTION_EPSILON,
        },
        "runtime_environment": _runtime_environment(),
        "selection_policy": {
            "kind": "PREDECLARED_PRIMARY_ONLY",
            "primary_seed": PRIMARY_SEED,
            "robustness_only_seeds": list(ROBUSTNESS_ONLY_SEEDS),
            "requires_all_seed_receipts": True,
            "training_loss_is_criterion": False,
        },
    }
    _validate_plan(plan)
    _validate_output_root(plan, destination)
    verify_plan_artifacts(plan, destination)
    _write_exclusive_json(destination, plan)
    return {
        "schema": "kev.preregistered-experiment-plan-result.v1",
        "plan_path": str(destination),
        "plan_sha256": file_sha256(destination),
        "training_executed": False,
    }


def _evidence_role(path: Path) -> str:
    by_name = {
        "ledger.jsonl": "ledger",
        "ledger-head.json": "ledger_head",
        "state.json": "state",
        "model-registry.json": "local_model_registry",
        "eval-card.json": "eval_card",
        "training-receipt.json": "training_receipt",
        "calibration-receipt.json": "calibration_receipt",
        "evolution-result.json": "evolution_result",
        "raw-failure.json": "raw_failure",
    }
    if path.name in by_name:
        return by_name[path.name]
    if path.suffix.casefold() in {".pt", ".pth"}:
        return "candidate_checkpoint"
    return "supporting_file"


def _inventory_seed(seed_root: Path, experiment_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not seed_root.is_dir():
        return records
    for path in sorted(
        candidate for candidate in seed_root.rglob("*") if candidate.is_file()
    ):
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(experiment_root.resolve()).as_posix()
        except ValueError as error:
            raise ValueError(
                f"seed evidence escapes experiment root: {resolved}"
            ) from error
        records.append(
            {
                "role": _evidence_role(resolved),
                "path": relative,
                "size_bytes": resolved.stat().st_size,
                "sha256": file_sha256(resolved),
            }
        )
    return records


def _terminal_receipt(state_dir: Path) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if not state_dir.is_dir() or not (state_dir / "ledger.jsonl").is_file():
        return None, {"valid": False, "reason": "ledger is missing"}
    store = AliveStore(state_dir)
    verification = store.verify_ledger()
    if not verification.get("valid"):
        return None, verification
    events = [
        json.loads(line)
        for line in store.ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    terminal = [event for event in events if event.get("kind") in TERMINAL_EVENTS]
    if len(terminal) != 1:
        return None, {
            **verification,
            "valid": False,
            "reason": f"expected one terminal model event, found {len(terminal)}",
        }
    return terminal[0], verification


def _json_object_file(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    payload = path.read_bytes()
    try:
        decoded = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} must be UTF-8 JSON") from error
    value = _object(decoded, label)
    canonical_json_sha256(value)
    return value, payload


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be an array")
    return value


def _exact_evidence_path(value: Any, expected: Path, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty path")
    actual = Path(value).expanduser().resolve()
    if actual != expected.resolve():
        raise ValueError(f"{label} does not match the fixed evidence path")
    return actual


def _report_evidence_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ValueError(f"{label} must be a non-empty path")
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def _self_hash(value: Mapping[str, Any], field: str, label: str) -> str:
    claimed = validate_sha256(str(value.get(field, "")), field=f"{label}.{field}")
    body = dict(value)
    body.pop(field, None)
    actual = canonical_json_sha256(body)
    if claimed != actual:
        raise ValueError(f"{label} {field} does not match its content")
    return actual


def _load_checkpoint(path: Path, label: str) -> tuple[Mapping[str, Any], bytes]:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symbolic link")
    payload = path.read_bytes()
    try:
        checkpoint = torch.load(
            io.BytesIO(payload), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError(f"{label} cannot be deserialized safely") from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{label} must contain a mapping")
    return checkpoint, payload


def _target_for_item(item: Mapping[str, Any]) -> Any:
    for field in ("expected", "expected_frames", "target"):
        if field in item:
            return item[field]
    raise ValueError(f"evaluation item {item.get('id')!r} lacks an expected target")


def _collection(value: Any) -> list[Any] | None:
    if isinstance(value, Mapping) and isinstance(value.get("frames"), list):
        return value["frames"]
    return value if isinstance(value, list) else None


def _compare_target(expected: Any, predicted: Any, comparison: str) -> bool:
    if comparison == "set_exact":
        expected_values = _collection(expected)
        predicted_values = _collection(predicted)
        if expected_values is None or predicted_values is None:
            return False
        expected_counter = Counter(
            canonical_json_sha256(value) for value in expected_values
        )
        predicted_counter = Counter(
            canonical_json_sha256(value) for value in predicted_values
        )
        return expected_counter == predicted_counter
    if comparison != "exact":
        raise ValueError(f"unsupported evaluation comparison: {comparison}")
    return canonical_json_bytes(expected) == canonical_json_bytes(predicted)


def _validate_model_report(
    report: Mapping[str, Any],
    *,
    label: str,
    suite: Any,
    checkpoint_path: Path,
    checkpoint_sha256: str,
    calibration_status: str,
) -> str:
    if report.get("schema") != EVALUATION_SCHEMA:
        raise ValueError(f"{label} schema is invalid")
    report_hash = _self_hash(report, "report_sha256", label)
    suite_record = _mapping(report.get("suite"), f"{label}.suite")
    if (
        suite_record.get("canonical_sha256") != suite.canonical_sha256
        or suite_record.get("file_sha256") != suite.file_sha256
        or suite_record.get("frozen") is not True
    ):
        raise ValueError(f"{label} suite binding differs from the plan")

    checkpoint = _mapping(report.get("checkpoint"), f"{label}.checkpoint")
    if (
        _report_evidence_path(checkpoint.get("path"), f"{label}.checkpoint.path")
        != checkpoint_path.resolve()
    ):
        raise ValueError(f"{label} checkpoint path differs from the evidence chain")
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise ValueError(f"{label} checkpoint hash differs from the evidence chain")
    if checkpoint.get("size_bytes") != checkpoint_path.stat().st_size:
        raise ValueError(f"{label} checkpoint size differs from its bytes")
    verify_file_sha256(checkpoint_path, checkpoint_sha256)

    expected_rows: list[tuple[str, Mapping[str, Any]]] = []
    for split, items in suite.splits.items():
        expected_rows.extend((split, item) for item in items)
    records = _sequence(report.get("records"), f"{label}.records")
    if len(records) != len(expected_rows):
        raise ValueError(f"{label} record count differs from the frozen suite")

    split_counts: dict[str, list[int]] = {split: [0, 0] for split in suite.splits}
    audit_counts: dict[str, list[int]] = {}
    failures: list[dict[str, Any]] = []
    prediction_evidence: list[dict[str, Any]] = []
    for index, ((split, item), raw_row) in enumerate(zip(expected_rows, records)):
        row = _mapping(raw_row, f"{label}.records[{index}]")
        expected = _target_for_item(item)
        comparison = item.get(
            "comparison",
            "set_exact"
            if split == "composition" and _collection(expected) is not None
            else "exact",
        )
        if row.get("item_id") != item.get("id") or row.get("split") != split:
            raise ValueError(f"{label} record order/identity differs from the suite")
        if row.get("expected") != expected or row.get("comparison") != comparison:
            raise ValueError(f"{label} record target differs from the suite")
        for optional in ("audit", "pair_id"):
            if row.get(optional) != item.get(optional):
                raise ValueError(f"{label} record {optional} differs from the suite")
        confidence = row.get("confidence")
        if confidence is not None and (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not math.isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            raise ValueError(f"{label} record confidence is invalid")
        correct = "error" not in row and _compare_target(
            expected, row.get("predicted"), str(comparison)
        )
        if row.get("correct") is not correct:
            raise ValueError(f"{label} record correctness is not reproducible")
        split_counts[split][0] += 1
        split_counts[split][1] += int(correct)
        if item.get("audit") is not None:
            audit = str(item["audit"])
            counts = audit_counts.setdefault(audit, [0, 0])
            counts[0] += 1
            counts[1] += int(correct)
        prediction = {
            "item_id": item["id"],
            "split": split,
            "predicted": row.get("predicted"),
            "confidence": confidence,
        }
        if "error" in row:
            prediction["error"] = row["error"]
        prediction_evidence.append(prediction)
        if not correct:
            failure = {
                "item_id": item["id"],
                "split": split,
                "input": item.get("input", item.get("text")),
                "expected": expected,
                "predicted": row.get("predicted"),
                "comparison": comparison,
            }
            for optional in ("audit", "pair_id"):
                if item.get(optional) is not None:
                    failure[optional] = item[optional]
            if "error" in row:
                failure["error"] = row["error"]
            failures.append(failure)

    if report.get("prediction_evidence_sha256") != canonical_json_sha256(
        prediction_evidence
    ):
        raise ValueError(f"{label} prediction evidence hash is invalid")
    split_metrics = _mapping(report.get("splits"), f"{label}.splits")
    if set(split_metrics) != set(split_counts):
        raise ValueError(f"{label} split metric inventory differs from the suite")
    for split, (total, correct_count) in split_counts.items():
        metric = _mapping(split_metrics[split], f"{label}.splits.{split}")
        expected_score = correct_count / total
        if (
            metric.get("total") != total
            or metric.get("correct") != correct_count
            or metric.get("failures") != total - correct_count
            or metric.get("primary_metric") != "exact_match_accuracy"
            or metric.get("primary_score") != expected_score
            or metric.get("accuracy") != expected_score
        ):
            raise ValueError(f"{label} {split} metrics do not match its records")
    audit_metrics = _mapping(report.get("audit_metrics"), f"{label}.audit_metrics")
    if set(audit_metrics) != set(audit_counts):
        raise ValueError(f"{label} audit metric inventory differs from its records")
    for audit, (total, correct_count) in audit_counts.items():
        metric = _mapping(audit_metrics[audit], f"{label}.audit_metrics.{audit}")
        if dict(metric) != {
            "total": total,
            "correct": correct_count,
            "failures": total - correct_count,
            "accuracy": correct_count / total,
        }:
            raise ValueError(f"{label} audit metrics do not match its records")
    raw_failures = _mapping(report.get("raw_failures"), f"{label}.raw_failures")
    if (
        raw_failures.get("available") is not True
        or raw_failures.get("count") != len(failures)
        or raw_failures.get("items") != failures
    ):
        raise ValueError(f"{label} raw failures do not match its records")
    calibration = _mapping(report.get("calibration"), f"{label}.calibration")
    if calibration.get("status") != calibration_status:
        raise ValueError(f"{label} calibration status is invalid")
    return report_hash


def _validate_training_evidence(
    *,
    receipt_path: Path,
    raw_checkpoint_path: Path,
    seed: int,
    resolved: Mapping[str, Path],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    receipt, receipt_bytes = _json_object_file(receipt_path, "training receipt")
    if receipt.get("schema") != "kev.training-receipt.v1":
        raise ValueError("training receipt schema is invalid")
    requested_terms = {
        line.strip().casefold()
        for line in resolved["held_out_vocabulary"]
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    required_terms, required_artifacts, required_artifacts_sha256 = (
        required_held_out_vocabulary()
    )
    required_terms_sorted = sorted(required_terms)
    full_terms = sorted(required_terms | requested_terms)
    required_terms_sha256 = canonical_json_sha256(required_terms_sorted)
    full_terms_sha256 = canonical_json_sha256(full_terms)
    _, exclusion_artifacts = _frozen_evaluation_exclusions()
    exclusion_hash = canonical_json_sha256(
        sorted(
            (
                {"role": artifact["role"], "sha256": artifact["sha256"]}
                for artifact in exclusion_artifacts
            ),
            key=lambda item: (item["role"], item["sha256"]),
        )
    )
    source_manifest = _current_training_source_manifest()
    source_manifest_sha256 = canonical_json_sha256(source_manifest)
    source_hashes = {entry["role"]: entry["sha256"] for entry in source_manifest}
    expected_values = {
        "artifact_status": "CHALLENGER",
        "activation": "NONE",
        "model_family": "FROZEN_SENTENCE_ENCODER_PROPOSAL",
        "parent_sha256": plan["artifacts"]["incumbent_checkpoint"]["sha256"],
        "lessons_sha256": plan["artifacts"]["reviewed_corpus"]["sha256"],
        "seed": seed,
        "steps": TRAINING_STEPS,
        "objective": TRAINING_OBJECTIVE,
        "score_status": "UNCALIBRATED",
        "promotion_features": [],
        "held_out_vocabulary": full_terms,
        "held_out_vocabulary_sha256": full_terms_sha256,
        "required_held_out_vocabulary": required_terms_sorted,
        "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
        "required_held_out_vocabulary_artifacts": required_artifacts,
        "required_held_out_vocabulary_artifacts_sha256": (required_artifacts_sha256),
        "pinned_held_out_vocabulary_file_sha256": file_sha256(
            PINNED_HELD_OUT_VOCABULARY
        ),
        "training_exclusion_artifacts": exclusion_artifacts,
        "training_exclusions_sha256": exclusion_hash,
        "trainer_source_sha256": source_hashes["FROZEN_ENCODER_TRAINER"],
        "encoder_runtime_source_sha256": source_hashes["ENCODER_RUNTIME"],
        "training_contract_source_sha256": source_hashes["SHARED_SEMANTIC_TRAINING"],
        "training_source_manifest": source_manifest,
        "training_source_manifest_sha256": source_manifest_sha256,
    }
    for field, expected in expected_values.items():
        if receipt.get(field) != expected:
            raise ValueError(f"training receipt {field} differs from the plan")
    _exact_evidence_path(
        receipt.get("parent_path"), resolved["incumbent_checkpoint"], "parent_path"
    )
    _exact_evidence_path(
        receipt.get("lessons_path"), resolved["reviewed_corpus"], "lessons_path"
    )
    _exact_evidence_path(
        receipt.get("challenger_path"), raw_checkpoint_path, "challenger_path"
    )
    raw_checkpoint_hash = verify_file_sha256(
        raw_checkpoint_path,
        validate_sha256(
            str(receipt.get("challenger_sha256", "")),
            field="training receipt challenger_sha256",
        ),
    )
    loss_history = _sequence(receipt.get("loss_history"), "loss_history")
    if len(loss_history) != TRAINING_STEPS or any(
        not isinstance(row, Mapping) or row.get("step") != index
        for index, row in enumerate(loss_history, 1)
    ):
        raise ValueError("training receipt loss history is not contiguous")
    receipt_environment = _mapping(receipt.get("environment"), "training environment")
    expected_environment = {
        "python": plan["runtime_environment"]["python"],
        "torch": plan["runtime_environment"]["packages"]["torch"],
        "transformers": plan["runtime_environment"]["packages"]["transformers"],
        "tokenizers": plan["runtime_environment"]["packages"]["tokenizers"],
        "safetensors": plan["runtime_environment"]["packages"]["safetensors"],
    }
    if dict(receipt_environment) != expected_environment:
        raise ValueError("training receipt environment differs from the plan")

    training_inputs = _mapping(receipt.get("training_inputs"), "training_inputs")
    if receipt.get("training_inputs_sha256") != canonical_json_sha256(training_inputs):
        raise ValueError("training_inputs hash does not match its content")

    embedding_reference = _mapping(
        receipt.get("frozen_embedding_cache"), "frozen_embedding_cache"
    )
    embedding_path = receipt_path.parent / "frozen-embeddings.pt"
    _exact_evidence_path(
        embedding_reference.get("path"), embedding_path, "frozen_embedding_cache.path"
    )
    embedding_file_sha256 = validate_sha256(
        str(embedding_reference.get("file_sha256", "")),
        field="frozen_embedding_cache.file_sha256",
    )
    verify_file_sha256(embedding_path, embedding_file_sha256)
    embedding_cache, _ = _load_checkpoint(embedding_path, "frozen embedding cache")
    embeddings = embedding_cache.get("embeddings")
    if not isinstance(embeddings, torch.Tensor) or embeddings.dtype != torch.float32:
        raise ValueError("frozen embedding cache requires a float32 tensor")
    embeddings = embeddings.detach().cpu().contiguous()
    embedding_body = (
        b"kev.frozen-embeddings.v1\0float32-le\0"
        + json.dumps(list(embeddings.shape), separators=(",", ":")).encode("ascii")
        + b"\0"
        + embeddings.numpy().astype("<f4", copy=False).tobytes()
    )
    embedding_values_sha256 = hashlib.sha256(embedding_body).hexdigest()
    expected_embedding_cache = {
        "schema": "kev.frozen-embeddings.v1",
        "encoder_manifest_sha256": plan["artifacts"]["encoder_manifest"]["sha256"],
        "lessons_sha256": plan["artifacts"]["reviewed_corpus"]["sha256"],
        "shape": list(embeddings.shape),
        "dtype": "float32-le",
        "values_sha256": embedding_values_sha256,
    }
    for field, expected in expected_embedding_cache.items():
        if embedding_cache.get(field) != expected:
            raise ValueError(f"frozen embedding cache {field} is invalid")
    for field, expected in {
        "values_sha256": embedding_values_sha256,
        "shape": list(embeddings.shape),
    }.items():
        if embedding_reference.get(field) != expected:
            raise ValueError(f"frozen_embedding_cache {field} is invalid")
    batch_size = embedding_reference.get("encoder_batch_size")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
    ):
        raise ValueError("frozen_embedding_cache encoder_batch_size is invalid")
    required_inputs = {
        "parent_sha256": plan["artifacts"]["incumbent_checkpoint"]["sha256"],
        "lessons_sha256": plan["artifacts"]["reviewed_corpus"]["sha256"],
        "held_out_vocabulary_sha256": full_terms_sha256,
        "pinned_held_out_vocabulary_file_sha256": file_sha256(
            PINNED_HELD_OUT_VOCABULARY
        ),
        "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
        "required_held_out_vocabulary_artifacts_sha256": (required_artifacts_sha256),
        "training_exclusions_sha256": exclusion_hash,
        "frozen_embedding_values_sha256": embedding_values_sha256,
        "encoder_manifest_sha256": plan["artifacts"]["encoder_manifest"]["sha256"],
        "trainer_source_sha256": source_hashes["FROZEN_ENCODER_TRAINER"],
        "encoder_runtime_source_sha256": source_hashes["ENCODER_RUNTIME"],
        "training_contract_source_sha256": source_hashes["SHARED_SEMANTIC_TRAINING"],
        "training_source_manifest": source_manifest,
        "training_source_manifest_sha256": source_manifest_sha256,
        "seed": seed,
        "steps": TRAINING_STEPS,
        "objective": TRAINING_OBJECTIVE,
    }
    for field, expected in required_inputs.items():
        if training_inputs.get(field) != expected:
            raise ValueError(f"training_inputs {field} differs from the plan")

    checkpoint, _ = _load_checkpoint(raw_checkpoint_path, "raw challenger")
    if checkpoint.get("schema") != FROZEN_SENTENCE_CHECKPOINT_SCHEMA:
        raise ValueError("raw challenger schema is invalid")
    expected_scalar_sources = {
        "trainer_source_sha256": source_hashes["FROZEN_ENCODER_TRAINER"],
        "encoder_runtime_source_sha256": source_hashes["ENCODER_RUNTIME"],
        "training_contract_source_sha256": source_hashes["SHARED_SEMANTIC_TRAINING"],
    }
    claimed_source_fields = {
        field for field in training_inputs if field.endswith("_source_sha256")
    }
    if claimed_source_fields != set(expected_scalar_sources):
        raise ValueError(
            "training_inputs source hash inventory is incomplete or unknown: "
            f"{sorted(claimed_source_fields)}"
        )
    for field, expected_source_hash in expected_scalar_sources.items():
        if (
            training_inputs.get(field) != expected_source_hash
            or receipt.get(field) != expected_source_hash
            or checkpoint.get(field) != expected_source_hash
        ):
            raise ValueError(f"training source lineage field {field} is invalid")
    for field, expected in {
        "training_source_manifest": source_manifest,
        "training_source_manifest_sha256": source_manifest_sha256,
    }.items():
        if (
            training_inputs.get(field) != expected
            or receipt.get(field) != expected
            or checkpoint.get(field) != expected
        ):
            raise ValueError(f"training source lineage field {field} is invalid")
    shared = (
        "model_family",
        "parent_sha256",
        "lessons_sha256",
        "held_out_vocabulary_sha256",
        "pinned_held_out_vocabulary_file_sha256",
        "required_held_out_vocabulary_terms_sha256",
        "required_held_out_vocabulary_artifacts",
        "required_held_out_vocabulary_artifacts_sha256",
        "training_exclusions_sha256",
        "paraphrase_groups_sha256",
        "contrastive_positive_pairs",
        "trainer_source_sha256",
        "encoder_runtime_source_sha256",
        "training_contract_source_sha256",
        "training_source_manifest",
        "training_source_manifest_sha256",
        "training_inputs_sha256",
        "training_scope",
        "seed",
        "steps",
        "objective",
        "optimizer",
        "torch_num_threads",
        "score_status",
        "artifact_status",
    )
    for field in shared:
        if field not in receipt or checkpoint.get(field) != receipt[field]:
            raise ValueError(f"raw challenger {field} differs from its receipt")
    if checkpoint.get("temperature") is not None:
        raise ValueError("raw challenger must remain uncalibrated")
    if checkpoint.get("frozen_embedding_values_sha256") != embedding_values_sha256:
        raise ValueError("raw challenger frozen embedding hash is invalid")
    encoder_artifact = _mapping(receipt.get("encoder_artifact"), "encoder_artifact")
    if (
        encoder_artifact.get("sha256")
        != plan["artifacts"]["encoder_manifest"]["sha256"]
    ):
        raise ValueError("training receipt encoder hash differs from the plan")
    checkpoint_encoder = _mapping(
        checkpoint.get("encoder_manifest"), "checkpoint encoder_manifest"
    )
    if checkpoint_encoder.get("sha256") != encoder_artifact.get("sha256"):
        raise ValueError("raw challenger encoder binding differs from its receipt")

    non_seed_inputs = dict(training_inputs)
    non_seed_inputs.pop("seed", None)
    return {
        "receipt": receipt,
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "raw_checkpoint_path": raw_checkpoint_path,
        "raw_checkpoint_sha256": raw_checkpoint_hash,
        "training_inputs_sha256": receipt["training_inputs_sha256"],
        "non_seed_training_inputs_sha256": canonical_json_sha256(non_seed_inputs),
    }


def _validate_eval_card(
    *,
    eval_card_path: Path,
    incumbent_path: Path,
    incumbent_sha256: str,
    challenger_path: Path,
    challenger_sha256: str,
    calibration_receipt_path: Path,
    calibration_receipt_sha256: str,
    calibration_receipt_canonical_sha256: str,
    suite: Any,
) -> tuple[dict[str, Any], str]:
    report, payload = _json_object_file(eval_card_path, "evaluation card")
    if report.get("schema") != CHALLENGER_REPORT_SCHEMA:
        raise ValueError("evaluation card schema is invalid")
    _self_hash(report, "report_sha256", "evaluation card")
    if report.get("suite_canonical_sha256") != suite.canonical_sha256:
        raise ValueError("evaluation card suite binding differs from the plan")
    incumbent = _mapping(report.get("incumbent"), "evaluation card incumbent")
    challenger = _mapping(report.get("challenger"), "evaluation card challenger")
    incumbent_report_sha256 = _validate_model_report(
        incumbent,
        label="incumbent evaluation",
        suite=suite,
        checkpoint_path=incumbent_path,
        checkpoint_sha256=incumbent_sha256,
        calibration_status="UNCALIBRATED",
    )
    challenger_report_sha256 = _validate_model_report(
        challenger,
        label="challenger evaluation",
        suite=suite,
        checkpoint_path=challenger_path,
        checkpoint_sha256=challenger_sha256,
        calibration_status="CALIBRATED",
    )
    challenger_calibration = _mapping(
        challenger.get("calibration"), "challenger evaluation calibration"
    )
    if (
        challenger_calibration.get("temperature_artifact_file_sha256")
        != calibration_receipt_sha256
        or challenger_calibration.get("temperature_artifact_canonical_sha256")
        != calibration_receipt_canonical_sha256
    ):
        raise ValueError("evaluation card calibration evidence is not receipt-bound")

    decision = _mapping(report.get("decision"), "evaluation card decision")
    if decision.get("schema") != COMPARISON_SCHEMA:
        raise ValueError("evaluation card decision schema is invalid")
    _self_hash(decision, "decision_sha256", "evaluation card decision")
    reproduced = compare_evaluations(
        incumbent, challenger, retention_epsilon=RETENTION_EPSILON
    )
    if dict(decision) != reproduced:
        raise ValueError("evaluation decision is not reproducible from its reports")
    evidence_hashes = _mapping(
        decision.get("evidence_hashes"), "evaluation decision evidence_hashes"
    )
    if (
        evidence_hashes.get("incumbent_report_sha256") != incumbent_report_sha256
        or evidence_hashes.get("challenger_report_sha256") != challenger_report_sha256
    ):
        raise ValueError("evaluation decision report hashes are invalid")
    if decision.get("policy", {}).get("training_loss_used") is not False:
        raise ValueError("evaluation decision cannot use training loss")
    raw_failures = _mapping(report.get("raw_failures"), "evaluation raw_failures")
    if raw_failures.get("incumbent") != incumbent.get(
        "raw_failures"
    ) or raw_failures.get("challenger") != challenger.get("raw_failures"):
        raise ValueError("evaluation card raw failures are internally inconsistent")
    incumbent_predictor = CheckpointPredictor(incumbent_path)
    challenger_predictor = CheckpointPredictor(challenger_path)
    if incumbent_predictor.sha256 != incumbent_sha256:
        raise ValueError("replay incumbent predictor loaded different checkpoint bytes")
    if challenger_predictor.sha256 != challenger_sha256:
        raise ValueError(
            "replay challenger predictor loaded different checkpoint bytes"
        )
    replayed = evaluate_challenger(
        suite,
        incumbent_predictor=incumbent_predictor,
        challenger_predictor=challenger_predictor,
        incumbent_path=incumbent_path,
        challenger_path=challenger_path,
        incumbent_sha256=incumbent_sha256,
        challenger_sha256=challenger_sha256,
        challenger_temperature=calibration_receipt_path,
        retention_epsilon=RETENTION_EPSILON,
    )
    if replayed != report:
        raise ValueError("evaluation card does not match independent checkpoint replay")
    verify_file_sha256(incumbent_path, incumbent_sha256)
    verify_file_sha256(challenger_path, challenger_sha256)
    verify_file_sha256(calibration_receipt_path, calibration_receipt_sha256)
    return report, hashlib.sha256(payload).hexdigest()


def _validate_seed_evidence(
    *,
    seed: int,
    seed_root: Path,
    result: Mapping[str, Any],
    terminal: Mapping[str, Any],
    resolved: Mapping[str, Path],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    data = _mapping(terminal.get("data"), "terminal ledger event data")
    run_id = data.get("run_id")
    if (
        not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
        or "/" in run_id
        or "\\" in run_id
    ):
        raise ValueError("terminal ledger event run_id is unsafe")
    run_dir = (seed_root / "candidates" / run_id).resolve()
    run_dir.relative_to(seed_root.resolve())
    if not run_dir.is_dir() or run_dir.is_symlink():
        raise ValueError("terminal run directory is missing or unsafe")

    incumbent = _mapping(data.get("incumbent"), "terminal incumbent")
    incumbent_sha256 = plan["artifacts"]["incumbent_checkpoint"]["sha256"]
    _exact_evidence_path(
        incumbent.get("path"), resolved["incumbent_checkpoint"], "terminal incumbent"
    )
    if incumbent.get("sha256") != incumbent_sha256:
        raise ValueError("terminal incumbent hash differs from the plan")
    incumbent_generation = incumbent.get("generation")
    if (
        isinstance(incumbent_generation, bool)
        or not isinstance(incumbent_generation, int)
        or incumbent_generation != 0
    ):
        raise ValueError("isolated seed incumbent generation must be zero")
    if incumbent.get("source") not in {"state", "public-registry"}:
        raise ValueError("terminal incumbent source is invalid")
    verify_file_sha256(resolved["incumbent_checkpoint"], incumbent_sha256)

    expected_decision = (
        "QUALIFY" if terminal.get("kind") == "MODEL_QUALIFIED" else "REJECT"
    )
    challenger = _mapping(data.get("challenger"), "terminal challenger")
    calibrated_checkpoint = run_dir / "calibrated" / "semantic-breadth.calibrated.pt"
    calibrated_sha256 = validate_sha256(
        str(challenger.get("sha256", "")), field="terminal challenger.sha256"
    )
    _exact_evidence_path(
        challenger.get("path"), calibrated_checkpoint, "terminal challenger.path"
    )
    verify_file_sha256(calibrated_checkpoint, calibrated_sha256)
    if (
        challenger.get("parent_sha256") != incumbent_sha256
        or challenger.get("score_status") != "CALIBRATED"
    ):
        raise ValueError("terminal challenger lineage/status is invalid")

    training_receipt_path = run_dir / "trained" / "training-receipt.json"
    raw_checkpoint_path = run_dir / "trained" / "semantic-breadth.pt"
    _exact_evidence_path(
        data.get("training_receipt_path"),
        training_receipt_path,
        "terminal training_receipt_path",
    )
    training = _validate_training_evidence(
        receipt_path=training_receipt_path,
        raw_checkpoint_path=raw_checkpoint_path,
        seed=seed,
        resolved=resolved,
        plan=plan,
    )
    if data.get("training_receipt_sha256") != training["receipt_sha256"]:
        raise ValueError("terminal training receipt hash differs from its bytes")

    calibration_receipt_path = run_dir / "calibrated" / "calibration-receipt.json"
    _exact_evidence_path(
        data.get("calibration_receipt_path"),
        calibration_receipt_path,
        "terminal calibration_receipt_path",
    )
    calibration_receipt, calibration_receipt_bytes = _json_object_file(
        calibration_receipt_path, "calibration receipt"
    )
    _exact_evidence_path(
        calibration_receipt.get("source_path"),
        raw_checkpoint_path,
        "calibration source_path",
    )
    _exact_evidence_path(
        calibration_receipt.get("output_path"),
        calibrated_checkpoint,
        "calibration output_path",
    )
    calibrated_checkpoint_bytes = calibrated_checkpoint.read_bytes()
    calibration_artifact = validate_calibration_binding(
        calibration_receipt_path,
        checkpoint_bytes=calibrated_checkpoint_bytes,
        promotion_suite_file_sha256=plan["artifacts"]["promotion_suite"]["sha256"],
        promotion_suite_canonical_sha256=plan["artifacts"]["promotion_suite"][
            "canonical_sha256"
        ],
        source_checkpoint_path=raw_checkpoint_path,
        calibration_fit_path=resolved["calibration_fit"],
        promotion_suite_path=resolved["promotion_suite"],
    )
    calibration_receipt_sha256 = hashlib.sha256(calibration_receipt_bytes).hexdigest()
    if data.get("calibration_receipt_sha256") != calibration_receipt_sha256:
        raise ValueError("terminal calibration receipt hash differs from its bytes")

    suite = load_frozen_suite(
        resolved["promotion_suite"],
        expected_file_sha256=plan["artifacts"]["promotion_suite"]["sha256"],
        expected_canonical_sha256=plan["artifacts"]["promotion_suite"].get(
            "canonical_sha256"
        ),
    )
    eval_card_path = run_dir / "eval-card.json"
    _exact_evidence_path(
        data.get("eval_card_path"), eval_card_path, "terminal eval_card_path"
    )
    _exact_evidence_path(
        challenger.get("eval_card"), eval_card_path, "challenger eval_card"
    )
    eval_card, eval_card_sha256 = _validate_eval_card(
        eval_card_path=eval_card_path,
        incumbent_path=resolved["incumbent_checkpoint"],
        incumbent_sha256=incumbent_sha256,
        challenger_path=calibrated_checkpoint,
        challenger_sha256=calibrated_sha256,
        calibration_receipt_path=calibration_receipt_path,
        calibration_receipt_sha256=calibration_receipt_sha256,
        calibration_receipt_canonical_sha256=calibration_artifact.canonical_sha256,
        suite=suite,
    )
    if (
        data.get("eval_card_sha256") != eval_card_sha256
        or challenger.get("eval_card_sha256") != eval_card_sha256
    ):
        raise ValueError("terminal eval-card hash differs from its bytes")
    decision = _mapping(eval_card.get("decision"), "evaluation card decision")
    if (
        decision.get("decision") != expected_decision
        or data.get("decision_sha256") != decision.get("decision_sha256")
        or data.get("reason_codes") != decision.get("reason_codes")
    ):
        raise ValueError("terminal decision differs from the evaluation card")
    if data.get("suite_sha256") != suite.canonical_sha256:
        raise ValueError("terminal suite hash differs from the plan")
    if data.get("training_loss_used_for_promotion") is not False:
        raise ValueError("terminal decision cannot use training loss")

    inputs = _mapping(data.get("input_artifacts"), "terminal input_artifacts")
    expected_inputs = {
        "lessons_sha256": plan["artifacts"]["reviewed_corpus"]["sha256"],
        "held_out_vocabulary_sha256": plan["artifacts"]["held_out_vocabulary"][
            "sha256"
        ],
        "calibration_fit_sha256": plan["artifacts"]["calibration_fit"]["sha256"],
        "promotion_suite_file_sha256": plan["artifacts"]["promotion_suite"]["sha256"],
        "promotion_suite_canonical_sha256": plan["artifacts"]["promotion_suite"][
            "canonical_sha256"
        ],
        "encoder_manifest_sha256": plan["artifacts"]["encoder_manifest"]["sha256"],
        "held_out_vocabulary_terms_sha256": training["receipt"].get(
            "held_out_vocabulary_sha256"
        ),
        "required_held_out_vocabulary_terms_sha256": training["receipt"].get(
            "required_held_out_vocabulary_terms_sha256"
        ),
        "required_held_out_vocabulary_artifacts_sha256": training["receipt"].get(
            "required_held_out_vocabulary_artifacts_sha256"
        ),
    }
    if dict(inputs) != expected_inputs:
        raise ValueError("terminal input_artifacts differ from the verified inputs")

    result_path = run_dir / "evolution-result.json"
    persisted_result, _ = _json_object_file(result_path, "evolution result")
    if dict(result) != persisted_result:
        raise ValueError("returned evolution result differs from persisted evidence")
    if persisted_result.get("schema") != "kev.evolution-result.v1":
        raise ValueError("evolution result schema is invalid")
    _self_hash(persisted_result, "result_sha256", "evolution result")
    if (
        persisted_result.get("run_id") != run_id
        or persisted_result.get("decision") != expected_decision
        or persisted_result.get("incumbent") != incumbent
        or persisted_result.get("challenger") != challenger
        or persisted_result.get("eval_card")
        != {"path": str(eval_card_path), "sha256": eval_card_sha256}
        or persisted_result.get("ledger_event") != terminal.get("hash")
        or persisted_result.get("candidate_preserved") is not True
        or persisted_result.get("old_incumbent_preserved") is not True
        or persisted_result.get("training_loss_used_for_promotion") is not False
    ):
        raise ValueError("evolution result does not match the terminal evidence chain")

    state, _ = _json_object_file(seed_root / "run-state" / "state.json", "state")
    expected_active_path = (
        calibrated_checkpoint
        if expected_decision == "QUALIFY"
        else resolved["incumbent_checkpoint"]
    )
    expected_active_hash = (
        calibrated_sha256 if expected_decision == "QUALIFY" else incumbent_sha256
    )
    expected_generation = incumbent_generation + int(expected_decision == "QUALIFY")
    if (
        Path(str(state.get("semantic_model", ""))).resolve()
        != expected_active_path.resolve()
        or state.get("semantic_model_sha256") != expected_active_hash
        or state.get("semantic_model_generation") != expected_generation
    ):
        raise ValueError("state active-model pointer disagrees with the decision")
    model_history = _sequence(state.get("model_history"), "state model_history")
    if expected_decision == "QUALIFY":
        expected_history = {
            "path": str(resolved["incumbent_checkpoint"]),
            "sha256": incumbent_sha256,
            "generation": incumbent_generation,
        }
        if not model_history or model_history[-1] != expected_history:
            raise ValueError("qualified state lacks exact incumbent history")
    elif model_history:
        raise ValueError("rejected isolated seed unexpectedly changed model history")
    local_registry_value = persisted_result.get("local_registry")
    if expected_decision == "QUALIFY":
        local_registry_path = seed_root / "run-state" / "model-registry.json"
        _exact_evidence_path(
            local_registry_value, local_registry_path, "evolution local_registry"
        )
        local_registry, _ = _json_object_file(
            local_registry_path, "qualified local model registry"
        )
        active = _mapping(local_registry.get("active"), "local registry active")
        if (
            Path(str(active.get("path", ""))).resolve()
            != calibrated_checkpoint.resolve()
            or active.get("sha256") != calibrated_sha256
            or local_registry.get("last_eval_card_sha256") != eval_card_sha256
        ):
            raise ValueError("qualified local registry is not bound to the evidence")
    elif local_registry_value is not None:
        raise ValueError("rejected evolution result cannot claim a local registry")

    return {
        "run_id": run_id,
        "decision": expected_decision,
        "challenger": {
            "path": str(calibrated_checkpoint),
            "sha256": calibrated_sha256,
        },
        "training_receipt_sha256": training["receipt_sha256"],
        "training_inputs_sha256": training["training_inputs_sha256"],
        "non_seed_training_inputs_sha256": training["non_seed_training_inputs_sha256"],
        "calibration_receipt_sha256": calibration_receipt_sha256,
        "calibration_receipt_canonical_sha256": (calibration_artifact.canonical_sha256),
        "eval_card_sha256": eval_card_sha256,
        "eval_card_report_sha256": eval_card["report_sha256"],
        "evolution_result_sha256": file_sha256(result_path),
        "terminal_event_sha256": terminal.get("hash"),
    }


def _seed_record(
    *,
    seed: int,
    role: str,
    seed_root: Path,
    experiment_root: Path,
    result: Mapping[str, Any] | None,
    error: Exception | None,
    resolved: Mapping[str, Path],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    state_dir = seed_root / "run-state"
    terminal, ledger_verification = _terminal_receipt(state_dir)
    evidence_files = _inventory_seed(seed_root, experiment_root)
    integrity_errors: list[str] = []
    evidence_validation: dict[str, Any] | None = None
    if terminal is None:
        integrity_errors.append("authoritative terminal ledger receipt is unavailable")
    if result is None:
        integrity_errors.append("evolution result is unavailable")
    if terminal is not None and result is not None:
        try:
            evidence_validation = _validate_seed_evidence(
                seed=seed,
                seed_root=seed_root,
                result=result,
                terminal=terminal,
                resolved=resolved,
                plan=plan,
            )
        except Exception as validation_error:
            integrity_errors.append(
                "evidence chain validation failed: "
                f"{type(validation_error).__name__}: {validation_error}"
            )
    roles = {record["role"] for record in evidence_files}
    required_roles = {"ledger", "state"}
    if result is not None:
        required_roles |= {
            "eval_card",
            "training_receipt",
            "calibration_receipt",
            "evolution_result",
            "candidate_checkpoint",
        }
    if error is not None:
        required_roles.add("raw_failure")
    missing_evidence = sorted(required_roles - roles)
    if missing_evidence:
        integrity_errors.append(f"missing evidence roles: {missing_evidence}")
    authoritative = None
    receipt = None
    if terminal is not None:
        authoritative = "QUALIFY" if terminal["kind"] == "MODEL_QUALIFIED" else "REJECT"
        receipt = {
            "kind": terminal["kind"],
            "sequence": terminal.get("sequence"),
            "event_hash": terminal.get("hash"),
        }
    complete = (
        error is None
        and result is not None
        and terminal is not None
        and ledger_verification.get("valid") is True
        and evidence_validation is not None
        and not integrity_errors
    )
    return {
        "seed": seed,
        "role": role,
        "promotion_recommendation_eligible": seed == PRIMARY_SEED,
        "paths": {
            "seed_root": seed_root.relative_to(experiment_root).as_posix(),
            "state_dir": state_dir.relative_to(experiment_root).as_posix(),
            "candidate_root": (seed_root / "candidates")
            .relative_to(experiment_root)
            .as_posix(),
        },
        "execution_status": "RETURNED" if error is None else "RAISED",
        "execution_error": (
            None
            if error is None
            else {"type": type(error).__name__, "message": str(error)}
        ),
        "authoritative_decision": authoritative,
        "terminal_ledger_receipt": receipt,
        "ledger_verification": ledger_verification,
        "evidence_integrity": "COMPLETE" if complete else "INCOMPLETE",
        "integrity_errors": integrity_errors,
        "run_id": (
            evidence_validation.get("run_id")
            if evidence_validation is not None
            else result.get("run_id")
            if result is not None
            else None
        ),
        "content_chain": evidence_validation,
        "evidence_files": evidence_files,
    }


def _not_run_record(seed: int, role: str, reason: str) -> dict[str, Any]:
    return {
        "seed": seed,
        "role": role,
        "promotion_recommendation_eligible": seed == PRIMARY_SEED,
        "execution_status": "NOT_RUN",
        "execution_error": {"type": "InputIntegrityError", "message": reason},
        "authoritative_decision": None,
        "terminal_ledger_receipt": None,
        "ledger_verification": {"valid": False, "reason": "not run"},
        "evidence_integrity": "INCOMPLETE",
        "integrity_errors": ["seed was not run after input integrity failed"],
        "run_id": None,
        "content_chain": None,
        "evidence_files": [],
    }


def _final_revalidate_returned_seeds(
    seed_records: list[dict[str, Any]],
    *,
    experiment_root: Path,
    resolved: Mapping[str, Path],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Re-read every returned seed immediately before aggregate decisions."""

    results: dict[str, dict[str, Any]] = {}
    returned_count = 0
    for record in seed_records:
        seed = int(record["seed"])
        if record.get("execution_status") != "RETURNED":
            record["final_revalidation"] = {
                "performed": False,
                "valid": False,
                "reason": "seed execution did not return",
            }
            results[str(seed)] = dict(record["final_revalidation"])
            continue
        returned_count += 1

        seed_root = experiment_root / "seed-results" / f"seed-{seed}"
        state_dir = seed_root / "run-state"
        initial_inventory = list(record["evidence_files"])
        fresh_inventory = _inventory_seed(seed_root, experiment_root)
        record["evidence_files"] = fresh_inventory
        record["content_chain"] = None
        valid = False
        error_message: str | None = None
        try:
            terminal, ledger_verification = _terminal_receipt(state_dir)
            record["ledger_verification"] = ledger_verification
            if terminal is None or ledger_verification.get("valid") is not True:
                raise ValueError("fresh terminal ledger receipt is unavailable")
            data = _mapping(terminal.get("data"), "fresh terminal ledger event data")
            run_id = data.get("run_id")
            if (
                not isinstance(run_id, str)
                or not run_id
                or Path(run_id).name != run_id
                or "/" in run_id
                or "\\" in run_id
            ):
                raise ValueError("fresh terminal ledger run_id is unsafe")
            result_path = seed_root / "candidates" / run_id / "evolution-result.json"
            persisted_result, _ = _json_object_file(
                result_path, "fresh evolution result"
            )
            validation = _validate_seed_evidence(
                seed=seed,
                seed_root=seed_root,
                result=persisted_result,
                terminal=terminal,
                resolved=resolved,
                plan=plan,
            )
            if fresh_inventory != initial_inventory:
                initial_by_path = {
                    str(item["path"]): item for item in initial_inventory
                }
                fresh_by_path = {str(item["path"]): item for item in fresh_inventory}
                added = sorted(set(fresh_by_path) - set(initial_by_path))
                removed = sorted(set(initial_by_path) - set(fresh_by_path))
                changed = sorted(
                    path
                    for path in set(initial_by_path) & set(fresh_by_path)
                    if initial_by_path[path] != fresh_by_path[path]
                )
                raise ValueError(
                    "fresh evidence inventory differs from its initial snapshot: "
                    f"added={added}, removed={removed}, changed={changed}"
                )
            record["content_chain"] = validation
            record["run_id"] = validation["run_id"]
            record["authoritative_decision"] = validation["decision"]
            record["terminal_ledger_receipt"] = {
                "kind": terminal["kind"],
                "sequence": terminal.get("sequence"),
                "event_hash": terminal.get("hash"),
            }
            valid = not record["integrity_errors"]
        except Exception as validation_error:
            error_message = (
                "final evidence revalidation failed: "
                f"{type(validation_error).__name__}: {validation_error}"
            )
            if error_message not in record["integrity_errors"]:
                record["integrity_errors"].append(error_message)

        record["evidence_integrity"] = "COMPLETE" if valid else "INCOMPLETE"
        record["final_revalidation"] = {
            "performed": True,
            "valid": valid,
            "error": error_message,
        }
        results[str(seed)] = dict(record["final_revalidation"])
    return {
        "valid": returned_count == len(ORDERED_SEEDS)
        and all(
            record["final_revalidation"]["valid"] is True
            for record in seed_records
            if record.get("execution_status") == "RETURNED"
        ),
        "per_seed": results,
    }


def _cross_seed_training_input_verification(
    seed_records: list[dict[str, Any]],
) -> dict[str, Any]:
    hashes: dict[str, str | None] = {}
    for record in seed_records:
        chain = record.get("content_chain")
        digest = (
            chain.get("non_seed_training_inputs_sha256")
            if isinstance(chain, Mapping)
            else None
        )
        hashes[str(record["seed"])] = digest
    available = [digest for digest in hashes.values() if digest is not None]
    valid = len(available) == len(ORDERED_SEEDS) and len(set(available)) == 1
    if not valid:
        message = "non-seed training_inputs evidence is missing or differs across seeds"
        for record in seed_records:
            if message not in record["integrity_errors"]:
                record["integrity_errors"].append(message)
            record["evidence_integrity"] = "INCOMPLETE"
    return {
        "valid": valid,
        "comparison": "CANONICAL_TRAINING_INPUTS_WITH_SEED_REMOVED",
        "per_seed_sha256": hashes,
        "common_sha256": available[0] if valid else None,
    }


def run_experiment(plan_path: str | Path) -> dict[str, Any]:
    """Execute all preregistered seeds and write one immutable aggregate."""

    source = Path(plan_path).expanduser().resolve()
    plan_payload = source.read_bytes()
    try:
        plan = json.loads(plan_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("plan must be UTF-8 JSON") from error
    plan = _validate_plan(plan)
    initial_plan_sha256 = hashlib.sha256(plan_payload).hexdigest()

    # All hashes and nested encoder bytes are verified before output creation
    # or any evolve call.
    resolved, verified_inputs = verify_plan_artifacts(plan, source)
    output_root = _validate_output_root(plan, source)
    output_root.mkdir(parents=True, exist_ok=False)
    plan_copy = output_root / "preregistered-plan.json"
    _write_exclusive(plan_copy, plan_payload)

    seed_records: list[dict[str, Any]] = []
    input_integrity_error: str | None = None
    for seed in ORDERED_SEEDS:
        role = "PRIMARY" if seed == PRIMARY_SEED else "ROBUSTNESS_ONLY"
        if input_integrity_error is not None:
            seed_records.append(_not_run_record(seed, role, input_integrity_error))
            continue
        try:
            # Re-hash before every seed so later runs cannot silently consume
            # bytes different from the preregistration.
            if file_sha256(source) != initial_plan_sha256:
                raise ValueError("preregistered plan bytes changed during execution")
            verify_plan_artifacts(plan, source)
        except Exception as error:  # evidence is written below; training stops closed
            input_integrity_error = f"{type(error).__name__}: {error}"
            seed_records.append(_not_run_record(seed, role, input_integrity_error))
            continue

        seed_root = output_root / "seed-results" / f"seed-{seed}"
        seed_root.mkdir(parents=True, exist_ok=False)
        state_dir = seed_root / "run-state"
        candidate_root = seed_root / "candidates"
        result: Mapping[str, Any] | None = None
        execution_error: Exception | None = None
        try:
            returned = evolve(
                resolved["reviewed_corpus"],
                state_dir=state_dir,
                registry_path=resolved["public_registry"],
                suite_path=resolved["promotion_suite"],
                calibration_path=resolved["calibration_fit"],
                held_out_vocabulary_path=resolved["held_out_vocabulary"],
                output_root=candidate_root,
                encoder_manifest_path=resolved["encoder_manifest"],
                steps=TRAINING_STEPS,
                seed=seed,
                retention_epsilon=RETENTION_EPSILON,
            )
            if not isinstance(returned, Mapping):
                raise TypeError("evolve must return an evidence mapping")
            result = returned
        except Exception as error:  # preserve and continue all preregistered seeds
            execution_error = error
        seed_records.append(
            _seed_record(
                seed=seed,
                role=role,
                seed_root=seed_root,
                experiment_root=output_root,
                result=result,
                error=execution_error,
                resolved=resolved,
                plan=plan,
            )
        )

    final_input_verification: dict[str, Any]
    preserved_copy_sha256: str | None = None
    try:
        if file_sha256(source) != initial_plan_sha256:
            raise ValueError("preregistered plan bytes changed during execution")
        preserved_plan_payload = plan_copy.read_bytes()
        preserved_copy_sha256 = hashlib.sha256(preserved_plan_payload).hexdigest()
        if preserved_plan_payload != plan_payload:
            raise ValueError(
                "preserved preregistered plan copy changed during execution"
            )
        if preserved_copy_sha256 != initial_plan_sha256:
            raise ValueError(
                "preserved preregistered plan copy hash differs from the source"
            )
        verify_plan_artifacts(plan, source)
        final_input_verification = {"valid": True, "error": None}
    except Exception as error:
        final_input_verification = {
            "valid": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }

    final_seed_revalidation = _final_revalidate_returned_seeds(
        seed_records,
        experiment_root=output_root,
        resolved=resolved,
        plan=plan,
    )
    cross_seed_training_inputs = _cross_seed_training_input_verification(seed_records)
    all_complete = (
        len(seed_records) == len(ORDERED_SEEDS)
        and all(record["evidence_integrity"] == "COMPLETE" for record in seed_records)
        and final_seed_revalidation["valid"] is True
        and cross_seed_training_inputs["valid"] is True
        and final_input_verification["valid"] is True
    )
    primary = next(record for record in seed_records if record["seed"] == PRIMARY_SEED)
    recommendation_eligible = (
        all_complete and primary["authoritative_decision"] == "QUALIFY"
    )
    if not all_complete:
        outcome = "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    elif recommendation_eligible:
        outcome = "PRIMARY_QUALIFIED_RECOMMENDATION_ELIGIBLE"
    else:
        outcome = "PRIMARY_REJECTED_NO_RECOMMENDATION"

    recommendation_checkpoint = None
    if recommendation_eligible:
        recommendation_checkpoint = dict(primary["content_chain"]["challenger"])

    aggregate: dict[str, Any] = {
        "schema": AGGREGATE_SCHEMA,
        "experiment_id": plan["experiment_id"],
        "plan": {
            "source_path": str(source),
            "source_sha256": initial_plan_sha256,
            "preserved_copy_path": plan_copy.relative_to(output_root).as_posix(),
            "preserved_copy_sha256": preserved_copy_sha256,
        },
        "verified_input_artifacts": verified_inputs,
        "execution": plan["execution"],
        "selection_policy": plan["selection_policy"],
        "seed_runs": seed_records,
        "final_seed_revalidation": final_seed_revalidation,
        "cross_seed_training_input_verification": cross_seed_training_inputs,
        "final_input_verification": final_input_verification,
        "outcome": outcome,
        "promotion_recommendation": {
            "eligible": recommendation_eligible,
            "seed": PRIMARY_SEED if recommendation_eligible else None,
            "checkpoint": recommendation_checkpoint,
            "robustness_seeds_can_substitute": False,
        },
        "decision_authority": "PER_SEED_LEDGER_TERMINAL_EVENT",
        "aggregate_has_model_activation_authority": False,
        "training_loss_used_as_aggregate_criterion": False,
        "candidate_preservation_policy": "NO_DELETION_AND_HASH_INVENTORY",
    }
    aggregate["integrity"] = {
        "algorithm": "sha256",
        "scope": "canonical JSON of this object with integrity omitted",
        "sha256": canonical_json_sha256(aggregate),
    }
    aggregate_path = output_root / "aggregate-evidence.json"
    _write_exclusive_json(aggregate_path, aggregate)
    return {
        "schema": "kev.preregistered-experiment-run-result.v1",
        "outcome": outcome,
        "aggregate_path": str(aggregate_path),
        "aggregate_sha256": file_sha256(aggregate_path),
        "recommendation_eligible": recommendation_eligible,
    }


def verify_plan(plan_path: str | Path) -> dict[str, Any]:
    source = Path(plan_path).expanduser().resolve()
    payload = source.read_bytes()
    try:
        plan = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("plan must be UTF-8 JSON") from error
    plan = _validate_plan(plan)
    _, records = verify_plan_artifacts(plan, source)
    output = _validate_output_root(plan, source)
    return {
        "schema": "kev.preregistered-experiment-plan-verification.v1",
        "valid": True,
        "plan_path": str(source),
        "plan_sha256": hashlib.sha256(payload).hexdigest(),
        "output_root": str(output),
        "artifacts": records,
        "training_executed": False,
    }


def _additional_artifacts(values: list[str]) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for value in values:
        role, separator, raw_path = value.partition("=")
        if not separator or not role.strip() or not raw_path.strip():
            raise ValueError("--additional-artifact must use ROLE=PATH")
        role = role.strip()
        if role in artifacts:
            raise ValueError(f"duplicate additional artifact role: {role}")
        artifacts[role] = Path(raw_path.strip())
    return artifacts


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create or execute KEV's fixed frozen-encoder experiment"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate", help="hash and freeze a plan")
    generate.add_argument("--plan", required=True)
    generate.add_argument("--experiment-id", required=True)
    generate.add_argument("--output-root", required=True)
    generate.add_argument("--public-registry", required=True)
    generate.add_argument("--promotion-suite", required=True)
    generate.add_argument("--calibration-fit", required=True)
    generate.add_argument("--held-out-vocabulary", required=True)
    generate.add_argument("--encoder-manifest", required=True)
    generate.add_argument("--reviewed-corpus", required=True)
    generate.add_argument(
        "--additional-artifact",
        action="append",
        default=[],
        metavar="ROLE=PATH",
        help="pin another evidence artifact; may be repeated",
    )

    verify = subparsers.add_parser("verify", help="verify without training")
    verify.add_argument("--plan", required=True)
    run = subparsers.add_parser("run", help="execute all fixed seeds")
    run.add_argument("--plan", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "generate":
            result = create_plan(
                plan_path=args.plan,
                experiment_id=args.experiment_id,
                output_root=args.output_root,
                public_registry=args.public_registry,
                promotion_suite=args.promotion_suite,
                calibration_fit=args.calibration_fit,
                held_out_vocabulary=args.held_out_vocabulary,
                encoder_manifest=args.encoder_manifest,
                reviewed_corpus=args.reviewed_corpus,
                additional_artifacts=_additional_artifacts(args.additional_artifact),
            )
        elif args.command == "verify":
            result = verify_plan(args.plan)
        else:
            result = run_experiment(args.plan)
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": "kev.preregistered-experiment-command-error.v1",
                    "error": {"type": type(error).__name__, "message": str(error)},
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    if (
        args.command == "run"
        and result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    ):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

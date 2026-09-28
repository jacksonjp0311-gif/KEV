from __future__ import annotations

import hashlib
import io
import json
import os
import tempfile
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Mapping

import torch

from kev.artifacts import (
    canonical_json_sha256,
    file_sha256,
    validate_sha256,
    verify_file_sha256,
)
from kev.calibration import calibrate_checkpoint
from kev.calibration_evidence import (
    load_calibration_receipt,
    validate_calibration_binding,
)
from kev.encoders import (
    FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
    verify_local_encoder_manifest,
)
from kev.evaluation import evaluate_challenger, load_frozen_suite
from kev.model_runtime import CheckpointPredictor
from kev.sentence_training import train_frozen_encoder_challenger
from kev.uc51a2.semantic_breadth import (
    _frozen_evaluation_exclusions,
    required_held_out_vocabulary,
    sha256_file,
    train,
)
from kev.uc51a3.alive import AliveStore, IncumbentCompareAndSwapError


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "models" / "registry.json"
DEFAULT_EVAL_MANIFEST = ROOT / "evals" / "frozen" / "manifest-v6.json"
DEFAULT_SUITE = ROOT / "evals" / "frozen" / "public-audit-v6-260.json"
DEFAULT_CALIBRATION = ROOT / "evals" / "frozen" / "calibration-fit-v2.jsonl"
DEFAULT_HELD_OUT = ROOT / "evals" / "frozen" / "held-out-vocabulary-v2.txt"
_TERMINAL_MODEL_EVENTS = frozenset({"MODEL_REJECTED", "MODEL_QUALIFIED"})
_TRAINING_RECEIPT_SCHEMA = "kev.training-receipt.v1"
_HASHED_CHECKPOINT_SCHEMA = "kev.semantic-frames.v052"
_TRAINING_OBJECTIVE = "L_relation + 0.20 L_cardinality + 0.35 L_contrastive"


def _current_sentence_training_source_manifest() -> list[dict[str, str]]:
    """Return the exact source closure used by frozen-encoder training."""

    return [
        {
            "role": "FROZEN_ENCODER_TRAINER",
            "path": "kev/sentence_training.py",
            "sha256": file_sha256(ROOT / "kev" / "sentence_training.py"),
        },
        {
            "role": "ENCODER_RUNTIME",
            "path": "kev/encoders.py",
            "sha256": file_sha256(ROOT / "kev" / "encoders.py"),
        },
        {
            "role": "SHARED_SEMANTIC_TRAINING",
            "path": "kev/uc51a2/semantic_breadth.py",
            "sha256": file_sha256(ROOT / "kev" / "uc51a2" / "semantic_breadth.py"),
        },
    ]


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_immutable_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _resolve_artifact(path_value: str | Path, base: Path = ROOT) -> Path:
    path = Path(path_value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def _public_incumbent(registry_path: str | Path = DEFAULT_REGISTRY) -> dict[str, Any]:
    registry_file = Path(registry_path).resolve()
    registry = json.loads(registry_file.read_text(encoding="utf-8"))
    if registry.get("schema") != "kev.model-registry.v1":
        raise ValueError("invalid model registry schema")
    active = dict(registry["active"])
    path = _resolve_artifact(active["path"], ROOT)
    verify_file_sha256(path, active["sha256"])
    return {**active, "path": str(path)}


def _trusted_eval_manifest(registry_path: str | Path) -> dict[str, Any]:
    """Anchor the canonical promotion manifest through public incumbent evidence."""

    registry = json.loads(Path(registry_path).resolve().read_text(encoding="utf-8"))
    if registry.get("schema") != "kev.model-registry.v1" or not isinstance(
        registry.get("active"), Mapping
    ):
        raise ValueError("invalid model registry schema")
    active = registry["active"]
    evidence_path = _resolve_artifact(str(active.get("evidence_manifest_path", "")))
    evidence_hash = validate_sha256(
        str(active.get("evidence_manifest_sha256", "")),
        field="active evidence_manifest_sha256",
    )
    evidence_payload = evidence_path.read_bytes()
    if hashlib.sha256(evidence_payload).hexdigest() != evidence_hash:
        raise ValueError("incumbent evidence SHA-256 mismatch")
    evidence = json.loads(evidence_payload.decode("utf-8"))
    manifest_path = _resolve_artifact(str(evidence.get("eval_manifest_path", "")))
    if manifest_path != DEFAULT_EVAL_MANIFEST.resolve():
        raise ValueError("incumbent evidence does not anchor canonical manifest-v6")
    manifest_hash = validate_sha256(
        str(evidence.get("eval_manifest_sha256", "")),
        field="incumbent evidence eval_manifest_sha256",
    )
    manifest_payload = manifest_path.read_bytes()
    if hashlib.sha256(manifest_payload).hexdigest() != manifest_hash:
        raise ValueError("evaluation manifest SHA-256 mismatch")
    suite_path = _resolve_artifact(str(evidence.get("suite_path", "")))
    if suite_path != DEFAULT_SUITE.resolve():
        raise ValueError("incumbent evidence does not anchor canonical promotion suite")
    verify_file_sha256(suite_path, str(evidence.get("suite_sha256", "")))

    manifest = json.loads(manifest_payload.decode("utf-8"))
    if manifest.get("schema") != "kev.eval-manifest.v1" or not manifest.get("frozen"):
        raise ValueError("invalid or unfrozen evaluation manifest")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("evaluation manifest requires artifacts")
    promotion = artifacts.get("promotion_suite")
    if not isinstance(promotion, Mapping):
        raise ValueError("evaluation manifest requires promotion_suite")
    if (
        _resolve_artifact(str(promotion.get("path", ""))) != suite_path
        or promotion.get("sha256") != evidence.get("suite_sha256")
        or promotion.get("canonical_sha256") != evidence.get("suite_canonical_sha256")
    ):
        raise ValueError(
            "canonical manifest promotion suite differs from incumbent evidence"
        )
    return manifest


def _manifest_artifact(
    role: str, registry_path: str | Path = DEFAULT_REGISTRY
) -> dict[str, Any]:
    manifest = _trusted_eval_manifest(registry_path)
    reference = dict(manifest["artifacts"][role])
    verify_file_sha256(_resolve_artifact(reference["path"]), reference["sha256"])
    return reference


def _active_incumbent(store: AliveStore, registry_path: str | Path) -> dict[str, Any]:
    state = store.read()
    path_value = state.get("semantic_model")
    digest = state.get("semantic_model_sha256")
    generation = state.get("semantic_model_generation")
    if (
        isinstance(generation, bool)
        or not isinstance(generation, int)
        or generation < 0
    ):
        raise ValueError("active state has an invalid semantic model generation")
    if path_value and digest:
        path = Path(path_value).resolve()
        verify_file_sha256(path, digest)
        return {
            "path": str(path),
            "sha256": digest,
            "generation": generation,
            "source": "state",
        }
    return {
        **_public_incumbent(registry_path),
        "generation": generation,
        "source": "public-registry",
    }


def _verified_default_suite(
    suite_path: str | Path, registry_path: str | Path = DEFAULT_REGISTRY
) -> Any:
    suite = Path(suite_path).resolve()
    if suite != DEFAULT_SUITE.resolve():
        raise ValueError(
            "MODEL_QUALIFIED requires the canonical manifest-v6 promotion suite"
        )
    reference = _manifest_artifact("promotion_suite", registry_path)
    return load_frozen_suite(
        suite,
        expected_file_sha256=reference["sha256"],
        expected_canonical_sha256=reference["canonical_sha256"],
    )


def _verified_gate_artifacts(
    *,
    suite_path: str | Path,
    calibration_path: str | Path | None,
    held_out_vocabulary_path: str | Path,
    registry_path: str | Path,
) -> Any:
    """Require the evidence-anchored canonical inputs for qualification."""

    if calibration_path is None or Path(calibration_path).resolve() != (
        DEFAULT_CALIBRATION.resolve()
    ):
        raise ValueError(
            "MODEL_QUALIFIED requires the canonical manifest-v6 calibration fit"
        )
    if Path(held_out_vocabulary_path).resolve() != DEFAULT_HELD_OUT.resolve():
        raise ValueError(
            "MODEL_QUALIFIED requires the canonical manifest-v6 held-out vocabulary"
        )
    frozen = _verified_default_suite(suite_path, registry_path)
    _manifest_artifact("calibration_fit", registry_path)
    _manifest_artifact("held_out_vocabulary", registry_path)
    return frozen


def _assert_train_eval_separation(lessons_path: Path, frozen_suite: Any) -> None:
    lesson_texts: set[str] = set()
    for line in lessons_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            lesson_texts.add(str(json.loads(line).get("text", "")).strip().casefold())
    eval_texts = {
        str(item.get("input", item.get("text", ""))).strip().casefold()
        for rows in frozen_suite.splits.values()
        for item in rows
    }
    overlap = sorted((lesson_texts & eval_texts) - {""})
    if overlap:
        raise ValueError(
            "reviewed lessons overlap the frozen promotion suite; evaluation leakage is forbidden"
        )


def _temperature_receipt_for(checkpoint: Path) -> Path | None:
    candidate = checkpoint.parent / "calibration-receipt.json"
    return candidate if candidate.is_file() else None


def _recorded_terminal_event(store: AliveStore, run_id: str) -> dict[str, Any] | None:
    """Return this run's sole durable model decision, if one exists."""

    verification = store.verify_ledger()
    if not verification["valid"]:
        raise RuntimeError(
            "cannot determine the evolution decision from an invalid ledger: "
            f"{verification.get('reason')}"
        )
    if not store.ledger_path.exists():
        return None
    matches: list[dict[str, Any]] = []
    for line in store.ledger_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        data = event.get("data")
        if (
            event.get("kind") in _TERMINAL_MODEL_EVENTS
            and isinstance(data, dict)
            and data.get("run_id") == run_id
        ):
            matches.append(event)
    if len(matches) > 1:
        raise RuntimeError(
            f"evolution run {run_id} has multiple terminal model decisions"
        )
    return matches[0] if matches else None


def _update_local_registry(
    store: AliveStore,
    incumbent: Mapping[str, Any],
    challenger: Mapping[str, Any],
    eval_card: Mapping[str, Any],
) -> Path:
    path = store.dir / "model-registry.json"
    if path.exists():
        registry = json.loads(path.read_text(encoding="utf-8"))
    else:
        registry = {
            "schema": "kev.model-registry.v1",
            "active": dict(incumbent),
            "history": [],
        }
    registry.setdefault("history", []).append(dict(registry.get("active", incumbent)))
    registry["active"] = dict(challenger)
    registry["last_eval_card_sha256"] = eval_card["sha256"]
    registry["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(path, registry)
    return path


def _require_training_value(
    record: Mapping[str, Any], field: str, expected: Any, *, record_name: str
) -> None:
    if field not in record or record[field] != expected:
        raise ValueError(
            f"{record_name} {field} does not match the verified evolution input"
        )


def _require_exact_training_path(
    value: Any, expected: Path, *, field: str, record_name: str
) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{record_name} {field} must be a non-empty path")
    if Path(value).expanduser().resolve() != expected.resolve():
        raise ValueError(
            f"{record_name} {field} does not match the verified evolution path"
        )


def _validate_training_artifacts(
    trainer_result: Mapping[str, Any],
    *,
    training_dir: Path,
    incumbent_path: Path,
    incumbent_sha256: str,
    lessons_path: Path,
    lessons_sha256: str,
    held_out_path: Path,
    held_out_file_sha256: str,
    held_out_terms: list[str],
    held_out_terms_sha256: str,
    required_held_out_terms: list[str],
    required_held_out_terms_sha256: str,
    required_held_out_artifacts: list[dict[str, Any]],
    required_held_out_artifacts_sha256: str,
    encoder_manifest_path: Path | None,
    encoder_manifest_sha256: str | None,
    seed: int,
    steps: int,
) -> dict[str, Any]:
    """Fail closed unless trainer output proves exact input/output lineage.

    The trainer return value is not an authority boundary.  This validator
    independently reads the immutable receipt and checkpoint, re-hashes every
    caller input that could have been changed during training, and requires
    the two artifacts to agree with each other and the pre-training snapshot.
    """

    if not isinstance(trainer_result, Mapping):
        raise ValueError("trainer must return a training receipt mapping")
    if training_dir.is_symlink():
        raise ValueError("training output directory cannot be a symbolic link")

    # A trainer is untrusted until it proves that the exact verified inputs
    # survived unchanged.  The required vocabulary loader also re-verifies all
    # frozen manifest declarations and vocabulary bytes.
    verify_file_sha256(incumbent_path, incumbent_sha256)
    verify_file_sha256(lessons_path, lessons_sha256)
    verify_file_sha256(held_out_path, held_out_file_sha256)
    (
        current_required_terms,
        current_required_artifacts,
        current_required_artifacts_sha256,
    ) = required_held_out_vocabulary()
    if sorted(current_required_terms) != required_held_out_terms:
        raise ValueError("required held-out vocabulary terms changed during training")
    if current_required_artifacts != required_held_out_artifacts:
        raise ValueError(
            "required held-out vocabulary artifact evidence changed during training"
        )
    if current_required_artifacts_sha256 != required_held_out_artifacts_sha256:
        raise ValueError(
            "required held-out vocabulary artifact hash changed during training"
        )
    _, current_exclusion_artifacts = _frozen_evaluation_exclusions()
    current_exclusions_sha256 = canonical_json_sha256(
        sorted(
            (
                {"role": artifact["role"], "sha256": artifact["sha256"]}
                for artifact in current_exclusion_artifacts
            ),
            key=lambda item: (item["role"], item["sha256"]),
        )
    )

    if encoder_manifest_path is not None:
        verified_encoder = verify_local_encoder_manifest(encoder_manifest_path)
        if verified_encoder["manifest_sha256"] != encoder_manifest_sha256:
            raise ValueError("encoder manifest changed during training")
    elif encoder_manifest_sha256 is not None:
        raise ValueError("encoder manifest hash exists without an encoder manifest")

    receipt_path = training_dir / "training-receipt.json"
    challenger_path = training_dir / "semantic-breadth.pt"
    if receipt_path.is_symlink() or challenger_path.is_symlink():
        raise ValueError("training artifacts cannot be symbolic links")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"training receipt is missing: {receipt_path}")
    if not challenger_path.is_file():
        raise FileNotFoundError(f"raw challenger is missing: {challenger_path}")

    receipt_bytes = receipt_path.read_bytes()
    try:
        receipt_value = json.loads(receipt_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("training receipt must be UTF-8 JSON") from error
    if not isinstance(receipt_value, dict):
        raise ValueError("training receipt must contain an object")
    receipt: dict[str, Any] = receipt_value
    # This also rejects non-finite or otherwise non-canonical JSON values.
    canonical_json_sha256(receipt)
    if dict(trainer_result) != receipt:
        raise ValueError(
            "trainer return value does not match the persisted training receipt"
        )

    _require_training_value(
        receipt, "schema", _TRAINING_RECEIPT_SCHEMA, record_name="training receipt"
    )
    _require_training_value(
        receipt, "artifact_status", "CHALLENGER", record_name="training receipt"
    )
    _require_training_value(
        receipt, "activation", "NONE", record_name="training receipt"
    )
    _require_training_value(
        receipt, "parent_sha256", incumbent_sha256, record_name="training receipt"
    )
    _require_exact_training_path(
        receipt.get("parent_path"),
        incumbent_path,
        field="parent_path",
        record_name="training receipt",
    )
    _require_training_value(
        receipt, "lessons_sha256", lessons_sha256, record_name="training receipt"
    )
    _require_exact_training_path(
        receipt.get("lessons_path"),
        lessons_path,
        field="lessons_path",
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "held_out_vocabulary",
        held_out_terms,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "held_out_vocabulary_sha256",
        held_out_terms_sha256,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "required_held_out_vocabulary",
        required_held_out_terms,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "required_held_out_vocabulary_terms_sha256",
        required_held_out_terms_sha256,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "required_held_out_vocabulary_artifacts",
        required_held_out_artifacts,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "required_held_out_vocabulary_artifacts_sha256",
        required_held_out_artifacts_sha256,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "training_exclusion_artifacts",
        current_exclusion_artifacts,
        record_name="training receipt",
    )
    _require_training_value(
        receipt,
        "training_exclusions_sha256",
        current_exclusions_sha256,
        record_name="training receipt",
    )
    _require_exact_training_path(
        receipt.get("challenger_path"),
        challenger_path,
        field="challenger_path",
        record_name="training receipt",
    )
    if type(receipt.get("seed")) is not int or receipt["seed"] != seed:
        raise ValueError("training receipt seed does not match the requested seed")
    if type(receipt.get("steps")) is not int or receipt["steps"] != steps:
        raise ValueError("training receipt steps do not match the requested steps")
    _require_training_value(
        receipt, "objective", _TRAINING_OBJECTIVE, record_name="training receipt"
    )
    _require_training_value(
        receipt, "score_status", "UNCALIBRATED", record_name="training receipt"
    )
    _require_training_value(
        receipt, "promotion_features", [], record_name="training receipt"
    )
    loss_history = receipt.get("loss_history")
    if not isinstance(loss_history, list) or len(loss_history) != steps:
        raise ValueError("training receipt loss history does not match requested steps")
    for index, row in enumerate(loss_history, start=1):
        if not isinstance(row, Mapping) or row.get("step") != index:
            raise ValueError("training receipt loss history is not contiguous")

    checkpoint_bytes = challenger_path.read_bytes()
    challenger_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
    _require_training_value(
        receipt,
        "challenger_sha256",
        challenger_sha256,
        record_name="training receipt",
    )
    try:
        checkpoint_value = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError(
            "raw challenger checkpoint cannot be deserialized safely"
        ) from error
    if not isinstance(checkpoint_value, Mapping):
        raise ValueError("raw challenger checkpoint must contain a mapping")
    checkpoint: Mapping[str, Any] = checkpoint_value

    expected_checkpoint_schema = (
        FROZEN_SENTENCE_CHECKPOINT_SCHEMA
        if encoder_manifest_path is not None
        else _HASHED_CHECKPOINT_SCHEMA
    )
    _require_training_value(
        checkpoint,
        "schema",
        expected_checkpoint_schema,
        record_name="raw challenger checkpoint",
    )
    sentence_training_evidence: dict[str, Any] = {}
    if encoder_manifest_path is not None:
        source_manifest = _current_sentence_training_source_manifest()
        source_hashes = {entry["role"]: entry["sha256"] for entry in source_manifest}
        sentence_training_evidence = {
            "training_contract_source_sha256": source_hashes[
                "SHARED_SEMANTIC_TRAINING"
            ],
            "training_source_manifest": source_manifest,
            "training_source_manifest_sha256": canonical_json_sha256(source_manifest),
        }

    checkpoint_expected = {
        "artifact_status": "CHALLENGER",
        "score_status": "UNCALIBRATED",
        "temperature": None,
        "parent_sha256": incumbent_sha256,
        "lessons_sha256": lessons_sha256,
        "held_out_vocabulary_sha256": held_out_terms_sha256,
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "required_held_out_vocabulary_artifacts": required_held_out_artifacts,
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "training_exclusions_sha256": current_exclusions_sha256,
        "seed": seed,
        "steps": steps,
        "objective": _TRAINING_OBJECTIVE,
        **sentence_training_evidence,
    }
    for field, expected in checkpoint_expected.items():
        _require_training_value(
            checkpoint, field, expected, record_name="raw challenger checkpoint"
        )

    # Every lineage- or objective-bearing field shared by the checkpoint and
    # receipt must be identical.  This prevents a forged receipt from blessing
    # unrelated raw weights even when its top-level hashes look plausible.
    shared_metadata = [
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
        "training_inputs_sha256",
        "training_scope",
        "seed",
        "steps",
        "objective",
        "optimizer",
        "torch_num_threads",
        "score_status",
        "artifact_status",
    ]
    if encoder_manifest_path is not None:
        shared_metadata.extend(
            [
                "encoder_runtime_source_sha256",
                "training_contract_source_sha256",
                "training_source_manifest",
                "training_source_manifest_sha256",
            ]
        )
    for field in shared_metadata:
        if field not in receipt or field not in checkpoint:
            raise ValueError(
                f"training evidence is missing shared checkpoint field {field}"
            )
        if receipt[field] != checkpoint[field]:
            raise ValueError(
                f"raw challenger checkpoint {field} does not match training receipt"
            )

    parameter_scope = receipt.get("parameter_scope")
    checkpoint_parameter_scope = (
        checkpoint.get("parameters")
        if encoder_manifest_path is not None
        else checkpoint.get("parameter_scope")
    )
    if (
        not isinstance(parameter_scope, Mapping)
        or parameter_scope != checkpoint_parameter_scope
    ):
        raise ValueError(
            "raw challenger checkpoint parameter scope does not match training receipt"
        )

    training_inputs = receipt.get("training_inputs")
    if not isinstance(training_inputs, Mapping):
        raise ValueError("training receipt requires a training_inputs object")
    if receipt["training_inputs_sha256"] != canonical_json_sha256(training_inputs):
        raise ValueError("training_inputs hash does not match its receipt value")
    expected_input_fields = {
        "parent_sha256": incumbent_sha256,
        "lessons_sha256": lessons_sha256,
        "held_out_vocabulary_sha256": held_out_terms_sha256,
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "training_exclusions_sha256": current_exclusions_sha256,
        "seed": seed,
        "steps": steps,
        "objective": _TRAINING_OBJECTIVE,
    }
    trainer_source_path = (
        ROOT / "kev" / "sentence_training.py"
        if encoder_manifest_path is not None
        else ROOT / "kev" / "uc51a2" / "semantic_breadth.py"
    )
    expected_input_fields["trainer_source_sha256"] = file_sha256(trainer_source_path)
    if encoder_manifest_path is not None:
        expected_input_fields["encoder_runtime_source_sha256"] = file_sha256(
            ROOT / "kev" / "encoders.py"
        )
        expected_input_fields.update(sentence_training_evidence)
    for field, expected_input in expected_input_fields.items():
        _require_training_value(
            training_inputs, field, expected_input, record_name="training_inputs"
        )

    if encoder_manifest_path is not None:
        # Independently require the current three-file source closure in all
        # evidence surfaces. Agreement alone is insufficient: a receipt and
        # checkpoint could otherwise consistently attest to stale or forged
        # source hashes.
        for record_name, record in (
            ("training receipt", receipt),
            ("raw challenger checkpoint", checkpoint),
            ("training_inputs", training_inputs),
        ):
            for field, expected in sentence_training_evidence.items():
                _require_training_value(
                    record, field, expected, record_name=record_name
                )

    if encoder_manifest_path is not None:
        _require_training_value(
            training_inputs,
            "encoder_manifest_sha256",
            encoder_manifest_sha256,
            record_name="training_inputs",
        )
        encoder_artifact = receipt.get("encoder_artifact")
        encoder_reference = checkpoint.get("encoder_manifest")
        if not isinstance(encoder_artifact, Mapping) or not isinstance(
            encoder_reference, Mapping
        ):
            raise ValueError("frozen-encoder training evidence lacks manifest bindings")
        for field in ("path", "sha256", "name", "source", "license", "network_policy"):
            if encoder_artifact.get(field) != encoder_reference.get(field):
                raise ValueError(
                    f"checkpoint encoder manifest {field} does not match receipt"
                )
        _require_training_value(
            encoder_artifact,
            "sha256",
            encoder_manifest_sha256,
            record_name="training receipt encoder artifact",
        )
        _require_training_value(
            receipt,
            "model_family",
            "FROZEN_SENTENCE_ENCODER_PROPOSAL",
            record_name="training receipt",
        )
        _require_training_value(
            checkpoint,
            "model_family",
            "FROZEN_SENTENCE_ENCODER_PROPOSAL",
            record_name="raw challenger checkpoint",
        )
    else:
        if "encoder_artifact" in receipt or "encoder_manifest" in checkpoint:
            raise ValueError(
                "hashed challenger unexpectedly claims an encoder artifact"
            )
        if "encoder_manifest_sha256" in training_inputs:
            raise ValueError(
                "hashed challenger training_inputs claim an encoder manifest"
            )

    return {
        "receipt": receipt,
        "receipt_path": receipt_path.resolve(),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "checkpoint": checkpoint,
        "challenger_path": challenger_path.resolve(),
        "challenger_sha256": challenger_sha256,
        "held_out_vocabulary_sha256": held_out_terms_sha256,
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "encoder_manifest_sha256": encoder_manifest_sha256,
        "training_source_manifest": (
            sentence_training_evidence.get("training_source_manifest")
            if encoder_manifest_path is not None
            else None
        ),
    }


def _validate_calibration_artifacts(
    calibrator_result: Mapping[str, Any],
    *,
    calibration_dir: Path,
    raw_challenger_path: Path,
    raw_challenger_sha256: str,
    calibration_fit_path: Path,
    promotion_suite_path: Path,
    frozen_suite: Any,
    registry_path: str | Path,
) -> dict[str, Any]:
    """Independently bind calibrated output to the verified raw challenger.

    The calibrator return value is not authoritative. The persisted receipt,
    exact output bytes, raw source, and both manifest-bound data surfaces must
    all agree before the calibrated child can enter evaluation.
    """

    if not isinstance(calibrator_result, Mapping):
        raise ValueError("calibrator must return a calibration receipt mapping")
    if calibration_dir.is_symlink():
        raise ValueError("calibration output directory cannot be a symbolic link")

    receipt_path = (calibration_dir / "calibration-receipt.json").resolve()
    output_path = (calibration_dir / "semantic-breadth.calibrated.pt").resolve()
    if receipt_path.is_symlink() or output_path.is_symlink():
        raise ValueError("calibration artifacts cannot be symbolic links")
    if not receipt_path.is_file():
        raise FileNotFoundError(f"calibration receipt is missing: {receipt_path}")
    if not output_path.is_file():
        raise FileNotFoundError(f"calibrated challenger is missing: {output_path}")

    raw_path = raw_challenger_path.resolve()
    fit_path = calibration_fit_path.resolve()
    suite_path = promotion_suite_path.resolve()
    verify_file_sha256(raw_path, raw_challenger_sha256)

    fit_reference = _manifest_artifact("calibration_fit", registry_path)
    fit_reference_path = _resolve_artifact(fit_reference["path"])
    if fit_reference_path != fit_path:
        raise ValueError("calibration fit differs from the canonical manifest artifact")
    fit_sha256 = validate_sha256(
        str(fit_reference["sha256"]), field="manifest calibration_fit sha256"
    )
    verify_file_sha256(fit_path, fit_sha256)

    suite_reference = _manifest_artifact("promotion_suite", registry_path)
    suite_reference_path = _resolve_artifact(suite_reference["path"])
    if suite_reference_path != suite_path:
        raise ValueError("promotion suite differs from the canonical manifest artifact")
    suite_file_sha256 = validate_sha256(
        str(suite_reference["sha256"]), field="manifest promotion_suite sha256"
    )
    suite_canonical_sha256 = validate_sha256(
        str(suite_reference["canonical_sha256"]),
        field="manifest promotion_suite canonical_sha256",
    )
    current_suite = load_frozen_suite(
        suite_path,
        expected_file_sha256=suite_file_sha256,
        expected_canonical_sha256=suite_canonical_sha256,
    )
    if (
        frozen_suite.file_sha256 != current_suite.file_sha256
        or frozen_suite.canonical_sha256 != current_suite.canonical_sha256
    ):
        raise ValueError("promotion suite changed after initial gate verification")

    artifact = load_calibration_receipt(receipt_path)
    receipt = artifact.data
    if artifact.file_sha256 is None:
        raise ValueError("persisted calibration receipt lacks a byte hash")
    if dict(calibrator_result) != receipt:
        raise ValueError(
            "calibrator return value does not match the persisted calibration receipt"
        )
    _require_exact_training_path(
        receipt.get("source_path"),
        raw_path,
        field="source_path",
        record_name="calibration receipt",
    )
    _require_exact_training_path(
        receipt.get("calibration_suite"),
        fit_path,
        field="calibration_suite",
        record_name="calibration receipt",
    )
    _require_exact_training_path(
        receipt.get("promotion_suite"),
        suite_path,
        field="promotion_suite",
        record_name="calibration receipt",
    )
    _require_exact_training_path(
        receipt.get("output_path"),
        output_path,
        field="output_path",
        record_name="calibration receipt",
    )
    _require_training_value(
        receipt,
        "source_sha256",
        raw_challenger_sha256,
        record_name="calibration receipt",
    )
    _require_training_value(
        receipt,
        "calibration_suite_sha256",
        fit_sha256,
        record_name="calibration receipt",
    )
    _require_training_value(
        receipt,
        "fit_data_sha256",
        fit_sha256,
        record_name="calibration receipt",
    )
    _require_training_value(
        receipt,
        "promotion_suite_file_sha256",
        suite_file_sha256,
        record_name="calibration receipt",
    )
    _require_training_value(
        receipt,
        "promotion_suite_canonical_sha256",
        suite_canonical_sha256,
        record_name="calibration receipt",
    )
    _require_training_value(
        receipt, "activation", "NONE", record_name="calibration receipt"
    )

    output_bytes = output_path.read_bytes()
    output_sha256 = hashlib.sha256(output_bytes).hexdigest()
    _require_training_value(
        receipt,
        "output_sha256",
        output_sha256,
        record_name="calibration receipt",
    )
    validated = validate_calibration_binding(
        artifact,
        checkpoint_bytes=output_bytes,
        promotion_suite_file_sha256=suite_file_sha256,
        promotion_suite_canonical_sha256=suite_canonical_sha256,
        source_checkpoint_path=raw_path,
        calibration_fit_path=fit_path,
        promotion_suite_path=suite_path,
    )
    verify_file_sha256(raw_path, raw_challenger_sha256)
    verify_file_sha256(fit_path, fit_sha256)
    verify_file_sha256(suite_path, suite_file_sha256)
    verify_file_sha256(receipt_path, artifact.file_sha256)

    return {
        "receipt": receipt,
        "receipt_path": receipt_path,
        "receipt_sha256": artifact.file_sha256,
        "receipt_canonical_sha256": validated.canonical_sha256,
        "challenger_path": output_path,
        "challenger_sha256": output_sha256,
        "challenger_bytes": output_bytes,
        "raw_challenger_sha256": raw_challenger_sha256,
        "raw_challenger_path": raw_path,
        "calibration_fit_sha256": fit_sha256,
        "calibration_fit_path": fit_path,
        "promotion_suite_file_sha256": suite_file_sha256,
        "promotion_suite_canonical_sha256": suite_canonical_sha256,
        "promotion_suite_path": suite_path,
    }


def _reverify_calibration_artifacts(verified: Mapping[str, Any]) -> None:
    """Recheck every calibrated-child input immediately before a decision."""

    verify_file_sha256(
        Path(verified["raw_challenger_path"]),
        str(verified["raw_challenger_sha256"]),
    )
    verify_file_sha256(Path(verified["receipt_path"]), str(verified["receipt_sha256"]))
    verify_file_sha256(
        Path(verified["calibration_fit_path"]),
        str(verified["calibration_fit_sha256"]),
    )
    verify_file_sha256(
        Path(verified["promotion_suite_path"]),
        str(verified["promotion_suite_file_sha256"]),
    )
    verify_file_sha256(
        Path(verified["challenger_path"]), str(verified["challenger_sha256"])
    )


def evolve(
    lessons: str | Path,
    *,
    state_dir: str | Path | None = None,
    registry_path: str | Path = DEFAULT_REGISTRY,
    suite_path: str | Path = DEFAULT_SUITE,
    calibration_path: str | Path | None = DEFAULT_CALIBRATION,
    held_out_vocabulary_path: str | Path = DEFAULT_HELD_OUT,
    output_root: str | Path | None = None,
    encoder_manifest_path: str | Path | None = None,
    steps: int = 400,
    seed: int = 52021,
    retention_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Run KEV's closed, evidence-bearing challenger gate once."""

    store = AliveStore(state_dir)
    lesson_path = Path(lessons).expanduser().resolve()
    lesson_fingerprint = (
        file_sha256(lesson_path)[:10] if lesson_path.is_file() else "missing"
    )
    run_id = (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime())
        + "-"
        + lesson_fingerprint
        + "-"
        + uuid.uuid4().hex[:8]
    )
    root = (
        Path(output_root).resolve()
        if output_root
        else store.dir / "evolution" / "candidates"
    )
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    failure_path = run_dir / "raw-failure.json"

    activated = False
    terminal_event: dict[str, Any] | None = None
    incumbent: dict[str, Any] | None = None
    frozen_suite: Any = None
    held_out_path = Path(held_out_vocabulary_path).resolve()
    calibration_file = (
        Path(calibration_path).resolve() if calibration_path is not None else None
    )
    encoder_manifest_file = (
        Path(encoder_manifest_path).expanduser().resolve()
        if encoder_manifest_path is not None
        else None
    )
    try:
        if not lesson_path.is_file():
            raise FileNotFoundError(lesson_path)
        incumbent = _active_incumbent(store, registry_path)
        frozen_suite = _verified_gate_artifacts(
            suite_path=suite_path,
            calibration_path=calibration_file,
            held_out_vocabulary_path=held_out_path,
            registry_path=registry_path,
        )
        _assert_train_eval_separation(lesson_path, frozen_suite)
        lessons_sha256 = file_sha256(lesson_path)
        held_out_file_sha256 = file_sha256(held_out_path)
        held_out = [
            line.strip()
            for line in held_out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        (
            required_held_out,
            required_held_out_artifacts,
            required_held_out_artifacts_sha256,
        ) = required_held_out_vocabulary()
        required_held_out_terms = sorted(required_held_out)
        required_held_out_terms_sha256 = canonical_json_sha256(required_held_out_terms)
        full_held_out_terms = sorted(
            required_held_out
            | {str(term).casefold().strip() for term in held_out if str(term).strip()}
        )
        full_held_out_terms_sha256 = canonical_json_sha256(full_held_out_terms)
        encoder_manifest_sha256: str | None = None
        if encoder_manifest_file is not None:
            encoder_manifest_sha256 = verify_local_encoder_manifest(
                encoder_manifest_file
            )["manifest_sha256"]
        training_dir = run_dir / "trained"
        if encoder_manifest_file is None:
            training_receipt = train(
                incumbent["path"],
                training_dir,
                steps=steps,
                seed=seed,
                extra_jsonl=lesson_path,
                held_out_vocabulary=held_out,
            )
        else:
            training_receipt = train_frozen_encoder_challenger(
                incumbent["path"],
                encoder_manifest_file,
                training_dir,
                steps=steps,
                seed=seed,
                extra_jsonl=lesson_path,
                held_out_vocabulary=held_out,
            )
        verified_training = _validate_training_artifacts(
            training_receipt,
            training_dir=training_dir,
            incumbent_path=Path(incumbent["path"]).resolve(),
            incumbent_sha256=str(incumbent["sha256"]),
            lessons_path=lesson_path,
            lessons_sha256=lessons_sha256,
            held_out_path=held_out_path,
            held_out_file_sha256=held_out_file_sha256,
            held_out_terms=full_held_out_terms,
            held_out_terms_sha256=full_held_out_terms_sha256,
            required_held_out_terms=required_held_out_terms,
            required_held_out_terms_sha256=required_held_out_terms_sha256,
            required_held_out_artifacts=required_held_out_artifacts,
            required_held_out_artifacts_sha256=(required_held_out_artifacts_sha256),
            encoder_manifest_path=encoder_manifest_file,
            encoder_manifest_sha256=encoder_manifest_sha256,
            seed=seed,
            steps=steps,
        )
        uncalibrated_path = Path(verified_training["challenger_path"])
        challenger_path = uncalibrated_path
        expected_challenger_sha256 = str(verified_training["challenger_sha256"])
        calibration_receipt: dict[str, Any] | None = None
        calibration_receipt_path: Path | None = None
        verified_calibration: dict[str, Any] | None = None
        if calibration_file is not None:
            calibration_dir = run_dir / "calibrated"
            calibration_receipt = calibrate_checkpoint(
                uncalibrated_path,
                calibration_file,
                calibration_dir,
                promotion_suite=suite_path,
            )
            verified_calibration = _validate_calibration_artifacts(
                calibration_receipt,
                calibration_dir=calibration_dir,
                raw_challenger_path=uncalibrated_path,
                raw_challenger_sha256=expected_challenger_sha256,
                calibration_fit_path=calibration_file,
                promotion_suite_path=Path(suite_path).resolve(),
                frozen_suite=frozen_suite,
                registry_path=registry_path,
            )
            calibration_receipt = verified_calibration["receipt"]
            challenger_path = verified_calibration["challenger_path"]
            expected_challenger_sha256 = verified_calibration["challenger_sha256"]
            calibration_receipt_path = verified_calibration["receipt_path"]

        incumbent_path = Path(incumbent["path"]).resolve()
        incumbent_bytes = incumbent_path.read_bytes()
        evaluated_incumbent_sha256 = hashlib.sha256(incumbent_bytes).hexdigest()
        if evaluated_incumbent_sha256 != incumbent["sha256"]:
            raise RuntimeError("incumbent changed before evaluation")
        expected_incumbent = {
            "path": str(incumbent_path),
            "sha256": evaluated_incumbent_sha256,
            "generation": incumbent["generation"],
        }
        incumbent_predictor = CheckpointPredictor(incumbent_path)
        if incumbent_predictor.sha256 != evaluated_incumbent_sha256:
            raise RuntimeError(
                "incumbent predictor did not load the verified incumbent"
            )
        challenger_predictor = CheckpointPredictor(challenger_path)
        if challenger_predictor.sha256 != expected_challenger_sha256:
            raise RuntimeError(
                "challenger changed between training/calibration and evaluation"
            )
        incumbent_temperature = _temperature_receipt_for(Path(incumbent["path"]))
        report = evaluate_challenger(
            frozen_suite,
            incumbent_predictor=incumbent_predictor,
            challenger_predictor=challenger_predictor,
            incumbent_path=incumbent_path,
            challenger_path=challenger_path,
            incumbent_sha256=evaluated_incumbent_sha256,
            challenger_sha256=sha256_file(challenger_path),
            incumbent_temperature=incumbent_temperature,
            challenger_temperature=calibration_receipt_path,
            retention_epsilon=retention_epsilon,
        )
        report_challenger = report.get("challenger", {}).get("checkpoint", {})
        evaluated_challenger_sha256 = validate_sha256(
            str(report_challenger.get("sha256", "")),
            field="evaluation report challenger checkpoint sha256",
        )
        if evaluated_challenger_sha256 != expected_challenger_sha256:
            raise RuntimeError(
                "evaluation report is not bound to the trained/calibrated challenger"
            )
        verify_file_sha256(challenger_path, evaluated_challenger_sha256)
        verify_file_sha256(incumbent_path, evaluated_incumbent_sha256)
        eval_card_path = run_dir / "eval-card.json"
        _write_immutable_json(eval_card_path, report)
        eval_card_hash = file_sha256(eval_card_path)
        decision = report["decision"]["decision"]
        challenger = {
            "path": str(challenger_path.resolve()),
            "sha256": evaluated_challenger_sha256,
            "parent_sha256": incumbent["sha256"],
            "score_status": "CALIBRATED" if calibration_receipt else "UNCALIBRATED",
            "eval_card": str(eval_card_path),
            "eval_card_sha256": eval_card_hash,
        }
        event_data = {
            "run_id": run_id,
            "incumbent": incumbent,
            "challenger": challenger,
            "suite_sha256": frozen_suite.canonical_sha256,
            "eval_card_path": str(eval_card_path),
            "eval_card_sha256": eval_card_hash,
            "decision_sha256": report["decision"]["decision_sha256"],
            "reason_codes": report["decision"]["reason_codes"],
            "training_receipt_path": str(verified_training["receipt_path"]),
            "training_receipt_sha256": verified_training["receipt_sha256"],
            "calibration_receipt_path": (
                str(verified_calibration["receipt_path"])
                if verified_calibration is not None
                else None
            ),
            "calibration_receipt_sha256": (
                verified_calibration["receipt_sha256"]
                if verified_calibration is not None
                else None
            ),
            "input_artifacts": {
                "lessons_sha256": lessons_sha256,
                "held_out_vocabulary_sha256": held_out_file_sha256,
                "held_out_vocabulary_terms_sha256": verified_training[
                    "held_out_vocabulary_sha256"
                ],
                "required_held_out_vocabulary_terms_sha256": verified_training[
                    "required_held_out_vocabulary_terms_sha256"
                ],
                "required_held_out_vocabulary_artifacts_sha256": verified_training[
                    "required_held_out_vocabulary_artifacts_sha256"
                ],
                "calibration_fit_sha256": (
                    verified_calibration["calibration_fit_sha256"]
                    if verified_calibration is not None
                    else None
                ),
                "promotion_suite_file_sha256": frozen_suite.file_sha256,
                "promotion_suite_canonical_sha256": frozen_suite.canonical_sha256,
                "encoder_manifest_sha256": verified_training["encoder_manifest_sha256"],
            },
            "training_loss_used_for_promotion": False,
        }
        # The terminal ledger event and any activation must continue to point
        # at the exact bytes evaluated above. AliveStore independently checks
        # the same hash under its mutation lock for MODEL_QUALIFIED.
        if verified_calibration is not None:
            _reverify_calibration_artifacts(verified_calibration)
        verify_file_sha256(
            Path(verified_training["receipt_path"]),
            str(verified_training["receipt_sha256"]),
        )
        verify_file_sha256(lesson_path, lessons_sha256)
        verify_file_sha256(held_out_path, held_out_file_sha256)
        if encoder_manifest_file is not None and encoder_manifest_sha256 is not None:
            verify_file_sha256(encoder_manifest_file, encoder_manifest_sha256)
        training_source_manifest = verified_training["training_source_manifest"]
        if training_source_manifest is not None:
            for source in training_source_manifest:
                verify_file_sha256(
                    _resolve_artifact(source["path"]), str(source["sha256"])
                )
        verify_file_sha256(incumbent_path, evaluated_incumbent_sha256)
        verify_file_sha256(challenger_path, evaluated_challenger_sha256)
        if decision == "QUALIFY":
            # State plus its ledger decision are the authoritative activation.
            # Commit them before projecting the derived local registry so a
            # process death can never make that registry claim an unqualified
            # challenger. Journal recovery completes an interrupted decision.
            ledger_event = store.record_model_decision(
                "MODEL_QUALIFIED",
                event_data,
                active_model={
                    "path": challenger["path"],
                    "sha256": challenger["sha256"],
                },
                expected_incumbent=expected_incumbent,
            )
            terminal_event = ledger_event
            activated = True
            local_registry = _update_local_registry(
                store,
                incumbent,
                {
                    **challenger,
                    "artifact_status": "INCUMBENT",
                    "qualified_by_decision_sha256": report["decision"][
                        "decision_sha256"
                    ],
                },
                {"path": str(eval_card_path), "sha256": eval_card_hash},
            )
        else:
            ledger_event = store.record_model_decision(
                "MODEL_REJECTED",
                event_data,
                expected_incumbent=expected_incumbent,
            )
            terminal_event = ledger_event
            local_registry = store.dir / "model-registry.json"

        result = {
            "schema": "kev.evolution-result.v1",
            "run_id": run_id,
            "decision": decision,
            "incumbent": incumbent,
            "challenger": challenger,
            "eval_card": {"path": str(eval_card_path), "sha256": eval_card_hash},
            "ledger_event": ledger_event["hash"],
            "local_registry": str(local_registry) if local_registry.exists() else None,
            "candidate_preserved": True,
            "old_incumbent_preserved": True,
            "training_loss_used_for_promotion": False,
        }
        result["result_sha256"] = canonical_json_sha256(result)
        _write_immutable_json(run_dir / "evolution-result.json", result)
        return result
    except Exception as error:
        recovery_error: Exception | None = None
        if store.pending_state_event_path.exists():
            try:
                store.recover_pending_state_event()
            except Exception as pending_error:
                # Do not guess reject/qualify while a durable model-decision
                # transaction is unresolved. Preserve both failures below.
                recovery_error = pending_error
        if recovery_error is None:
            try:
                recorded_terminal = _recorded_terminal_event(store, run_id)
                if recorded_terminal is not None:
                    if (
                        terminal_event is not None
                        and terminal_event["hash"] != recorded_terminal["hash"]
                    ):
                        raise RuntimeError(
                            "in-memory and durable evolution decisions do not match"
                        )
                    terminal_event = recorded_terminal
                    activated = recorded_terminal["kind"] == "MODEL_QUALIFIED"
            except Exception as decision_error:
                # A new reject would be unsafe when the existing terminal
                # decision cannot be determined with confidence.
                recovery_error = decision_error
        stale_error = error if isinstance(error, IncumbentCompareAndSwapError) else None
        stale_incumbent = stale_error is not None
        failure = {
            "schema": "kev.evolution-failure.v1",
            "run_id": run_id,
            "status": (
                "POST_QUALIFICATION_RECORDING_FAILURE"
                if activated
                else (
                    "POST_REJECTION_RECORDING_FAILURE"
                    if terminal_event is not None
                    else (
                        "STALE_INCUMBENT_REJECTED"
                        if stale_incumbent
                        else "REJECTED_PIPELINE_FAILURE"
                    )
                )
            ),
            "incumbent": incumbent or {"status": "UNRESOLVED"},
            "lessons_path": str(lesson_path),
            "lessons_sha256": file_sha256(lesson_path)
            if lesson_path.is_file()
            else None,
            "suite_sha256": (
                frozen_suite.canonical_sha256 if frozen_suite is not None else None
            ),
            "error": {
                "type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
            },
            "compare_and_swap": (
                {
                    "expected": stale_error.expected,
                    "current": stale_error.current,
                }
                if stale_error is not None
                else None
            ),
            "decision_recovery_error": (
                {
                    "type": type(recovery_error).__name__,
                    "message": str(recovery_error),
                }
                if recovery_error is not None
                else None
            ),
            "candidate_directory": str(run_dir),
            "terminal_decision": (
                {
                    "kind": terminal_event["kind"],
                    "ledger_event": terminal_event["hash"],
                }
                if terminal_event is not None
                else None
            ),
        }
        _write_immutable_json(failure_path, failure)
        if recovery_error is not None:
            raise RuntimeError(
                f"evolution failed and its pending model decision could not be recovered; "
                f"inspect {failure_path}"
            ) from error
        if terminal_event is not None:
            postprocessing_kind = (
                "QUALIFICATION_POSTPROCESSING_FAILED"
                if activated
                else "REJECTION_POSTPROCESSING_FAILED"
            )
            store.append_event(
                postprocessing_kind,
                {
                    "run_id": run_id,
                    "terminal_decision": {
                        "kind": terminal_event["kind"],
                        "ledger_event": terminal_event["hash"],
                    },
                    "raw_failure_path": str(failure_path),
                    "raw_failure_sha256": file_sha256(failure_path),
                    "candidate_directory": str(run_dir),
                },
            )
        else:
            rejection_data: dict[str, Any] = {
                "run_id": run_id,
                "reason_codes": [
                    "STALE_INCUMBENT_COMPARE_AND_SWAP_FAILED"
                    if stale_incumbent
                    else "PIPELINE_FAILURE"
                ],
                "raw_failure_path": str(failure_path),
                "raw_failure_sha256": file_sha256(failure_path),
                "candidate_directory": str(run_dir),
            }
            if stale_error is not None:
                rejection_data["compare_and_swap"] = {
                    "expected": stale_error.expected,
                    "current": stale_error.current,
                }
            store.record_model_decision(
                "MODEL_REJECTED",
                rejection_data,
            )
        raise

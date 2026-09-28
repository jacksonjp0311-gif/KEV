"""Frozen-suite evaluation and conservative promotion decisions for KEV.

This module has no dependency on a particular model implementation.  A caller
may supply prediction records or a callback, which lets the CLI connect the
same evidence logic to the current hashed encoder or a future frozen encoder.
Training loss is intentionally absent from the promotion policy.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeAlias, cast

from kev.artifacts import (
    canonical_json_bytes,
    canonical_json_sha256,
    validate_sha256,
)
from kev.calibration_evidence import (
    CalibrationReceiptArtifact,
    load_calibration_receipt,
    validate_calibration_binding,
)


SUITE_SCHEMA = "kev.eval-suite.v1"
EVALUATION_SCHEMA = "kev.model-evaluation.v1"
COMPARISON_SCHEMA = "kev.promotion-decision.v1"
CHALLENGER_REPORT_SCHEMA = "kev.challenger-evaluation.v1"
HISTORICAL_REPLAY_SCHEMA = "kev.historical-eval-replay.v1"

REQUIRED_PROMOTION_SPLITS = ("fresh", "retention", "oov")
OPTIONAL_SPLITS = ("composition", "calibration")
KNOWN_SPLITS = REQUIRED_PROMOTION_SPLITS + OPTIONAL_SPLITS

PredictionCallback = Callable[[Mapping[str, Any]], Any]
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _evidence_path(path: str | Path) -> str:
    """Use a clone-portable path for repository artifacts."""

    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return str(resolved)


@dataclass(frozen=True)
class FrozenEvalSuite:
    """Validated suite content and both useful forms of its content hash."""

    data: dict[str, Any]
    canonical_sha256: str
    file_sha256: str | None = None
    path: str | None = None

    @property
    def suite_id(self) -> str:
        return str(self.data["id"])

    @property
    def splits(self) -> Mapping[str, list[dict[str, Any]]]:
        return self.data["splits"]


TemperatureArtifact: TypeAlias = CalibrationReceiptArtifact


def _copy_json(value: Any) -> Any:
    """Make a JSON-only defensive copy while enforcing canonicalizability."""

    return json.loads(canonical_json_bytes(value).decode("utf-8"))


def _target_for(item: Mapping[str, Any]) -> Any:
    for key in ("expected", "expected_frames", "target"):
        if key in item:
            return item[key]
    raise ValueError(
        f"evaluation item {item.get('id', '<unknown>')!r} has no expected target"
    )


def validate_frozen_suite(data: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a public evaluation suite and return a defensive JSON copy.

    A current promotion suite must contain non-empty ``fresh``, ``retention``,
    and ``oov`` splits.  Composition and calibration splits are optional, but
    become binding evidence when present.  Item identifiers are unique across
    the entire suite so predictions cannot silently move between splits.
    """

    if not isinstance(data, Mapping):
        raise TypeError("evaluation suite must be a JSON object")
    suite = _copy_json(data)
    if suite.get("schema") != SUITE_SCHEMA:
        raise ValueError(f"suite schema must be {SUITE_SCHEMA!r}")
    if suite.get("frozen") is not True:
        raise ValueError("evaluation suite must declare frozen=true")
    if not isinstance(suite.get("id"), str) or not suite["id"].strip():
        raise ValueError("evaluation suite requires a non-empty id")
    if "provenance" in suite and not isinstance(suite["provenance"], Mapping):
        raise ValueError("suite provenance must be a JSON object")
    splits = suite.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("evaluation suite requires a splits object")
    unknown = sorted(set(splits) - set(KNOWN_SPLITS))
    if unknown:
        raise ValueError(f"unknown evaluation split(s): {', '.join(unknown)}")
    missing = [name for name in REQUIRED_PROMOTION_SPLITS if name not in splits]
    if missing:
        raise ValueError(f"missing required evaluation split(s): {', '.join(missing)}")

    seen_ids: set[str] = set()
    for split_name, items in splits.items():
        if not isinstance(items, list) or not items:
            raise ValueError(f"split {split_name!r} must be a non-empty list")
        for index, item in enumerate(items):
            if not isinstance(item, Mapping):
                raise ValueError(f"{split_name}[{index}] must be a JSON object")
            item_id = item.get("id")
            if not isinstance(item_id, str) or not item_id.strip():
                raise ValueError(f"{split_name}[{index}] requires a non-empty id")
            if item_id in seen_ids:
                raise ValueError(f"duplicate evaluation item id: {item_id}")
            seen_ids.add(item_id)
            source_text = item.get("input", item.get("text"))
            if not isinstance(source_text, str) or not source_text.strip():
                raise ValueError(f"evaluation item {item_id!r} requires input or text")
            _target_for(item)
            comparison = item.get("comparison")
            if comparison not in (None, "exact", "set_exact"):
                raise ValueError(
                    f"evaluation item {item_id!r} has unsupported comparison {comparison!r}"
                )
    return suite


def frozen_suite(data: Mapping[str, Any]) -> FrozenEvalSuite:
    """Create a validated in-memory frozen suite."""

    suite = validate_frozen_suite(data)
    return FrozenEvalSuite(data=suite, canonical_sha256=canonical_json_sha256(suite))


def load_frozen_suite(
    path: str | Path,
    *,
    expected_file_sha256: str | None = None,
    expected_canonical_sha256: str | None = None,
    expected_sha256: str | None = None,
) -> FrozenEvalSuite:
    """Load, validate, and content-address a frozen JSON evaluation suite.

    ``expected_sha256`` is a convenience alias for ``expected_file_sha256``.
    Supplying both aliases with different values is rejected.
    """

    source = Path(path)
    raw = source.read_bytes()
    # Parse and hash the same captured bytes so a path replacement cannot
    # bind one suite hash to different item content.
    byte_hash = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        if expected_file_sha256 is not None and expected_file_sha256 != expected_sha256:
            raise ValueError("conflicting expected file hashes")
        expected_file_sha256 = expected_sha256
    if expected_file_sha256 is not None:
        expected = validate_sha256(expected_file_sha256, field="expected_file_sha256")
        if byte_hash != expected:
            raise ValueError(
                f"suite file SHA-256 mismatch: expected {expected}, got {byte_hash}"
            )
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON evaluation suite: {source}") from exc
    suite = validate_frozen_suite(decoded)
    canonical_hash = canonical_json_sha256(suite)
    if expected_canonical_sha256 is not None:
        expected = validate_sha256(
            expected_canonical_sha256, field="expected_canonical_sha256"
        )
        if canonical_hash != expected:
            raise ValueError(
                f"suite canonical SHA-256 mismatch: expected {expected}, got {canonical_hash}"
            )
    return FrozenEvalSuite(
        data=suite,
        canonical_sha256=canonical_hash,
        file_sha256=byte_hash,
        path=_evidence_path(source),
    )


def _coerce_suite(
    value: FrozenEvalSuite | Mapping[str, Any] | str | Path,
) -> FrozenEvalSuite:
    if isinstance(value, FrozenEvalSuite):
        current_hash = canonical_json_sha256(value.data)
        if current_hash != value.canonical_sha256:
            raise ValueError("frozen evaluation suite was mutated after validation")
        return value
    if isinstance(value, (str, Path)):
        return load_frozen_suite(value)
    return frozen_suite(value)


def load_temperature_artifact(
    value: TemperatureArtifact | Mapping[str, Any] | str | Path,
) -> TemperatureArtifact:
    """Validate a receipt-backed temperature and content-address its artifact."""
    return load_calibration_receipt(value)


def _bind_temperature_artifact(
    artifact: TemperatureArtifact,
    checkpoint: Mapping[str, Any],
    suite: FrozenEvalSuite,
    checkpoint_bytes: bytes | None,
    calibration_fit_path: str | Path | None,
) -> None:
    """Require calibration evidence to bind this checkpoint and promotion suite."""

    if checkpoint_bytes is None or not isinstance(checkpoint.get("sha256"), str):
        raise ValueError(
            "calibrated evaluation requires exact evaluated checkpoint bytes"
        )
    fit_path = calibration_fit_path
    if fit_path is None:
        receipt_fit_path = artifact.data.get("calibration_suite")
        if isinstance(receipt_fit_path, str) and receipt_fit_path.strip():
            resolved_fit_path = Path(receipt_fit_path).expanduser()
            if not resolved_fit_path.is_absolute() and artifact.path is not None:
                resolved_fit_path = Path(artifact.path).parent / resolved_fit_path
            fit_path = resolved_fit_path.resolve()
    validate_calibration_binding(
        artifact,
        checkpoint_bytes=checkpoint_bytes,
        promotion_suite_file_sha256=suite.file_sha256,
        promotion_suite_canonical_sha256=suite.canonical_sha256,
        calibration_fit_path=fit_path,
    )


def _prediction_index(
    records: Mapping[str, Any] | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    if isinstance(records, Mapping):
        return {str(key): value for key, value in records.items()}
    indexed: dict[str, Any] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("prediction records must be JSON objects")
        item_id = record.get("item_id", record.get("id"))
        if not isinstance(item_id, str) or not item_id:
            raise ValueError("prediction record requires item_id or id")
        if item_id in indexed:
            raise ValueError(f"duplicate prediction for evaluation item {item_id!r}")
        indexed[item_id] = dict(record)
    return indexed


def _softmax(logits: Sequence[float], temperature: float) -> list[float]:
    if not logits:
        raise ValueError("logits must not be empty")
    scaled = [float(value) / temperature for value in logits]
    if any(not math.isfinite(value) for value in scaled):
        raise ValueError("logits must be finite")
    maximum = max(scaled)
    exponentials = [math.exp(value - maximum) for value in scaled]
    denominator = sum(exponentials)
    return [value / denominator for value in exponentials]


def _normalise_prediction(
    raw: Any,
    calibration: TemperatureArtifact | None,
    expected_checkpoint_sha256: str | None,
) -> dict[str, Any]:
    logits: Any = None
    if isinstance(raw, Mapping):
        record = dict(raw)
        score_status = record.get("score_status")
        record_temperature = record.get("temperature")
        model_sha256 = record.get("model_sha256")
        if model_sha256 is not None:
            model_sha256 = validate_sha256(
                model_sha256, field="prediction model_sha256"
            )
            if (
                expected_checkpoint_sha256 is None
                or model_sha256 != expected_checkpoint_sha256
            ):
                raise ValueError(
                    "prediction model SHA-256 does not match the evaluated checkpoint"
                )
        sentinel = object()
        predicted = sentinel
        for key in ("predicted", "prediction", "output", "frames", "intent"):
            if key in record:
                predicted = record[key]
                break
        confidence = record.get("confidence")
        logits = record.get("logits")
        labels = record.get("labels")
        if logits is not None:
            if calibration is not None and model_sha256 is None:
                raise ValueError(
                    "calibrated prediction logits require the bound model SHA-256"
                )
            if score_status == "CALIBRATED" and calibration is not None:
                raise ValueError(
                    "prediction logits already declare calibration; refusing to apply "
                    "a temperature twice"
                )
            if isinstance(logits, Mapping):
                labels = list(logits.keys())
                values = list(logits.values())
            elif isinstance(logits, Sequence) and not isinstance(logits, (str, bytes)):
                values = list(logits)
                if not isinstance(labels, Sequence) or isinstance(labels, (str, bytes)):
                    raise ValueError("sequence logits require a labels sequence")
                labels = list(labels)
            else:
                raise ValueError("logits must be an object or sequence")
            if len(labels) != len(values) or not labels:
                raise ValueError("labels and logits must have equal non-zero lengths")
            probabilities = _softmax(
                values, calibration.temperature if calibration is not None else 1.0
            )
            winner = max(range(len(probabilities)), key=probabilities.__getitem__)
            if predicted is sentinel:
                predicted = labels[winner]
            confidence = probabilities[winner]
        if predicted is sentinel:
            # A mapping can itself be a structured prediction.  Metadata-only
            # records are not treated as predictions.
            metadata_keys = {
                "id",
                "item_id",
                "split",
                "confidence",
                "logits",
                "labels",
                "score_status",
                "temperature",
                "model_sha256",
            }
            payload = {
                key: value for key, value in record.items() if key not in metadata_keys
            }
            predicted = payload if payload else None
    else:
        predicted = raw
        confidence = None
        score_status = None
        record_temperature = None
        model_sha256 = None
    if confidence is not None and logits is None:
        if calibration is None and score_status == "CALIBRATED":
            # A checkpoint may embed a temperature, but without its bound fit
            # receipt the already-scaled score is not accepted as calibrated
            # evidence. Exact predictions remain evaluable.
            confidence = None
        elif calibration is not None:
            if score_status != "CALIBRATED":
                # A scalar cannot be correctly reapplied to an already
                # collapsed confidence. Keep the prediction, omit the score.
                confidence = None
            elif model_sha256 is None:
                # Direct confidences cannot be recomputed by the evaluator;
                # require the predictor to identify the bound checkpoint.
                confidence = None
            else:
                if isinstance(record_temperature, bool) or not isinstance(
                    record_temperature, (int, float)
                ):
                    raise ValueError(
                        "calibrated prediction confidence requires its temperature"
                    )
                if not math.isclose(
                    float(record_temperature),
                    calibration.temperature,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ):
                    raise ValueError(
                        "prediction temperature does not match the bound artifact"
                    )
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError("prediction confidence must be numeric")
        confidence = round(float(confidence), 6)
        if not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError("prediction confidence must be between zero and one")
    return {"predicted": predicted, "confidence": confidence}


def _collection(value: Any) -> list[Any] | None:
    if isinstance(value, Mapping) and isinstance(value.get("frames"), list):
        return value["frames"]
    if isinstance(value, list):
        return value
    return None


def _canonical_counter(values: Sequence[Any]) -> Counter[str]:
    return Counter(canonical_json_sha256(value) for value in values)


def _compare_target(expected: Any, predicted: Any, mode: str) -> bool:
    if mode == "set_exact":
        expected_values = _collection(expected)
        predicted_values = _collection(predicted)
        if expected_values is None or predicted_values is None:
            return False
        return _canonical_counter(expected_values) == _canonical_counter(
            predicted_values
        )
    return canonical_json_bytes(expected) == canonical_json_bytes(predicted)


def _frame_counts(expected: Any, predicted: Any) -> tuple[int, int, int] | None:
    expected_values = _collection(expected)
    predicted_values = _collection(predicted)
    if expected_values is None or predicted_values is None:
        return None
    expected_counter = _canonical_counter(expected_values)
    predicted_counter = _canonical_counter(predicted_values)
    true_positive = sum((expected_counter & predicted_counter).values())
    false_positive = sum((predicted_counter - expected_counter).values())
    false_negative = sum((expected_counter - predicted_counter).values())
    return true_positive, false_positive, false_negative


def _calibration_summary(
    rows: Sequence[Mapping[str, Any]], status: str
) -> dict[str, Any]:
    usable = [row for row in rows if row.get("confidence") is not None]
    result: dict[str, Any] = {
        "status": status,
        "observations": len(usable),
        "available": bool(usable),
    }
    if not usable:
        result["reason"] = "NO_CONFIDENCE_VALUES"
        return result
    confidences = [float(row["confidence"]) for row in usable]
    outcomes = [1.0 if row["correct"] else 0.0 for row in usable]
    bins = 10
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        members = [
            position
            for position, confidence in enumerate(confidences)
            if lower <= confidence <= upper
            and (index == bins - 1 or confidence < upper)
        ]
        if not members:
            continue
        bin_confidence = sum(confidences[position] for position in members) / len(
            members
        )
        bin_accuracy = sum(outcomes[position] for position in members) / len(members)
        ece += len(members) / len(usable) * abs(bin_accuracy - bin_confidence)
    result.update(
        {
            "mean_confidence": sum(confidences) / len(confidences),
            "accuracy": sum(outcomes) / len(outcomes),
            "expected_calibration_error_10_bin": ece,
            "top_label_brier": sum(
                (confidence - outcome) ** 2
                for confidence, outcome in zip(confidences, outcomes)
            )
            / len(usable),
        }
    )
    return result


def _checkpoint_record(
    checkpoint_path: str | Path | None,
    checkpoint_sha256: str | None,
    checkpoint_bytes: bytes | None = None,
) -> dict[str, Any]:
    if checkpoint_path is not None:
        source = Path(checkpoint_path)
        payload = source.read_bytes() if checkpoint_bytes is None else checkpoint_bytes
        actual = hashlib.sha256(payload).hexdigest()
        if checkpoint_sha256 is not None:
            expected = validate_sha256(checkpoint_sha256, field="checkpoint_sha256")
            if actual != expected:
                raise ValueError(
                    f"checkpoint SHA-256 mismatch: expected {expected}, got {actual}"
                )
        return {
            "path": _evidence_path(source),
            "sha256": actual,
            "size_bytes": len(payload),
        }
    if checkpoint_sha256 is not None:
        return {
            "path": None,
            "sha256": validate_sha256(checkpoint_sha256, field="checkpoint_sha256"),
            "size_bytes": None,
        }
    return {"path": None, "sha256": None, "size_bytes": None, "status": "NOT_SUPPLIED"}


def evaluate_suite(
    suite: FrozenEvalSuite | Mapping[str, Any] | str | Path,
    *,
    prediction_records: Mapping[str, Any] | Iterable[Mapping[str, Any]] | None = None,
    predictor: PredictionCallback | None = None,
    checkpoint_path: str | Path | None = None,
    checkpoint_sha256: str | None = None,
    temperature_artifact: TemperatureArtifact
    | Mapping[str, Any]
    | str
    | Path
    | None = None,
    calibration_fit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate all frozen items from records or a model-agnostic callback.

    The predictor receives a copy of the suite item plus a ``split`` field,
    with expected targets and comparison policy removed so the callback cannot
    accidentally train on or echo the answer key.  It may return a bare
    prediction or an object containing ``predicted`` (or ``prediction``),
    optional ``confidence``, and optional ``logits``/``labels``.  Missing
    predictions and callback exceptions are retained as raw failures.
    """

    if (prediction_records is None) == (predictor is None):
        raise ValueError("provide exactly one of prediction_records or predictor")
    frozen = _coerce_suite(suite)
    index = (
        _prediction_index(prediction_records)
        if prediction_records is not None
        else None
    )
    known_ids = {item["id"] for items in frozen.splits.values() for item in items}
    if index is not None:
        unknown_ids = sorted(set(index) - known_ids)
        if unknown_ids:
            raise ValueError(
                f"predictions contain unknown item id(s): {', '.join(unknown_ids)}"
            )
    calibration = (
        load_temperature_artifact(temperature_artifact)
        if temperature_artifact is not None
        else None
    )
    checkpoint_bytes = (
        Path(checkpoint_path).read_bytes() if checkpoint_path is not None else None
    )
    checkpoint = _checkpoint_record(
        checkpoint_path, checkpoint_sha256, checkpoint_bytes
    )
    if calibration is not None:
        _bind_temperature_artifact(
            calibration,
            checkpoint,
            frozen,
            checkpoint_bytes,
            calibration_fit_path,
        )
    calibration_status = "CALIBRATED" if calibration is not None else "UNCALIBRATED"

    all_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    split_metrics: dict[str, dict[str, Any]] = {}
    for split_name, items in frozen.splits.items():
        split_rows: list[dict[str, Any]] = []
        frame_tp = frame_fp = frame_fn = 0
        for item in items:
            item_id = item["id"]
            expected = _target_for(item)
            mode = item.get(
                "comparison",
                "set_exact"
                if split_name == "composition" and _collection(expected) is not None
                else "exact",
            )
            error: dict[str, str] | None = None
            if index is not None and item_id not in index:
                raw_prediction = None
                error = {
                    "kind": "MISSING_PREDICTION",
                    "message": "no prediction supplied",
                }
            else:
                try:
                    if predictor is not None:
                        callback_item = _copy_json(item)
                        for answer_key in (
                            "expected",
                            "expected_frames",
                            "target",
                            "comparison",
                            "audit",
                            "pair_id",
                        ):
                            callback_item.pop(answer_key, None)
                        callback_item["split"] = split_name
                        raw_prediction = predictor(callback_item)
                    else:
                        raw_prediction = index[item_id]  # type: ignore[index]
                except Exception as exc:  # raw model failures are evaluation evidence
                    raw_prediction = None
                    error = {"kind": type(exc).__name__, "message": str(exc)}
            try:
                normalized = _normalise_prediction(
                    raw_prediction,
                    calibration,
                    checkpoint.get("sha256"),
                )
            except Exception as exc:
                normalized = {"predicted": None, "confidence": None}
                error = {"kind": type(exc).__name__, "message": str(exc)}
            predicted = normalized["predicted"]
            correct = error is None and _compare_target(expected, predicted, mode)
            row: dict[str, Any] = {
                "item_id": item_id,
                "split": split_name,
                "expected": expected,
                "predicted": predicted,
                "comparison": mode,
                "correct": correct,
                "confidence": normalized["confidence"],
            }
            if item.get("audit") is not None:
                row["audit"] = item["audit"]
            if item.get("pair_id") is not None:
                row["pair_id"] = item["pair_id"]
            if error is not None:
                row["error"] = error
            split_rows.append(row)
            all_rows.append(row)
            counts = _frame_counts(expected, predicted)
            if counts is not None:
                frame_tp += counts[0]
                frame_fp += counts[1]
                frame_fn += counts[2]
            if not correct:
                failure = {
                    "item_id": item_id,
                    "split": split_name,
                    "input": item.get("input", item.get("text")),
                    "expected": expected,
                    "predicted": predicted,
                    "comparison": mode,
                }
                if item.get("audit") is not None:
                    failure["audit"] = item["audit"]
                if item.get("pair_id") is not None:
                    failure["pair_id"] = item["pair_id"]
                if error is not None:
                    failure["error"] = error
                failures.append(failure)
        correct_count = sum(1 for row in split_rows if row["correct"])
        metric: dict[str, Any] = {
            "total": len(split_rows),
            "correct": correct_count,
            "failures": len(split_rows) - correct_count,
            "primary_metric": "exact_match_accuracy",
            "primary_score": correct_count / len(split_rows),
            "accuracy": correct_count / len(split_rows),
            "calibration": _calibration_summary(split_rows, calibration_status),
        }
        if frame_tp + frame_fp + frame_fn:
            precision = frame_tp / (frame_tp + frame_fp) if frame_tp + frame_fp else 0.0
            recall = frame_tp / (frame_tp + frame_fn) if frame_tp + frame_fn else 0.0
            metric["frame_micro"] = {
                "true_positive": frame_tp,
                "false_positive": frame_fp,
                "false_negative": frame_fn,
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0,
            }
        split_metrics[split_name] = metric

    prediction_evidence = [
        {
            "item_id": row["item_id"],
            "split": row["split"],
            "predicted": row["predicted"],
            "confidence": row["confidence"],
            **({"error": row["error"]} if "error" in row else {}),
        }
        for row in all_rows
    ]
    audit_metrics: dict[str, dict[str, Any]] = {}
    for audit in sorted(
        {str(row["audit"]) for row in all_rows if row.get("audit") is not None}
    ):
        rows = [row for row in all_rows if row.get("audit") == audit]
        audit_correct = sum(1 for row in rows if row["correct"])
        audit_metrics[audit] = {
            "total": len(rows),
            "correct": audit_correct,
            "failures": len(rows) - audit_correct,
            "accuracy": audit_correct / len(rows),
        }
    report: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "suite": {
            "id": frozen.suite_id,
            "canonical_sha256": frozen.canonical_sha256,
            "file_sha256": frozen.file_sha256,
            "path": frozen.path,
            "frozen": True,
        },
        "checkpoint": checkpoint,
        "prediction_evidence_sha256": canonical_json_sha256(prediction_evidence),
        "splits": split_metrics,
        "audit_metrics": audit_metrics,
        "calibration": {
            **_calibration_summary(all_rows, calibration_status),
            "temperature": calibration.temperature if calibration is not None else None,
            "temperature_artifact_canonical_sha256": (
                calibration.canonical_sha256 if calibration is not None else None
            ),
            "temperature_artifact_file_sha256": (
                calibration.file_sha256 if calibration is not None else None
            ),
        },
        "records": all_rows,
        "raw_failures": {
            "available": True,
            "count": len(failures),
            "items": failures,
        },
    }
    if prediction_records is not None:
        report["prediction_authority"] = "EXTERNAL_RECORDS_UNVERIFIED"
    report["report_sha256"] = canonical_json_sha256(report)
    return report


# Public alias spelling out that supplied prediction evidence is supported.
evaluate_predictions = evaluate_suite


def _metric_fraction(metric: Mapping[str, Any]) -> Fraction:
    correct = metric.get("correct")
    total = metric.get("total")
    if (
        isinstance(correct, int)
        and not isinstance(correct, bool)
        and isinstance(total, int)
        and not isinstance(total, bool)
        and total > 0
        and 0 <= correct <= total
    ):
        return Fraction(correct, total)
    score = metric.get("primary_score", metric.get("accuracy"))
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError(
            "split metric requires valid correct/total counts or a numeric score"
        )
    score_float = float(score)
    if not math.isfinite(score_float) or not 0 <= score_float <= 1:
        raise ValueError("split score must be between zero and one")
    return Fraction(str(score_float))


def _gate(
    name: str,
    incumbent: Mapping[str, Any] | None,
    challenger: Mapping[str, Any] | None,
    *,
    relation: str,
    epsilon: Fraction = Fraction(0),
    require_item_counts: bool = False,
) -> dict[str, Any]:
    if not isinstance(incumbent, Mapping) or not isinstance(challenger, Mapping):
        return {
            "name": name,
            "passed": False,
            "reason_code": f"{name.upper()}_METRICS_MISSING",
            "reason": f"{name} metrics are required for both checkpoints",
        }
    if require_item_counts:
        required = (
            incumbent.get("correct"),
            incumbent.get("total"),
            challenger.get("correct"),
            challenger.get("total"),
        )
        if any(value is None for value in required):
            return {
                "name": name,
                "passed": False,
                "reason_code": f"{name.upper()}_METRICS_MISSING",
                "reason": f"{name} correct and total counts are required for both checkpoints",
            }
        if any(
            isinstance(value, bool) or not isinstance(value, int) for value in required
        ):
            return {
                "name": name,
                "passed": False,
                "reason_code": f"{name.upper()}_METRICS_INVALID",
                "reason": f"{name} correct and total counts must be integers",
            }
        old_correct = cast(int, required[0])
        old_total = cast(int, required[1])
        new_correct = cast(int, required[2])
        new_total = cast(int, required[3])
        if (
            old_total <= 0
            or new_total <= 0
            or old_correct < 0
            or new_correct < 0
            or old_correct > old_total
            or new_correct > new_total
        ):
            return {
                "name": name,
                "passed": False,
                "reason_code": f"{name.upper()}_METRICS_INVALID",
                "reason": f"{name} correct and total counts are outside their valid range",
            }
    incumbent_total = incumbent.get("total")
    challenger_total = challenger.get("total")
    if incumbent_total != challenger_total:
        return {
            "name": name,
            "passed": False,
            "reason_code": f"{name.upper()}_SUITE_SIZE_MISMATCH",
            "reason": f"{name} item totals differ between checkpoints",
            "incumbent_total": incumbent_total,
            "challenger_total": challenger_total,
        }
    old = _metric_fraction(incumbent)
    new = _metric_fraction(challenger)
    if relation == "strict":
        passed = new > old
        if new == old:
            reason_code = f"{name.upper()}_TIE"
            reason = f"{name} tied; strict improvement is required"
        elif passed:
            reason_code = f"{name.upper()}_IMPROVED"
            reason = f"{name} strictly improved"
        else:
            reason_code = f"{name.upper()}_REGRESSED"
            reason = f"{name} did not strictly improve"
    elif relation == "no_regression":
        passed = new >= old
        reason_code = (
            f"{name.upper()}_NO_REGRESSION" if passed else f"{name.upper()}_REGRESSED"
        )
        reason = f"{name} did not regress" if passed else f"{name} regressed"
    elif relation == "epsilon":
        passed = new + epsilon >= old
        reason_code = (
            f"{name.upper()}_WITHIN_EPSILON"
            if passed
            else f"{name.upper()}_REGRESSION_EXCEEDS_EPSILON"
        )
        reason = (
            f"{name} regression is within epsilon"
            if passed
            else f"{name} regression exceeds epsilon"
        )
    else:
        raise ValueError(f"unknown gate relation: {relation}")
    return {
        "name": name,
        "passed": passed,
        "reason_code": reason_code,
        "reason": reason,
        "incumbent": float(old),
        "challenger": float(new),
        "delta": float(new - old),
        **({"epsilon": float(epsilon)} if relation == "epsilon" else {}),
    }


def promotion_decision(
    incumbent_splits: Mapping[str, Mapping[str, Any]],
    challenger_splits: Mapping[str, Mapping[str, Any]],
    *,
    retention_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Apply KEV's promotion gate; all gates must pass.

    Fresh and (when present) composition require strict improvement.  Retention
    may regress only by the declared non-negative epsilon.  OOV may not regress.
    A tie on a strict gate rejects.
    """

    if isinstance(retention_epsilon, bool) or not isinstance(
        retention_epsilon, (int, float)
    ):
        raise TypeError("retention_epsilon must be numeric")
    if not math.isfinite(float(retention_epsilon)) or retention_epsilon < 0:
        raise ValueError("retention_epsilon must be finite and non-negative")
    epsilon = Fraction(str(float(retention_epsilon)))
    gates = [
        _gate(
            "fresh",
            incumbent_splits.get("fresh"),
            challenger_splits.get("fresh"),
            relation="strict",
        ),
        _gate(
            "retention",
            incumbent_splits.get("retention"),
            challenger_splits.get("retention"),
            relation="epsilon",
            epsilon=epsilon,
        ),
        _gate(
            "oov",
            incumbent_splits.get("oov"),
            challenger_splits.get("oov"),
            relation="no_regression",
        ),
    ]
    if "composition" in incumbent_splits or "composition" in challenger_splits:
        gates.append(
            _gate(
                "composition",
                incumbent_splits.get("composition"),
                challenger_splits.get("composition"),
                relation="strict",
            )
        )
    passed = all(gate["passed"] for gate in gates)
    failed = [gate for gate in gates if not gate["passed"]]
    result: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "decision": "QUALIFY" if passed else "REJECT",
        "promote": passed,
        "policy": {
            "fresh": "STRICT_IMPROVEMENT",
            "retention": "NO_REGRESSION_BEYOND_EPSILON",
            "retention_epsilon": float(epsilon),
            "oov": "NO_REGRESSION",
            "composition": "STRICT_IMPROVEMENT_WHEN_PRESENT",
            "tie": "REJECT",
            "training_loss_used": False,
        },
        "gates": gates,
        "reason_codes": [gate["reason_code"] for gate in failed]
        if failed
        else ["ALL_PROMOTION_GATES_PASSED"],
        "reasons": [gate["reason"] for gate in failed]
        if failed
        else ["all promotion gates passed"],
        "evidence_hashes": {
            "incumbent_splits_sha256": canonical_json_sha256(incumbent_splits),
            "challenger_splits_sha256": canonical_json_sha256(challenger_splits),
        },
        "raw_failures": {
            "available": False,
            "reason": "AGGREGATE_SPLIT_METRICS_ONLY",
            "incumbent": [],
            "challenger": [],
        },
    }
    result["decision_sha256"] = canonical_json_sha256(result)
    return result


def compare_evaluations(
    incumbent_report: Mapping[str, Any],
    challenger_report: Mapping[str, Any],
    *,
    retention_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Compare reports from one frozen suite, including each audit family."""

    def evidence_hash(report: Mapping[str, Any]) -> str:
        body = dict(report)
        claimed = body.pop("report_sha256", None)
        actual = canonical_json_sha256(body)
        if claimed is not None and claimed != actual:
            raise ValueError("evaluation report SHA-256 does not match its content")
        return actual

    incumbent_report_hash = evidence_hash(incumbent_report)
    challenger_report_hash = evidence_hash(challenger_report)
    incumbent_suite = incumbent_report.get("suite", {})
    challenger_suite = challenger_report.get("suite", {})
    incumbent_hash = incumbent_suite.get("canonical_sha256")
    challenger_hash = challenger_suite.get("canonical_sha256")
    if not incumbent_hash or incumbent_hash != challenger_hash:
        mismatch: dict[str, Any] = {
            "schema": COMPARISON_SCHEMA,
            "decision": "REJECT",
            "promote": False,
            "policy": {"same_frozen_suite_required": True, "training_loss_used": False},
            "gates": [],
            "reason_codes": ["SUITE_HASH_MISMATCH"],
            "reasons": [
                "incumbent and challenger were not evaluated on the same frozen suite"
            ],
            "incumbent_suite_sha256": incumbent_hash,
            "challenger_suite_sha256": challenger_hash,
            "evidence_hashes": {
                "incumbent_report_sha256": incumbent_report_hash,
                "challenger_report_sha256": challenger_report_hash,
            },
            "raw_failures": {
                "incumbent": incumbent_report.get("raw_failures"),
                "challenger": challenger_report.get("raw_failures"),
            },
        }
        mismatch["decision_sha256"] = canonical_json_sha256(mismatch)
        return mismatch
    result = promotion_decision(
        incumbent_report.get("splits", {}),
        challenger_report.get("splits", {}),
        retention_epsilon=retention_epsilon,
    )
    incumbent_audits = incumbent_report.get("audit_metrics", {})
    challenger_audits = challenger_report.get("audit_metrics", {})
    if not isinstance(incumbent_audits, Mapping):
        incumbent_audits = {}
    if not isinstance(challenger_audits, Mapping):
        challenger_audits = {}
    audit_families = sorted(set(incumbent_audits) | set(challenger_audits))
    audit_gates: list[dict[str, Any]] = []
    for family in audit_families:
        reason_name = "audit_" + "".join(
            character if character.isalnum() else "_" for character in family
        ).strip("_")
        gate = _gate(
            reason_name,
            incumbent_audits.get(family),
            challenger_audits.get(family),
            relation="no_regression",
            require_item_counts=True,
        )
        gate["name"] = f"audit:{family}"
        gate["audit_family"] = family
        audit_gates.append(gate)
    result["gates"].extend(audit_gates)
    result["policy"]["audit_families"] = (
        "NO_REGRESSION_FOR_EVERY_REPORTED_FAMILY; MISSING_OR_SIZE_MISMATCH_REJECTS"
    )
    result["policy"]["audit_family_count"] = len(audit_families)
    external_prediction_evidence = any(
        report.get("prediction_authority") == "EXTERNAL_RECORDS_UNVERIFIED"
        for report in (incumbent_report, challenger_report)
    )
    if external_prediction_evidence:
        result["gates"].append(
            {
                "name": "prediction_authority",
                "passed": False,
                "reason_code": "EXTERNAL_PREDICTIONS_PROMOTION_INELIGIBLE",
                "reason": (
                    "caller-supplied prediction records were not produced by the "
                    "checkpoint-bound evaluator"
                ),
            }
        )
        result["policy"]["checkpoint_bound_predictions_required"] = True
    failed = [gate for gate in result["gates"] if not gate["passed"]]
    result["promote"] = not failed
    result["decision"] = "QUALIFY" if result["promote"] else "REJECT"
    result["reason_codes"] = (
        [gate["reason_code"] for gate in failed]
        if failed
        else ["ALL_PROMOTION_GATES_PASSED"]
    )
    result["reasons"] = (
        [gate["reason"] for gate in failed]
        if failed
        else ["all promotion gates passed"]
    )
    result["suite_canonical_sha256"] = incumbent_hash
    result["evidence_hashes"] = {
        **result["evidence_hashes"],
        "incumbent_report_sha256": incumbent_report_hash,
        "challenger_report_sha256": challenger_report_hash,
    }
    result["raw_failures"] = {
        "incumbent": incumbent_report.get("raw_failures"),
        "challenger": challenger_report.get("raw_failures"),
    }
    result["calibration_status"] = {
        "incumbent": incumbent_report.get("calibration", {}).get(
            "status", "UNCALIBRATED"
        ),
        "challenger": challenger_report.get("calibration", {}).get(
            "status", "UNCALIBRATED"
        ),
    }
    # Refresh after adding cross-report evidence.
    result.pop("decision_sha256", None)
    result["decision_sha256"] = canonical_json_sha256(result)
    return result


def evaluate_challenger(
    suite: FrozenEvalSuite | Mapping[str, Any] | str | Path,
    *,
    incumbent_prediction_records: Mapping[str, Any]
    | Iterable[Mapping[str, Any]]
    | None = None,
    challenger_prediction_records: Mapping[str, Any]
    | Iterable[Mapping[str, Any]]
    | None = None,
    incumbent_predictor: PredictionCallback | None = None,
    challenger_predictor: PredictionCallback | None = None,
    incumbent_path: str | Path | None = None,
    challenger_path: str | Path | None = None,
    incumbent_sha256: str | None = None,
    challenger_sha256: str | None = None,
    incumbent_temperature: TemperatureArtifact
    | Mapping[str, Any]
    | str
    | Path
    | None = None,
    challenger_temperature: TemperatureArtifact
    | Mapping[str, Any]
    | str
    | Path
    | None = None,
    retention_epsilon: float = 0.0,
) -> dict[str, Any]:
    """Evaluate and compare an incumbent/challenger pair without activating it."""

    frozen = _coerce_suite(suite)
    incumbent = evaluate_suite(
        frozen,
        prediction_records=incumbent_prediction_records,
        predictor=incumbent_predictor,
        checkpoint_path=incumbent_path,
        checkpoint_sha256=incumbent_sha256,
        temperature_artifact=incumbent_temperature,
    )
    challenger = evaluate_suite(
        frozen,
        prediction_records=challenger_prediction_records,
        predictor=challenger_predictor,
        checkpoint_path=challenger_path,
        checkpoint_sha256=challenger_sha256,
        temperature_artifact=challenger_temperature,
    )
    decision = compare_evaluations(
        incumbent, challenger, retention_epsilon=retention_epsilon
    )
    report: dict[str, Any] = {
        "schema": CHALLENGER_REPORT_SCHEMA,
        "suite_canonical_sha256": frozen.canonical_sha256,
        "incumbent": incumbent,
        "challenger": challenger,
        "decision": decision,
        "raw_failures": {
            "incumbent": incumbent["raw_failures"],
            "challenger": challenger["raw_failures"],
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report


def replay_historical_aggregate(
    *,
    parent_correct: int = 190,
    challenger_correct: int = 190,
    total: int = 260,
    suite_id: str = "published-audit-260",
) -> dict[str, Any]:
    """Replay the published aggregate tie without inventing absent item data.

    This helper is deliberately not a current-policy promotion evaluator: the
    historical record has no public item texts, raw predictions, retention,
    OOV, or composition splits.  It can faithfully replay only the documented
    strict fresh-score comparison.
    """

    for name, value in (
        ("parent_correct", parent_correct),
        ("challenger_correct", challenger_correct),
        ("total", total),
    ):
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be an integer")
    if (
        total <= 0
        or not 0 <= parent_correct <= total
        or not 0 <= challenger_correct <= total
    ):
        raise ValueError("historical aggregate counts are out of range")
    aggregate = {
        "suite_id": suite_id,
        "split": "fresh",
        "parent": {"correct": parent_correct, "total": total},
        "challenger": {"correct": challenger_correct, "total": total},
        "source": "published aggregate; item-level prompts and predictions unavailable",
    }
    old = Fraction(parent_correct, total)
    new = Fraction(challenger_correct, total)
    improved = new > old
    if new == old:
        reason_code = "FRESH_TIE"
        reason = "fresh aggregate tied; strict improvement is required"
    elif improved:
        reason_code = "FRESH_IMPROVED"
        reason = "fresh aggregate strictly improved"
    else:
        reason_code = "FRESH_REGRESSED"
        reason = "fresh aggregate regressed"
    report: dict[str, Any] = {
        "schema": HISTORICAL_REPLAY_SCHEMA,
        "evidence_kind": "HISTORICAL_AGGREGATE_ONLY",
        "aggregate": aggregate,
        "aggregate_sha256": canonical_json_sha256(aggregate),
        "checkpoint_hashes": {
            "parent": None,
            "challenger": None,
            "status": "NOT_PRESENT_IN_PUBLISHED_AGGREGATE",
        },
        "scores": {
            "parent": float(old),
            "challenger": float(new),
            "delta": float(new - old),
        },
        "decision": "QUALIFY" if improved else "REJECT",
        "promote": improved,
        "reason_codes": [reason_code],
        "reasons": [reason],
        "policy": {
            "scope": "HISTORICAL_FRESH_AGGREGATE_REPLAY",
            "fresh": "STRICT_IMPROVEMENT",
            "tie": "REJECT",
            "eligible_under_current_full_policy": False,
            "training_loss_used": False,
        },
        "calibration": {
            "status": "UNCALIBRATED",
            "reason": "NO_TEMPERATURE_ARTIFACT_IN_PUBLISHED_AGGREGATE",
        },
        "raw_failures": {
            "available": False,
            "items": [],
            "reason": "ITEM_LEVEL_PROMPTS_AND_PREDICTIONS_NOT_PUBLISHED",
        },
    }
    report["report_sha256"] = canonical_json_sha256(report)
    return report

"""Validation for receipt-bound temperature calibration evidence.

An embedded temperature is only checkpoint metadata.  It becomes trusted
calibration evidence after a ``kev.calibration-receipt.v1`` record is checked
against the exact checkpoint bytes, calibration-fit bytes (when supplied),
and promotion-suite hashes.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from kev.artifacts import canonical_json_sha256, file_sha256, validate_sha256


CALIBRATION_RECEIPT_SCHEMA = "kev.calibration-receipt.v1"
CALIBRATION_METADATA_FIELDS = frozenset(
    {
        "temperature",
        "score_status",
        "calibration_suite_sha256",
        "calibration_parent_sha256",
        "promotion_suite_file_sha256",
        "promotion_suite_canonical_sha256",
    }
)


@dataclass(frozen=True)
class CalibrationReceiptArtifact:
    """A structurally valid, content-addressed calibration receipt."""

    data: dict[str, Any]
    temperature: float
    canonical_sha256: str
    file_sha256: str | None = None
    path: str | None = None


def _json_copy(value: Mapping[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(
            json.dumps(
                value,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    except (TypeError, ValueError) as error:
        raise ValueError(
            "calibration receipt must contain only finite JSON values"
        ) from error


def _temperature(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{field} must be finite and greater than zero")
    return result


def _required_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} is required")
    return validate_sha256(value, field=field)


def load_calibration_receipt(
    value: CalibrationReceiptArtifact | Mapping[str, Any] | str | Path,
) -> CalibrationReceiptArtifact:
    """Load and structurally validate a calibration receipt.

    Structural validation is intentionally separate from binding validation:
    callers must also invoke :func:`validate_calibration_binding` with the
    exact checkpoint and promotion-suite hashes before trusting the scalar.
    """

    if isinstance(value, CalibrationReceiptArtifact):
        current_hash = canonical_json_sha256(value.data)
        if current_hash != value.canonical_sha256:
            raise ValueError("calibration receipt was mutated after validation")
        validated = load_calibration_receipt(value.data)
        return CalibrationReceiptArtifact(
            data=validated.data,
            temperature=validated.temperature,
            canonical_sha256=validated.canonical_sha256,
            file_sha256=value.file_sha256,
            path=value.path,
        )

    receipt_path: str | None = None
    byte_hash: str | None = None
    if isinstance(value, (str, Path)):
        source = Path(value).expanduser().resolve()
        payload = source.read_bytes()
        byte_hash = hashlib.sha256(payload).hexdigest()
        receipt_path = str(source)
        try:
            decoded = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid UTF-8 JSON calibration receipt: {source}"
            ) from error
    else:
        decoded = value
    if not isinstance(decoded, Mapping):
        raise ValueError("calibration receipt must be a JSON object")
    receipt = _json_copy(decoded)
    if receipt.get("schema") != CALIBRATION_RECEIPT_SCHEMA:
        raise ValueError(
            "calibrated evidence requires a receipt with schema "
            f"{CALIBRATION_RECEIPT_SCHEMA!r}; a generic temperature is insufficient"
        )

    temperature = _temperature(receipt.get("temperature"), field="temperature")
    fit_data_hash = receipt.get("fit_data_sha256")
    calibration_suite_hash = receipt.get("calibration_suite_sha256")
    fit_hash = fit_data_hash if fit_data_hash is not None else calibration_suite_hash
    if fit_hash is None:
        raise ValueError("calibration receipt requires a calibration data SHA-256")
    normalized_fit_hash = validate_sha256(fit_hash, field="calibration data SHA-256")
    if (
        calibration_suite_hash is not None
        and validate_sha256(calibration_suite_hash, field="calibration_suite_sha256")
        != normalized_fit_hash
    ):
        raise ValueError("calibration receipt fit-data SHA-256 fields disagree")
    for field in ("source_sha256", "output_sha256"):
        _required_sha256(receipt.get(field), field=field)
    if receipt.get("calibration_fit_split") != "calibration_fit":
        raise ValueError(
            "calibration receipt must identify a held-out calibration_fit split"
        )
    if receipt.get("score_status") != "CALIBRATED":
        raise ValueError("calibration receipt must declare score_status=CALIBRATED")
    for field in (
        "promotion_suite_sha256",
        "promotion_suite_file_sha256",
        "promotion_suite_canonical_sha256",
    ):
        if receipt.get(field) is not None:
            validate_sha256(receipt[field], field=field)
    if not any(
        receipt.get(field) is not None
        for field in ("promotion_suite_file_sha256", "promotion_suite_sha256")
    ):
        raise ValueError("calibration receipt requires a promotion-suite file SHA-256")
    if receipt.get("promotion_suite_canonical_sha256") is None:
        raise ValueError(
            "calibration receipt requires a promotion-suite canonical SHA-256"
        )

    return CalibrationReceiptArtifact(
        data=receipt,
        temperature=temperature,
        canonical_sha256=canonical_json_sha256(receipt),
        file_sha256=byte_hash,
        path=receipt_path,
    )


def _checkpoint_mapping(
    checkpoint_bytes: bytes, *, role: str = "calibrated"
) -> Mapping[str, Any]:
    try:
        checkpoint = torch.load(
            io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
        )
    except Exception as error:
        raise ValueError(
            f"{role} checkpoint is not a readable Torch artifact"
        ) from error
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{role} checkpoint must contain a mapping")
    return checkpoint


def _checkpoint_key_equal(source: Any, child: Any) -> bool:
    """Compare mapping keys without Python's bool/int key aliasing."""

    if type(source) is not type(child):
        return False
    if not isinstance(source, (str, bytes, int, float, bool, type(None))):
        raise ValueError(
            "checkpoint mappings require scalar string/number/bytes/null keys"
        )
    if isinstance(source, float) and math.isnan(source):
        return isinstance(child, float) and math.isnan(child)
    return bool(source == child)


def _matching_checkpoint_key(key: Any, candidates: Sequence[Any]) -> Any | None:
    matches = [
        candidate for candidate in candidates if _checkpoint_key_equal(key, candidate)
    ]
    if len(matches) > 1:
        raise ValueError("checkpoint mapping contains ambiguous keys")
    return matches[0] if matches else None


def _checkpoint_path(parent: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{parent}.{key}"
    return f"{parent}[{key!r}]"


def _assert_checkpoint_content_equal(source: Any, child: Any, *, path: str) -> None:
    """Recursively prove that non-calibration checkpoint content is unchanged."""

    if isinstance(source, torch.Tensor) or isinstance(child, torch.Tensor):
        if not isinstance(source, torch.Tensor) or not isinstance(child, torch.Tensor):
            raise ValueError(
                f"non-calibration checkpoint content type differs at {path}"
            )
        if type(source) is not type(child):
            raise ValueError(
                f"non-calibration checkpoint tensor type differs at {path}"
            )
        tensor_metadata_matches = (
            source.dtype == child.dtype
            and source.layout == child.layout
            and source.shape == child.shape
            and source.requires_grad == child.requires_grad
        )
        if source.layout == torch.strided and child.layout == torch.strided:
            tensor_metadata_matches = (
                tensor_metadata_matches
                and source.stride() == child.stride()
                and source.storage_offset() == child.storage_offset()
            )
        tensor_values_match = False
        if tensor_metadata_matches:
            if source.layout == torch.strided and not source.is_quantized:
                source_bytes = (
                    source.detach().contiguous().reshape(-1).view(torch.uint8)
                )
                child_bytes = child.detach().contiguous().reshape(-1).view(torch.uint8)
                tensor_values_match = torch.equal(source_bytes, child_bytes)
            else:
                tensor_values_match = torch.equal(source, child)
        if not tensor_metadata_matches or not tensor_values_match:
            raise ValueError(f"non-calibration checkpoint tensor differs at {path}")
        return

    if isinstance(source, Mapping) or isinstance(child, Mapping):
        if not isinstance(source, Mapping) or not isinstance(child, Mapping):
            raise ValueError(
                f"non-calibration checkpoint content type differs at {path}"
            )
        source_keys = list(source.keys())
        child_keys = list(child.keys())
        if len(source_keys) != len(child_keys):
            raise ValueError(
                f"non-calibration checkpoint mapping keys differ at {path}"
            )
        for source_key in source_keys:
            child_key = _matching_checkpoint_key(source_key, child_keys)
            if child_key is None:
                raise ValueError(
                    f"non-calibration checkpoint mapping keys differ at {path}"
                )
            _assert_checkpoint_content_equal(
                source[source_key],
                child[child_key],
                path=_checkpoint_path(path, source_key),
            )
        return

    sequence_types = (list, tuple)
    if isinstance(source, sequence_types) or isinstance(child, sequence_types):
        if type(source) is not type(child):
            raise ValueError(
                f"non-calibration checkpoint sequence type differs at {path}"
            )
        if len(source) != len(child):
            raise ValueError(
                f"non-calibration checkpoint sequence length differs at {path}"
            )
        for index, (source_item, child_item) in enumerate(zip(source, child)):
            _assert_checkpoint_content_equal(
                source_item, child_item, path=f"{path}[{index}]"
            )
        return

    scalar_types = (str, bytes, int, float, complex, bool, type(None))
    if isinstance(source, scalar_types) or isinstance(child, scalar_types):
        if type(source) is not type(child):
            raise ValueError(
                f"non-calibration checkpoint scalar type differs at {path}"
            )
        if isinstance(source, (float, complex)):
            source_nan = math.isnan(source.real) or (
                isinstance(source, complex) and math.isnan(source.imag)
            )
            child_nan = math.isnan(child.real) or (
                isinstance(child, complex) and math.isnan(child.imag)
            )
            if source_nan or child_nan:
                if source_nan and child_nan and repr(source) == repr(child):
                    return
                raise ValueError(f"non-calibration checkpoint scalar differs at {path}")
        if source != child:
            raise ValueError(f"non-calibration checkpoint scalar differs at {path}")
        return

    if type(source) is not type(child):
        raise ValueError(f"non-calibration checkpoint content type differs at {path}")
    try:
        equal = source == child
    except Exception as error:
        raise ValueError(
            f"unsupported checkpoint value at {path}: {type(source).__name__}"
        ) from error
    if not isinstance(equal, bool) or not equal:
        raise ValueError(f"non-calibration checkpoint value differs at {path}")


def _assert_calibration_only_change(
    source: Mapping[str, Any], child: Mapping[str, Any]
) -> None:
    """Require a schema-compatible child with only calibration metadata changed."""

    source_schema = source.get("schema")
    child_schema = child.get("schema")
    if not isinstance(source_schema, str) or not source_schema.strip():
        raise ValueError("source checkpoint requires a non-empty schema")
    if child_schema != source_schema:
        raise ValueError("source and calibrated checkpoint schemas are incompatible")
    if (
        source.get("temperature") is not None
        or source.get("score_status") == "CALIBRATED"
    ):
        raise ValueError("calibration source checkpoint must be uncalibrated")
    if any(not isinstance(key, str) for key in source) or any(
        not isinstance(key, str) for key in child
    ):
        raise ValueError("checkpoint root mappings require string keys")

    source_content = {
        key: value
        for key, value in source.items()
        if key not in CALIBRATION_METADATA_FIELDS
    }
    child_content = {
        key: value
        for key, value in child.items()
        if key not in CALIBRATION_METADATA_FIELDS
    }
    _assert_checkpoint_content_equal(source_content, child_content, path="checkpoint")


def _receipt_file_binding(receipt: Mapping[str, Any]) -> str:
    primary = receipt.get("promotion_suite_file_sha256")
    legacy = receipt.get("promotion_suite_sha256")
    if primary is None:
        primary = legacy
    if primary is None:
        raise ValueError("calibration receipt lacks a promotion-suite file binding")
    result = validate_sha256(primary, field="promotion_suite_file_sha256")
    if (
        legacy is not None
        and validate_sha256(legacy, field="promotion_suite_sha256") != result
    ):
        raise ValueError("calibration receipt promotion-suite file hashes disagree")
    return result


def validate_calibration_binding(
    receipt: CalibrationReceiptArtifact | Mapping[str, Any] | str | Path,
    *,
    checkpoint_bytes: bytes,
    promotion_suite_canonical_sha256: str,
    promotion_suite_file_sha256: str | None = None,
    source_checkpoint_path: str | Path | None = None,
    calibration_fit_path: str | Path | None = None,
    promotion_suite_path: str | Path | None = None,
) -> CalibrationReceiptArtifact:
    """Bind a receipt to exact checkpoint, fit, and promotion evidence.

    Explicit path arguments may override receipt paths for a moved evidence
    bundle. The source checkpoint, calibration-fit data, and promotion suite
    must all be available, and their current bytes must match the receipt and
    calibrated-checkpoint metadata.
    """

    if not isinstance(checkpoint_bytes, bytes):
        raise TypeError("checkpoint_bytes must be bytes")
    artifact = load_calibration_receipt(receipt)
    checkpoint = _checkpoint_mapping(checkpoint_bytes)
    checkpoint_hash = hashlib.sha256(checkpoint_bytes).hexdigest()

    output_hash = _required_sha256(
        artifact.data.get("output_sha256"), field="output_sha256"
    )
    if output_hash != checkpoint_hash:
        raise ValueError(
            "calibration receipt output SHA-256 does not match the evaluated checkpoint"
        )

    receipt_parent = _required_sha256(
        artifact.data.get("source_sha256"), field="source_sha256"
    )
    checkpoint_parent = _required_sha256(
        checkpoint.get("calibration_parent_sha256"),
        field="checkpoint calibration_parent_sha256",
    )
    if receipt_parent != checkpoint_parent:
        raise ValueError(
            "calibration receipt source SHA-256 does not match checkpoint "
            "calibration_parent_sha256"
        )
    source_path = (
        Path(source_checkpoint_path).expanduser().resolve()
        if source_checkpoint_path is not None
        else _path_from_receipt(artifact, "source_path")
    )
    if source_path is None:
        raise ValueError("calibration receipt lacks a source checkpoint path")
    if not source_path.is_file():
        raise FileNotFoundError(f"artifact is not a file: {source_path}")
    source_bytes = source_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != receipt_parent:
        raise ValueError(
            "calibration source checkpoint bytes do not match source_sha256"
        )
    source_checkpoint = _checkpoint_mapping(source_bytes, role="source")
    _assert_calibration_only_change(source_checkpoint, checkpoint)

    receipt_fit = _required_sha256(
        artifact.data.get(
            "fit_data_sha256", artifact.data.get("calibration_suite_sha256")
        ),
        field="calibration data SHA-256",
    )
    checkpoint_fit = _required_sha256(
        checkpoint.get("calibration_suite_sha256"),
        field="checkpoint calibration_suite_sha256",
    )
    if receipt_fit != checkpoint_fit:
        raise ValueError(
            "calibration receipt fit SHA-256 does not match checkpoint "
            "calibration_suite_sha256"
        )
    fit_path = (
        Path(calibration_fit_path).expanduser().resolve()
        if calibration_fit_path is not None
        else _path_from_receipt(artifact, "calibration_suite")
    )
    if fit_path is None:
        raise ValueError("calibration receipt lacks a calibration-fit path")
    actual_fit = file_sha256(fit_path)
    if actual_fit != receipt_fit:
        raise ValueError(
            "calibration-fit file SHA-256 does not match receipt/checkpoint binding"
        )

    checkpoint_temperature = _temperature(
        checkpoint.get("temperature"), field="checkpoint temperature"
    )
    if not math.isclose(
        artifact.temperature,
        checkpoint_temperature,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "calibration receipt temperature does not match checkpoint temperature"
        )
    if checkpoint.get("score_status") != "CALIBRATED":
        raise ValueError("calibrated checkpoint must declare score_status=CALIBRATED")

    receipt_file_hash = _receipt_file_binding(artifact.data)
    receipt_canonical_hash = _required_sha256(
        artifact.data.get("promotion_suite_canonical_sha256"),
        field="promotion_suite_canonical_sha256",
    )
    checkpoint_file_hash = _required_sha256(
        checkpoint.get("promotion_suite_file_sha256"),
        field="checkpoint promotion_suite_file_sha256",
    )
    checkpoint_canonical_hash = _required_sha256(
        checkpoint.get("promotion_suite_canonical_sha256"),
        field="checkpoint promotion_suite_canonical_sha256",
    )
    if receipt_file_hash != checkpoint_file_hash:
        raise ValueError(
            "calibration receipt promotion-suite file SHA-256 does not match checkpoint"
        )
    if receipt_canonical_hash != checkpoint_canonical_hash:
        raise ValueError(
            "calibration receipt promotion-suite canonical SHA-256 does not match checkpoint"
        )

    actual_canonical_hash = validate_sha256(
        promotion_suite_canonical_sha256,
        field="evaluated promotion-suite canonical SHA-256",
    )
    if receipt_canonical_hash != actual_canonical_hash:
        raise ValueError(
            "calibration receipt promotion-suite canonical SHA-256 mismatch"
        )
    if promotion_suite_file_sha256 is not None:
        actual_file_hash = validate_sha256(
            promotion_suite_file_sha256,
            field="evaluated promotion-suite file SHA-256",
        )
        if receipt_file_hash != actual_file_hash:
            raise ValueError(
                "calibration receipt promotion-suite file SHA-256 mismatch"
            )

    promotion_path = (
        Path(promotion_suite_path).expanduser().resolve()
        if promotion_suite_path is not None
        else _path_from_receipt(artifact, "promotion_suite")
    )
    if promotion_path is None:
        raise ValueError("calibration receipt lacks a promotion-suite path")
    promotion_payload = promotion_path.read_bytes()
    try:
        promotion_value = json.loads(promotion_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("promotion suite must be valid UTF-8 JSON") from error
    if hashlib.sha256(promotion_payload).hexdigest() != receipt_file_hash:
        raise ValueError(
            "promotion-suite file bytes do not match receipt/checkpoint binding"
        )
    if canonical_json_sha256(promotion_value) != receipt_canonical_hash:
        raise ValueError(
            "promotion-suite canonical value does not match receipt/checkpoint binding"
        )

    if receipt_fit in {
        receipt_file_hash,
        receipt_canonical_hash,
        *(
            [promotion_suite_file_sha256]
            if promotion_suite_file_sha256 is not None
            else []
        ),
        promotion_suite_canonical_sha256,
    }:
        raise ValueError("calibration data must not be the promotion suite")
    return artifact


def _path_from_receipt(artifact: CalibrationReceiptArtifact, field: str) -> Path | None:
    value = artifact.data.get(field)
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value).expanduser()
    if not path.is_absolute() and artifact.path is not None:
        path = Path(artifact.path).parent / path
    return path.resolve()


def validate_runtime_calibration(
    checkpoint_path: str | Path,
    checkpoint_bytes: bytes,
    *,
    receipt: CalibrationReceiptArtifact | Mapping[str, Any] | str | Path | None = None,
    calibration_fit_path: str | Path | None = None,
    promotion_suite_path: str | Path | None = None,
) -> CalibrationReceiptArtifact | None:
    """Validate the exact sibling/provided receipt for a runtime checkpoint.

    Absence of a receipt returns ``None``. A present but invalid receipt raises;
    callers may either reject loading or explicitly downgrade the model to
    uncalibrated output.
    """

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    supplied_receipt = receipt is not None
    if receipt is None:
        sibling = checkpoint.parent / "calibration-receipt.json"
        if not sibling.is_file():
            return None
        receipt = sibling
    artifact = load_calibration_receipt(receipt)

    fit = (
        Path(calibration_fit_path).expanduser().resolve()
        if calibration_fit_path is not None
        else _path_from_receipt(artifact, "calibration_suite")
    )
    promotion = (
        Path(promotion_suite_path).expanduser().resolve()
        if promotion_suite_path is not None
        else _path_from_receipt(artifact, "promotion_suite")
    )
    if fit is None:
        qualifier = "provided" if supplied_receipt else "sibling"
        raise ValueError(f"{qualifier} calibration receipt lacks a fit-data path")
    if promotion is None:
        qualifier = "provided" if supplied_receipt else "sibling"
        raise ValueError(
            f"{qualifier} calibration receipt lacks a promotion-suite path"
        )

    promotion_payload = promotion.read_bytes()
    try:
        promotion_value = json.loads(promotion_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("promotion suite must be valid UTF-8 JSON") from error
    return validate_calibration_binding(
        artifact,
        checkpoint_bytes=checkpoint_bytes,
        promotion_suite_file_sha256=hashlib.sha256(promotion_payload).hexdigest(),
        promotion_suite_canonical_sha256=canonical_json_sha256(promotion_value),
        source_checkpoint_path=_path_from_receipt(artifact, "source_path"),
        calibration_fit_path=fit,
        promotion_suite_path=promotion,
    )

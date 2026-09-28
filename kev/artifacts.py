"""Content-addressing helpers for KEV research artifacts.

The functions in this module deliberately distinguish a file's byte hash from
the hash of the JSON value it contains.  A file hash changes when formatting
changes; a canonical JSON hash does not.  Evidence records should normally
carry both when the source is a JSON file.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


SHA256_HEX_LENGTH = 64


def _validate_json_value(value: Any, location: str = "$") -> None:
    """Reject values whose JSON representation would be ambiguous or invalid."""

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {location}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{location}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"JSON object key at {location} is not a string")
            _validate_json_value(item, f"{location}.{key}")
        return
    raise TypeError(
        f"value at {location} is not JSON-compatible: {type(value).__name__}"
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Return KEV's deterministic UTF-8 JSON representation of ``value``.

    This is an internal canonicalization contract, not a claim of full RFC 8785
    compatibility.  It is stable for JSON-native Python values and rejects
    NaN, infinity, non-string object keys, tuples, and custom objects.
    """

    _validate_json_value(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_json_sha256(value: Any) -> str:
    """Return the SHA-256 digest of :func:`canonical_json_bytes`."""

    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_sha256(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Hash a file as stored on disk without loading it all into memory."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"artifact is not a file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def validate_sha256(value: str, *, field: str = "sha256") -> str:
    """Validate and normalize a hexadecimal SHA-256 digest."""

    normalized = str(value).strip().casefold()
    if len(normalized) != SHA256_HEX_LENGTH:
        raise ValueError(f"{field} must contain 64 hexadecimal characters")
    try:
        int(normalized, 16)
    except ValueError as exc:
        raise ValueError(f"{field} must contain 64 hexadecimal characters") from exc
    return normalized


def verify_file_sha256(path: str | Path, expected_sha256: str) -> str:
    """Verify a file hash and return the normalized digest on success."""

    expected = validate_sha256(expected_sha256, field="expected_sha256")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"artifact SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def json_artifact_hashes(path: str | Path) -> dict[str, Any]:
    """Return byte-level and canonical hashes for a JSON artifact."""

    source = Path(path)
    raw = source.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid UTF-8 JSON artifact: {source}") from exc
    return {
        "path": str(source),
        "size_bytes": len(raw),
        "file_sha256": hashlib.sha256(raw).hexdigest(),
        "canonical_json_sha256": canonical_json_sha256(value),
    }


def file_artifact(path: str | Path, *, role: str | None = None) -> dict[str, Any]:
    """Describe an arbitrary immutable artifact by path, size, and SHA-256."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"artifact is not a file: {source}")
    record: dict[str, Any] = {
        "schema": "kev.artifact-ref.v1",
        "path": str(source),
        "size_bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }
    if role is not None:
        record["role"] = str(role)
    return record

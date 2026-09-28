"""Explicitly acquire or offline-verify KEV's selected frozen encoder.

This is the only network-bearing encoder path. Runtime loading never calls the
Hub: it consumes a local manifest and hash-verifies the exact bytes first.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.request
from typing import Any, Mapping

from kev.encoders import verify_local_encoder_manifest


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    ROOT
    / "models"
    / "encoder-sources"
    / "all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json"
)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_object(path: Path) -> tuple[dict[str, Any], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value, payload


def _contained(root: Path, relative: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts or "://" in relative:
        raise ValueError(f"local relative path required: {relative!r}")
    result = (root / raw).resolve()
    try:
        result.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"path escapes acquisition root: {relative!r}") from error
    return result


def _verify_payload(payload: bytes, record: Mapping[str, Any]) -> None:
    actual = _sha256(payload)
    expected = str(record.get("sha256", ""))
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {record.get('path')}: expected {expected}, got {actual}"
        )
    expected_size = record.get("size_bytes")
    if expected_size != len(payload):
        raise ValueError(
            f"size mismatch for {record.get('path')}: expected {expected_size}, "
            f"got {len(payload)}"
        )


def _write_exclusive(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _source(source_path: Path) -> tuple[dict[str, Any], bytes, Path]:
    source, payload = _read_object(source_path)
    if source.get("schema") != "kev.encoder-acquisition.v1":
        raise ValueError("invalid encoder acquisition schema")
    if source.get("network_policy") != "ACQUISITION_ONLY_RUNTIME_LOCAL_ONLY":
        raise ValueError("acquisition must forbid runtime network access")
    files = source.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("acquisition manifest requires files")
    paths = [record.get("path") for record in files if isinstance(record, Mapping)]
    if len(paths) != len(files) or len(set(paths)) != len(paths):
        raise ValueError("acquisition file paths must be unique strings")
    target = _contained(ROOT, str(source.get("local_directory", "")))
    return source, payload, target


def _download_huggingface(source: Mapping[str, Any], relative: str) -> bytes:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError(
            "acquisition requires huggingface-hub; install the semantic-encoder extra"
        ) from error
    identity = source["source"]
    cached = hf_hub_download(
        repo_id=str(identity["repository"]),
        revision=str(identity["revision"]),
        filename=relative,
    )
    return Path(cached).read_bytes()


def _download_license(source: Mapping[str, Any]) -> bytes:
    url = str(source["license"]["license_text_url"])
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310
        return response.read()


def _verify_receipts(
    source: Mapping[str, Any], source_payload: bytes, target: Path
) -> dict[str, Any]:
    manifest_path = target / str(source["local_manifest_name"])
    receipt_path = target / str(source["acquisition_receipt_name"])
    verified = verify_local_encoder_manifest(manifest_path)
    expected_manifest_sha256 = str(source.get("expected_local_manifest_sha256", ""))
    if verified["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError(
            "local encoder manifest differs from the source-pinned expected hash: "
            f"expected {expected_manifest_sha256}, got {verified['manifest_sha256']}"
        )
    receipt, receipt_payload = _read_object(receipt_path)
    if receipt.get("schema") != "kev.encoder-acquisition-receipt.v1":
        raise ValueError("invalid encoder acquisition receipt schema")
    if receipt.get("source_manifest_sha256") != _sha256(source_payload):
        raise ValueError("acquisition receipt source-manifest hash mismatch")
    if receipt.get("encoder_manifest_sha256") != verified["manifest_sha256"]:
        raise ValueError("acquisition receipt encoder-manifest hash mismatch")
    if receipt.get("upstream_revision") != source["source"]["revision"]:
        raise ValueError("acquisition receipt upstream revision mismatch")
    expected = {
        record["path"]: (record["sha256"], record["size_bytes"])
        for record in source["files"]
    }
    observed = {
        record["path"]: (record["sha256"], record["size_bytes"])
        for record in verified["files"]
    }
    if observed != expected:
        raise ValueError("local encoder manifest differs from acquisition source")
    return {
        "schema": "kev.encoder-offline-verification.v1",
        "ok": True,
        "upstream_revision": source["source"]["revision"],
        "encoder_manifest_path": str(manifest_path),
        "encoder_manifest_sha256": verified["manifest_sha256"],
        "acquisition_receipt_path": str(receipt_path),
        "acquisition_receipt_sha256": _sha256(receipt_payload),
        "files_verified": len(observed),
        "network_access": "NONE",
    }


def acquire(args: argparse.Namespace) -> dict[str, Any]:
    source_path = Path(args.source_manifest).expanduser().resolve()
    source, source_payload, target = _source(source_path)
    if args.verify_only:
        return _verify_receipts(source, source_payload, target)
    if args.accept_license != source["license"]["identifier"]:
        raise ValueError(
            f"pass --accept-license {source['license']['identifier']} after review"
        )
    if not args.reviewed_by.strip() or not args.reviewed_at.strip():
        raise ValueError("--reviewed-by and --reviewed-at are required")
    target.mkdir(parents=True, exist_ok=True)

    downloaded: list[str] = []
    reused: list[str] = []
    for record in source["files"]:
        relative = str(record["path"])
        destination = _contained(target, relative)
        if destination.exists():
            payload = destination.read_bytes()
            _verify_payload(payload, record)
            reused.append(relative)
            continue
        if record["origin"] == "huggingface":
            payload = _download_huggingface(source, relative)
        elif record["origin"] == "license_url":
            payload = _download_license(source)
        else:
            raise ValueError(f"unsupported acquisition origin: {record['origin']}")
        _verify_payload(payload, record)
        _write_exclusive(destination, payload)
        downloaded.append(relative)

    local_manifest = {
        "schema": "kev.local-encoder.v1",
        "name": source["name"],
        "embedding_dim": source["embedding_dim"],
        "network_policy": "LOCAL_ONLY",
        "source": source["source"],
        "permission": {
            "status": "APPROVED",
            "reviewed_by": args.reviewed_by.strip(),
            "reviewed_at": args.reviewed_at.strip(),
            "scope": args.permission_scope.strip(),
        },
        "license": source["license"],
        "loader": source["loader"],
        "files": [
            {
                "path": record["path"],
                "sha256": record["sha256"],
            }
            for record in source["files"]
        ],
    }
    manifest_path = target / str(source["local_manifest_name"])
    manifest_payload = _json_bytes(local_manifest)
    _write_exclusive(manifest_path, manifest_payload)
    verified = verify_local_encoder_manifest(manifest_path)
    expected_manifest_sha256 = str(source.get("expected_local_manifest_sha256", ""))
    if verified["manifest_sha256"] != expected_manifest_sha256:
        raise ValueError(
            "generated encoder manifest differs from the source-pinned expected hash: "
            f"expected {expected_manifest_sha256}, got {verified['manifest_sha256']}"
        )

    receipt = {
        "schema": "kev.encoder-acquisition-receipt.v1",
        "name": source["name"],
        "source_manifest_path": str(source_path),
        "source_manifest_sha256": _sha256(source_payload),
        "upstream_repository": source["source"]["repository"],
        "upstream_revision": source["source"]["revision"],
        "upstream_url": source["source"]["url"],
        "license": source["license"],
        "permission": local_manifest["permission"],
        "encoder_manifest_path": str(manifest_path),
        "encoder_manifest_sha256": verified["manifest_sha256"],
        "files": verified["files"],
        "downloaded_files": downloaded,
        "reused_verified_files": reused,
        "network_access": "ACQUISITION_ONLY",
        "runtime_network_policy": "LOCAL_ONLY",
        "dynamic_remote_code": False,
        "safetensors_only": True,
        "offline_replay": {
            "copy": "copy this entire encoder directory without changing bytes",
            "verify_command": (
                "python scripts/acquire_frozen_encoder.py --verify-only "
                f"--source-manifest {source_path}"
            ),
        },
        "environment": {
            "python": sys.version.split()[0],
        },
    }
    try:
        import huggingface_hub

        receipt["environment"]["huggingface_hub"] = huggingface_hub.__version__
    except (ImportError, AttributeError):
        receipt["environment"]["huggingface_hub"] = "UNAVAILABLE"
    receipt_path = target / str(source["acquisition_receipt_name"])
    receipt_payload = _json_bytes(receipt)
    _write_exclusive(receipt_path, receipt_payload)
    return {
        "schema": "kev.encoder-acquisition-result.v1",
        "ok": True,
        "encoder_manifest_path": str(manifest_path),
        "encoder_manifest_sha256": verified["manifest_sha256"],
        "acquisition_receipt_path": str(receipt_path),
        "acquisition_receipt_sha256": _sha256(receipt_payload),
        "downloaded_files": downloaded,
        "reused_verified_files": reused,
        "runtime_network_policy": "LOCAL_ONLY",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", default=str(DEFAULT_SOURCE))
    parser.add_argument("--reviewed-by", default="")
    parser.add_argument("--reviewed-at", default="")
    parser.add_argument(
        "--permission-scope",
        default=(
            "local KEV frozen feature extraction and proposal-head research under "
            "Apache-2.0; no redistribution claim beyond the license"
        ),
    )
    parser.add_argument("--accept-license", default="")
    parser.add_argument("--verify-only", action="store_true")
    return parser


def main() -> int:
    result = acquire(build_parser().parse_args())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

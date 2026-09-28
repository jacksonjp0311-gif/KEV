from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from kev.evaluation import load_frozen_suite
from kev.model_runtime import load_proposal_model_bytes, proposal_logits
from kev.uc51a2.semantic_breadth import INTENTS, sha256_file


def _rows_from_bytes(payload: bytes, *, source: str | Path) -> list[dict[str, Any]]:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"calibration suite is not valid UTF-8: {source}") from error
    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError("calibration suite is empty")
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers) or len(
        set(identifiers)
    ) != len(identifiers):
        raise ValueError("calibration items require unique ids")
    for row in rows:
        if row.get("split") != "calibration_fit":
            raise ValueError(
                "temperature fitting requires a calibration_fit-only suite"
            )
    return rows


def _rows(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    return _rows_from_bytes(source.read_bytes(), source=source)


def _expected_kinds(row: dict[str, Any]) -> list[str]:
    kinds = row.get("frame_kinds")
    if kinds is None:
        kinds = [frame["kind"] for frame in row.get("expected_frames", [])]
    if not kinds and row.get("expected_intent"):
        kinds = [row["expected_intent"]]
    if not kinds or any(kind not in INTENTS for kind in kinds):
        raise ValueError(f"invalid calibration target for {row.get('id')}")
    return list(dict.fromkeys(kinds))


def _promotion_rows(payload: bytes, *, source: str | Path) -> list[dict[str, Any]]:
    try:
        raw = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"promotion suite is not valid UTF-8: {source}") from error
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        promotion_rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        if isinstance(parsed, dict) and isinstance(parsed.get("splits"), dict):
            promotion_rows = [
                row
                for rows in parsed["splits"].values()
                for row in rows
                if isinstance(row, dict)
            ]
        elif isinstance(parsed, list):
            promotion_rows = [row for row in parsed if isinstance(row, dict)]
        else:
            raise ValueError(
                "promotion suite must be a frozen suite object or item list"
            )
    return promotion_rows


def _assert_disjoint_rows(
    calibration_rows: list[dict[str, Any]], promotion_rows: list[dict[str, Any]]
) -> None:

    calibration_ids = {str(row["id"]) for row in calibration_rows}
    promotion_ids = {str(row.get("id", "")) for row in promotion_rows}
    id_overlap = sorted(calibration_ids & promotion_ids)
    calibration_text = {
        str(row.get("text", row.get("input", ""))).strip().casefold()
        for row in calibration_rows
    } - {""}
    promotion_text = {
        str(row.get("input", row.get("text", ""))).strip().casefold()
        for row in promotion_rows
    } - {""}
    text_overlap = sorted(calibration_text & promotion_text)
    if id_overlap or text_overlap:
        details = []
        if id_overlap:
            details.append("ids=" + ",".join(id_overlap))
        if text_overlap:
            details.append("surface_forms=" + ",".join(text_overlap))
        raise ValueError(
            "calibration and promotion suites overlap: " + "; ".join(details)
        )


def assert_disjoint_suites(
    calibration_path: str | Path, promotion_path: str | Path
) -> None:
    """Reject identifier *or surface-form* leakage across calibration/promotion.

    Calibration input is JSONL while the frozen promotion pack is a JSON
    object. Supporting both forms explicitly also keeps this check independent
    of filename extensions.
    """

    calibration_rows = _rows(calibration_path)
    promotion_file = Path(promotion_path)
    promotion_rows = _promotion_rows(promotion_file.read_bytes(), source=promotion_file)
    _assert_disjoint_rows(calibration_rows, promotion_rows)


def calibrate_checkpoint(
    checkpoint_path: str | Path,
    calibration_suite: str | Path,
    output_dir: str | Path,
    promotion_suite: str | Path | None = None,
    steps: int = 200,
) -> dict[str, Any]:
    """Fit one temperature without changing model weights.

    A distinct checkpoint is written. The uncalibrated challenger remains on
    disk and is never overwritten.
    """

    if promotion_suite is None:
        raise ValueError(
            "promotion_suite is required to prove calibration-fit disjointness"
        )
    if steps <= 0:
        raise ValueError("calibration steps must be positive")
    calibration_path = Path(calibration_suite).resolve()
    calibration_payload = calibration_path.read_bytes()
    calibration_suite_hash = hashlib.sha256(calibration_payload).hexdigest()
    rows = _rows_from_bytes(calibration_payload, source=calibration_path)
    promotion_path = Path(promotion_suite).resolve()
    promotion = load_frozen_suite(promotion_path)
    promotion_rows = [
        row for split_rows in promotion.data["splits"].values() for row in split_rows
    ]
    _assert_disjoint_rows(rows, promotion_rows)
    source = Path(checkpoint_path).resolve()
    source_bytes = source.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    source_checkpoint = torch.load(
        io.BytesIO(source_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(source_checkpoint, dict):
        raise ValueError("source checkpoint must contain a mapping")
    if (
        not isinstance(source_checkpoint.get("schema"), str)
        or not source_checkpoint["schema"].strip()
    ):
        raise ValueError("source checkpoint requires a non-empty schema")
    if (
        source_checkpoint.get("temperature") is not None
        or source_checkpoint.get("score_status") == "CALIBRATED"
    ):
        raise ValueError(
            "temperature fitting requires an uncalibrated source checkpoint"
        )
    model = load_proposal_model_bytes(source_bytes)
    model.eval()
    targets = torch.zeros((len(rows), len(INTENTS)), dtype=torch.float32)
    for row_index, row in enumerate(rows):
        for kind in _expected_kinds(row):
            targets[row_index, INTENTS.index(kind)] = 1.0
    logits = proposal_logits(model, [row["text"] for row in rows])

    log_temperature = torch.tensor(0.0, requires_grad=True)
    optimizer = torch.optim.Adam([log_temperature], lr=0.05)
    history: list[float] = []
    for _ in range(steps):
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        history.append(float(loss.detach()))
    fitted_temperature = float(log_temperature.detach().exp().clamp(0.05, 20.0))
    if not math.isfinite(fitted_temperature):
        raise RuntimeError("temperature fitting produced a non-finite value")

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / "semantic-breadth.calibrated.pt"
    receipt_path = destination / "calibration-receipt.json"
    if output.exists() or receipt_path.exists():
        raise FileExistsError("calibrated artifact is immutable and already exists")
    if sha256_file(source) != source_hash:
        raise RuntimeError("source checkpoint changed during calibration")
    if (
        hashlib.sha256(calibration_path.read_bytes()).hexdigest()
        != calibration_suite_hash
    ):
        raise RuntimeError("calibration suite changed during temperature fitting")
    if sha256_file(promotion_path) != promotion.file_sha256:
        raise RuntimeError("promotion suite changed during temperature fitting")
    calibrated = dict(source_checkpoint)
    calibrated.update(
        {
            "temperature": fitted_temperature,
            "score_status": "CALIBRATED",
            "calibration_suite_sha256": calibration_suite_hash,
            "calibration_parent_sha256": source_hash,
            "promotion_suite_file_sha256": promotion.file_sha256,
            "promotion_suite_canonical_sha256": promotion.canonical_sha256,
        }
    )
    with output.open("xb") as handle:
        torch.save(calibrated, handle)
    receipt = {
        "schema": "kev.calibration-receipt.v1",
        "source_path": str(source),
        "source_sha256": source_hash,
        "calibration_suite": str(calibration_path),
        "calibration_suite_sha256": calibration_suite_hash,
        "fit_data_sha256": calibration_suite_hash,
        "calibration_fit_split": "calibration_fit",
        "promotion_suite": str(promotion_path),
        "promotion_suite_sha256": promotion.file_sha256,
        "promotion_suite_file_sha256": promotion.file_sha256,
        "promotion_suite_canonical_sha256": promotion.canonical_sha256,
        "temperature": fitted_temperature,
        "loss_history": history,
        "output_path": str(output),
        "output_sha256": sha256_file(output),
        "score_status": "CALIBRATED",
        "activation": "NONE",
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt

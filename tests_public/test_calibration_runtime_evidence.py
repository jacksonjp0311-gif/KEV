from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.calibration import calibrate_checkpoint
from kev.calibration_evidence import validate_calibration_binding
from kev.evolution import DEFAULT_CALIBRATION, DEFAULT_REGISTRY, DEFAULT_SUITE
from kev.model_runtime import CheckpointPredictor
from kev.uc51a3.alive import AliveRuntime, AliveStore


ROOT = Path(__file__).resolve().parents[1]


def _parent() -> Path:
    registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    return ROOT / registry["active"]["path"]


def _binding_fixture(tmp_path: Path) -> tuple[bytes, Path, dict[str, object], str, str]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    fit = tmp_path / "fit.jsonl"
    fit.write_text('{"id":"fit-1","split":"calibration_fit"}\n', encoding="utf-8")
    promotion = tmp_path / "promotion.json"
    promotion_value = {"schema": "kev.eval-suite.v1", "id": "promotion"}
    promotion.write_text(json.dumps(promotion_value), encoding="utf-8")
    parent = tmp_path / "parent.pt"
    parent_checkpoint = {
        "schema": "kev.test-checkpoint.v1",
        "state_dict": {"projection.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])},
        "lineage": {"parent_sha256": "0" * 64, "generation": 1},
        "objective": {"name": "test-objective", "weights": [1.0, 0.35]},
        "temperature": None,
        "score_status": "UNCALIBRATED",
    }
    torch.save(parent_checkpoint, parent)
    parent_hash = file_sha256(parent)
    fit_hash = file_sha256(fit)
    promotion_file_hash = file_sha256(promotion)
    promotion_canonical_hash = canonical_json_sha256(promotion_value)
    checkpoint_path = tmp_path / "calibrated.pt"
    calibrated_checkpoint = dict(parent_checkpoint)
    calibrated_checkpoint.update(
        {
            "temperature": 1.75,
            "score_status": "CALIBRATED",
            "calibration_parent_sha256": parent_hash,
            "calibration_suite_sha256": fit_hash,
            "promotion_suite_file_sha256": promotion_file_hash,
            "promotion_suite_canonical_sha256": promotion_canonical_hash,
        }
    )
    torch.save(calibrated_checkpoint, checkpoint_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    receipt: dict[str, object] = {
        "schema": "kev.calibration-receipt.v1",
        "source_sha256": parent_hash,
        "source_path": str(parent),
        "calibration_suite": str(fit),
        "calibration_suite_sha256": fit_hash,
        "fit_data_sha256": fit_hash,
        "calibration_fit_split": "calibration_fit",
        "promotion_suite": str(promotion),
        "promotion_suite_sha256": promotion_file_hash,
        "promotion_suite_file_sha256": promotion_file_hash,
        "promotion_suite_canonical_sha256": promotion_canonical_hash,
        "temperature": 1.75,
        "output_path": str(checkpoint_path),
        "output_sha256": file_sha256(checkpoint_path),
        "score_status": "CALIBRATED",
    }
    return checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash


def _validate_fixture(
    checkpoint_bytes: bytes,
    fit: Path,
    receipt: dict[str, object],
    promotion_file_hash: str,
    promotion_canonical_hash: str,
):
    return validate_calibration_binding(
        receipt,
        checkpoint_bytes=checkpoint_bytes,
        calibration_fit_path=fit,
        promotion_suite_file_sha256=promotion_file_hash,
        promotion_suite_canonical_sha256=promotion_canonical_hash,
    )


def test_complete_calibration_binding_is_valid(tmp_path: Path):
    fixture = _binding_fixture(tmp_path)
    validated = _validate_fixture(*fixture)
    assert validated.temperature == 1.75


def test_calibration_binding_rejects_tensor_mutation(tmp_path: Path):
    _checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    checkpoint_path = Path(str(receipt["output_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["state_dict"]["projection.weight"][0, 0] = 99.0
    torch.save(checkpoint, checkpoint_path)
    receipt["output_sha256"] = file_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="checkpoint tensor differs"):
        _validate_fixture(
            checkpoint_path.read_bytes(),
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("lineage", {"parent_sha256": "f" * 64, "generation": 1}),
        ("objective", {"name": "mutated-objective", "weights": [1.0, 0.35]}),
        ("unauthorized_metadata", {"claim": "calibration-only"}),
    ],
)
def test_calibration_binding_rejects_noncalibration_metadata_mutation(
    tmp_path: Path, field: str, value: object
):
    _checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    checkpoint_path = Path(str(receipt["output_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint[field] = value
    torch.save(checkpoint, checkpoint_path)
    receipt["output_sha256"] = file_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="non-calibration checkpoint"):
        _validate_fixture(
            checkpoint_path.read_bytes(),
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_calibration_binding_rejects_incompatible_checkpoint_schema(tmp_path: Path):
    _checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    checkpoint_path = Path(str(receipt["output_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["schema"] = "kev.incompatible-checkpoint.v1"
    torch.save(checkpoint, checkpoint_path)
    receipt["output_sha256"] = file_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="schemas are incompatible"):
        _validate_fixture(
            checkpoint_path.read_bytes(),
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_calibration_binding_rejects_invalid_torch_source(tmp_path: Path):
    _checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    source_path = Path(str(receipt["source_path"]))
    source_path.write_bytes(b"not a Torch checkpoint")
    source_hash = file_sha256(source_path)
    receipt["source_sha256"] = source_hash

    checkpoint_path = Path(str(receipt["output_path"]))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    checkpoint["calibration_parent_sha256"] = source_hash
    torch.save(checkpoint, checkpoint_path)
    receipt["output_sha256"] = file_sha256(checkpoint_path)

    with pytest.raises(ValueError, match="source checkpoint is not a readable Torch"):
        _validate_fixture(
            checkpoint_path.read_bytes(),
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_relative_source_path_is_resolved_from_receipt(tmp_path: Path):
    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    receipt["source_path"] = "parent.pt"
    receipt_path = tmp_path / "calibration-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    validated = validate_calibration_binding(
        receipt_path,
        checkpoint_bytes=checkpoint_bytes,
        calibration_fit_path=fit,
        promotion_suite_file_sha256=promotion_file_hash,
        promotion_suite_canonical_sha256=promotion_canonical_hash,
    )
    assert validated.temperature == 1.75


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("output_sha256", "2" * 64, "output SHA-256"),
        ("source_sha256", "2" * 64, "calibration_parent_sha256"),
        ("temperature", 2.25, "checkpoint temperature"),
        ("promotion_suite_canonical_sha256", "2" * 64, "canonical"),
    ],
)
def test_mutated_calibration_receipt_binding_rejects(
    tmp_path: Path, field: str, value: object, message: str
):
    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    receipt[field] = value
    with pytest.raises(ValueError, match=message):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_mutated_fit_binding_and_actual_fit_bytes_reject(tmp_path: Path):
    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    receipt["fit_data_sha256"] = "2" * 64
    receipt["calibration_suite_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="checkpoint calibration_suite_sha256"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )

    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path / "actual-bytes")
    )
    fit.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="calibration-fit file SHA-256"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_missing_or_tampered_source_fit_and_promotion_artifacts_reject(
    tmp_path: Path,
):
    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path / "missing-source")
    )
    receipt.pop("source_path")
    with pytest.raises(ValueError, match="lacks a source checkpoint path"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )

    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path / "tampered-source")
    )
    Path(str(receipt["source_path"])).write_bytes(b"tampered parent")
    with pytest.raises(ValueError, match="source checkpoint bytes"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )

    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path / "missing-fit")
    )
    fit.unlink()
    with pytest.raises(FileNotFoundError, match="artifact is not a file"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )

    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path / "tampered-promotion")
    )
    Path(str(receipt["promotion_suite"])).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="promotion-suite file bytes"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_missing_output_artifact_cannot_be_a_runtime_model(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CheckpointPredictor(tmp_path / "missing-calibrated-output.pt")


def test_mutated_promotion_file_binding_rejects(tmp_path: Path):
    checkpoint_bytes, fit, receipt, promotion_file_hash, promotion_canonical_hash = (
        _binding_fixture(tmp_path)
    )
    receipt["promotion_suite_sha256"] = "2" * 64
    receipt["promotion_suite_file_sha256"] = "2" * 64
    with pytest.raises(ValueError, match="file SHA-256 does not match checkpoint"):
        _validate_fixture(
            checkpoint_bytes,
            fit,
            receipt,
            promotion_file_hash,
            promotion_canonical_hash,
        )


def test_runtime_requires_exact_receipt_and_downgrades_invalid_sibling(
    tmp_path: Path,
):
    calibrated_dir = tmp_path / "calibrated"
    receipt = calibrate_checkpoint(
        _parent(),
        DEFAULT_CALIBRATION,
        calibrated_dir,
        promotion_suite=DEFAULT_SUITE,
        steps=1,
    )
    checkpoint = Path(receipt["output_path"])
    valid = CheckpointPredictor(checkpoint)
    valid_result = valid(
        {"id": "probe", "input": "reduce latency", "projection": "kinds"}
    )
    assert valid_result["score_status"] == "CALIBRATED"
    assert valid_result["temperature"] == pytest.approx(receipt["temperature"])

    copied_dir = tmp_path / "copied-without-receipt"
    copied_dir.mkdir()
    copied = copied_dir / checkpoint.name
    shutil.copy2(checkpoint, copied)
    copied_predictor = CheckpointPredictor(copied)
    copied_result = copied_predictor(
        {"id": "probe", "input": "reduce latency", "projection": "kinds"}
    )
    assert copied_result["score_status"] == "UNCALIBRATED"
    assert copied_result["temperature"] is None
    assert copied_predictor.calibration_error == "CALIBRATION_RECEIPT_MISSING"

    forged = dict(receipt)
    forged["source_sha256"] = "2" * 64
    receipt_path = calibrated_dir / "calibration-receipt.json"
    receipt_path.write_text(
        json.dumps(forged, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    downgraded = CheckpointPredictor(checkpoint)
    downgraded_result = downgraded(
        {"id": "probe", "input": "reduce latency", "projection": "kinds"}
    )
    assert downgraded_result["score_status"] == "UNCALIBRATED"
    assert downgraded_result["temperature"] is None
    assert downgraded.calibration_error is not None
    assert "calibration_parent_sha256" in downgraded.calibration_error
    with pytest.raises(ValueError, match="invalid calibration evidence"):
        CheckpointPredictor(checkpoint, calibration_receipt=receipt_path)


def test_alive_runtime_does_not_trust_copied_embedded_temperature(tmp_path: Path):
    receipt = calibrate_checkpoint(
        _parent(),
        DEFAULT_CALIBRATION,
        tmp_path / "calibrated",
        promotion_suite=DEFAULT_SUITE,
        steps=1,
    )
    source = Path(receipt["output_path"])
    copied_dir = tmp_path / "runtime-copy"
    copied_dir.mkdir()
    copied = copied_dir / source.name
    shutil.copy2(source, copied)

    state_dir = tmp_path / "state"
    store = AliveStore(state_dir)
    incumbent_state = store.read()
    store.record_model_decision(
        "MODEL_QUALIFIED",
        {"run_id": "copied-calibration-evidence-test"},
        active_model={"path": str(copied), "sha256": file_sha256(copied)},
        expected_incumbent={
            "path": incumbent_state["semantic_model"],
            "sha256": incumbent_state["semantic_model_sha256"],
            "generation": incumbent_state["semantic_model_generation"],
        },
    )
    runtime = AliveRuntime(state_dir)
    proposal = runtime._proposal("reduce latency")
    assert proposal["score_status"] == "UNCALIBRATED"
    assert proposal["temperature"] is None
    assert proposal["calibration_error"] == "CALIBRATION_RECEIPT_MISSING"

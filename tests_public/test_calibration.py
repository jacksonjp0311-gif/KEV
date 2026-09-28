from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import kev.calibration as calibration_module
from kev.artifacts import file_sha256
from kev.calibration import assert_disjoint_suites, calibrate_checkpoint
from kev.evaluation import load_frozen_suite
from kev.evolution import DEFAULT_CALIBRATION, DEFAULT_REGISTRY, DEFAULT_SUITE


ROOT = Path(__file__).resolve().parents[1]


def _parent() -> Path:
    registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    return ROOT / registry["active"]["path"]


def test_published_calibration_slice_is_disjoint_from_promotion_suite():
    assert_disjoint_suites(DEFAULT_CALIBRATION, DEFAULT_SUITE)


def test_calibration_rejects_surface_form_overlap_even_with_distinct_ids(
    tmp_path: Path,
):
    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        json.dumps(
            {
                "id": "cal-1",
                "split": "calibration_fit",
                "text": "same surface",
                "frame_kinds": ["GOAL"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema": "kev.eval-suite.v1",
                "id": "test",
                "frozen": True,
                "splits": {
                    "fresh": [{"id": "fresh-1", "input": "same surface"}],
                    "retention": [],
                    "oov": [],
                },
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="surface_forms"):
        assert_disjoint_suites(calibration, promotion)


def test_calibration_requires_a_promotion_suite_for_disjointness(tmp_path: Path):
    with pytest.raises(ValueError, match="promotion_suite is required"):
        calibrate_checkpoint(
            _parent(), DEFAULT_CALIBRATION, tmp_path / "output", steps=1
        )


def test_calibration_receipt_binds_output_fit_data_and_promotion_suite(tmp_path: Path):
    receipt = calibrate_checkpoint(
        _parent(),
        DEFAULT_CALIBRATION,
        tmp_path / "output",
        promotion_suite=DEFAULT_SUITE,
        steps=1,
    )
    suite = load_frozen_suite(DEFAULT_SUITE)

    assert receipt["source_sha256"] == file_sha256(_parent())
    assert Path(receipt["source_path"]) == _parent().resolve()
    assert receipt["calibration_suite_sha256"] == file_sha256(DEFAULT_CALIBRATION)
    assert receipt["fit_data_sha256"] == file_sha256(DEFAULT_CALIBRATION)
    assert receipt["calibration_fit_split"] == "calibration_fit"
    assert Path(receipt["promotion_suite"]) == DEFAULT_SUITE.resolve()
    assert receipt["promotion_suite_file_sha256"] == suite.file_sha256
    assert receipt["promotion_suite_canonical_sha256"] == suite.canonical_sha256
    assert receipt["output_sha256"] == file_sha256(receipt["output_path"])
    checkpoint = torch.load(
        receipt["output_path"], map_location="cpu", weights_only=True
    )
    assert checkpoint["temperature"] == receipt["temperature"]
    assert checkpoint["score_status"] == "CALIBRATED"
    assert checkpoint["promotion_suite_file_sha256"] == suite.file_sha256
    assert checkpoint["promotion_suite_canonical_sha256"] == suite.canonical_sha256

    with pytest.raises(ValueError, match="uncalibrated source checkpoint"):
        calibrate_checkpoint(
            receipt["output_path"],
            DEFAULT_CALIBRATION,
            tmp_path / "recalibrated",
            promotion_suite=DEFAULT_SUITE,
            steps=1,
        )


def test_calibration_rejects_fit_file_changed_after_captured_parse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    calibration = tmp_path / "calibration-fit.jsonl"
    calibration.write_bytes(DEFAULT_CALIBRATION.read_bytes())
    original_proposal_logits = calibration_module.proposal_logits

    def mutate_fit_after_model_input_is_built(model, texts):
        result = original_proposal_logits(model, texts)
        calibration.write_bytes(calibration.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(
        calibration_module, "proposal_logits", mutate_fit_after_model_input_is_built
    )
    with pytest.raises(RuntimeError, match="calibration suite changed"):
        calibrate_checkpoint(
            _parent(),
            calibration,
            tmp_path / "changed-fit-output",
            promotion_suite=DEFAULT_SUITE,
            steps=1,
        )

    assert not (
        tmp_path / "changed-fit-output" / "semantic-breadth.calibrated.pt"
    ).exists()

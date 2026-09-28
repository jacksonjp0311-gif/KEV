from __future__ import annotations

import hashlib
import json

import pytest
import torch

from kev.artifacts import canonical_json_sha256, file_sha256, json_artifact_hashes
from kev.evaluation import (
    compare_evaluations,
    evaluate_challenger,
    evaluate_suite,
    load_frozen_suite,
    load_temperature_artifact,
    promotion_decision,
    replay_historical_aggregate,
)


def _suite() -> dict:
    return {
        "schema": "kev.eval-suite.v1",
        "id": "public-eval-test-v1",
        "frozen": True,
        "provenance": {"purpose": "unit-test fixture", "training_use": "FORBIDDEN"},
        "splits": {
            "fresh": [
                {"id": "fresh-1", "input": "alpha", "expected": "GOAL"},
                {"id": "fresh-2", "input": "beta", "expected": "CONSTRAINT"},
            ],
            "retention": [
                {"id": "retention-1", "input": "gamma", "expected": "COPY_VALUE"}
            ],
            "oov": [
                {"id": "oov-1", "input": "unseen wording", "expected": "PREDICTION"}
            ],
            "composition": [
                {
                    "id": "composition-1",
                    "input": "one goal and one constraint",
                    "expected_frames": ["GOAL", "CONSTRAINT"],
                },
                {
                    "id": "composition-2",
                    "input": "one observation and one prediction",
                    "expected_frames": ["OBSERVATION", "PREDICTION"],
                },
            ],
            "calibration": [
                {"id": "calibration-1", "input": "calibration only", "expected": "GOAL"}
            ],
        },
    }


def _incumbent_predictions() -> dict:
    return {
        "fresh-1": {"predicted": "GOAL", "confidence": 0.8},
        "fresh-2": {"predicted": "GOAL", "confidence": 0.6},
        "retention-1": {"predicted": "COPY_VALUE", "confidence": 0.9},
        "oov-1": {"predicted": "PREDICTION", "confidence": 0.7},
        "composition-1": {"predicted": ["CONSTRAINT", "GOAL"], "confidence": 0.7},
        "composition-2": {"predicted": ["OBSERVATION"], "confidence": 0.6},
        "calibration-1": {"predicted": "GOAL", "confidence": 0.8},
    }


def _challenger_predictions() -> dict:
    return {
        "fresh-1": {"predicted": "GOAL", "confidence": 0.9},
        "fresh-2": {"predicted": "CONSTRAINT", "confidence": 0.8},
        "retention-1": {"predicted": "COPY_VALUE", "confidence": 0.9},
        "oov-1": {"predicted": "PREDICTION", "confidence": 0.75},
        "composition-1": {"predicted": ["GOAL", "CONSTRAINT"], "confidence": 0.8},
        "composition-2": {
            "predicted": ["PREDICTION", "OBSERVATION"],
            "confidence": 0.75,
        },
        "calibration-1": {"predicted": "GOAL", "confidence": 0.85},
    }


def _calibration_fixture(tmp_path, *, temperature: float = 1.7):
    suite_path = tmp_path / "suite.json"
    suite_path.write_text(json.dumps(_suite()), encoding="utf-8")
    fit_path = tmp_path / "calibration.jsonl"
    fit_path.write_text(
        json.dumps(
            {
                "id": "fit-1",
                "split": "calibration_fit",
                "text": "independent calibration surface",
                "frame_kinds": ["GOAL"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source_path = tmp_path / "parent.pt"
    source_checkpoint = {
        "schema": "kev.test-checkpoint.v1",
        "state_dict": {"projection.weight": torch.tensor([[1.0, 2.0]])},
        "lineage": {"parent_sha256": "0" * 64},
        "temperature": None,
        "score_status": "UNCALIBRATED",
    }
    torch.save(source_checkpoint, source_path)
    parent_hash = file_sha256(source_path)
    fit_hash = file_sha256(fit_path)
    suite_file_hash = file_sha256(suite_path)
    suite_canonical_hash = canonical_json_sha256(_suite())
    checkpoint = tmp_path / "candidate.pt"
    calibrated_checkpoint = dict(source_checkpoint)
    calibrated_checkpoint.update(
        {
            "temperature": temperature,
            "score_status": "CALIBRATED",
            "calibration_parent_sha256": parent_hash,
            "calibration_suite_sha256": fit_hash,
            "promotion_suite_file_sha256": suite_file_hash,
            "promotion_suite_canonical_sha256": suite_canonical_hash,
        }
    )
    torch.save(calibrated_checkpoint, checkpoint)
    receipt = {
        "schema": "kev.calibration-receipt.v1",
        "temperature": temperature,
        "calibration_suite": str(fit_path.resolve()),
        "calibration_suite_sha256": fit_hash,
        "fit_data_sha256": fit_hash,
        "calibration_fit_split": "calibration_fit",
        "source_sha256": parent_hash,
        "source_path": str(source_path.resolve()),
        "output_sha256": file_sha256(checkpoint),
        "promotion_suite": str(suite_path.resolve()),
        "promotion_suite_sha256": suite_file_hash,
        "promotion_suite_file_sha256": suite_file_hash,
        "promotion_suite_canonical_sha256": suite_canonical_hash,
        "score_status": "CALIBRATED",
    }
    return checkpoint, suite_path, fit_path, receipt


def test_canonical_and_file_hashes_distinguish_value_from_formatting(tmp_path):
    first = {"b": [2, 3], "a": 1}
    second = {"a": 1, "b": [2, 3]}
    assert canonical_json_sha256(first) == canonical_json_sha256(second)

    compact = tmp_path / "compact.json"
    pretty = tmp_path / "pretty.json"
    compact.write_text(json.dumps(first, separators=(",", ":")), encoding="utf-8")
    pretty.write_text(json.dumps(first, indent=2), encoding="utf-8")
    assert file_sha256(compact) != file_sha256(pretty)
    assert (
        json_artifact_hashes(compact)["canonical_json_sha256"]
        == json_artifact_hashes(pretty)["canonical_json_sha256"]
    )


def test_frozen_suite_load_hash_check_and_callback_evaluation(tmp_path):
    path = tmp_path / "suite.json"
    path.write_text(json.dumps(_suite(), indent=2), encoding="utf-8")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    suite = load_frozen_suite(path, expected_file_sha256=expected)

    predictions = _challenger_predictions()
    seen_items = []

    def predict_without_answer_key(item):
        assert not ({"expected", "expected_frames", "target"} & set(item))
        seen_items.append(item["id"])
        return predictions[item["id"]]

    report = evaluate_suite(suite, predictor=predict_without_answer_key)
    assert report["suite"]["file_sha256"] == expected
    assert report["suite"]["canonical_sha256"] == canonical_json_sha256(_suite())
    assert report["splits"]["fresh"]["correct"] == 2
    assert report["splits"]["composition"]["correct"] == 2
    assert report["calibration"]["status"] == "UNCALIBRATED"
    assert report["raw_failures"] == {"available": True, "count": 0, "items": []}
    assert len(seen_items) == 7
    assert len(report["prediction_evidence_sha256"]) == 64
    assert len(report["report_sha256"]) == 64

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_frozen_suite(path, expected_file_sha256="0" * 64)


def test_challenger_gate_requires_fresh_and_composition_improvement(tmp_path):
    incumbent = tmp_path / "incumbent.pt"
    challenger = tmp_path / "challenger.pt"
    incumbent.write_bytes(b"incumbent immutable bytes")
    challenger.write_bytes(b"challenger immutable bytes")

    report = evaluate_challenger(
        _suite(),
        incumbent_prediction_records=_incumbent_predictions(),
        challenger_prediction_records=_challenger_predictions(),
        incumbent_path=incumbent,
        challenger_path=challenger,
        retention_epsilon=0.0,
    )
    assert report["decision"]["decision"] == "REJECT"
    assert report["decision"]["promote"] is False
    assert (
        "EXTERNAL_PREDICTIONS_PROMOTION_INELIGIBLE"
        in (report["decision"]["reason_codes"])
    )
    assert report["incumbent"]["prediction_authority"] == (
        "EXTERNAL_RECORDS_UNVERIFIED"
    )
    assert report["decision"]["policy"]["training_loss_used"] is False
    assert report["incumbent"]["checkpoint"]["sha256"] == file_sha256(incumbent)
    assert report["challenger"]["checkpoint"]["sha256"] == file_sha256(challenger)
    assert report["raw_failures"]["incumbent"]["count"] == 2

    tied = promotion_decision(
        {
            "fresh": {"correct": 1, "total": 2},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
        },
        {
            "fresh": {"correct": 1, "total": 2},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
        },
    )
    assert tied["decision"] == "REJECT"
    assert "FRESH_TIE" in tied["reason_codes"]

    single_label_only_gain = promotion_decision(
        {
            "fresh": {"correct": 1, "total": 2},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
            "composition": {"correct": 1, "total": 2},
        },
        {
            "fresh": {"correct": 2, "total": 2},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
            "composition": {"correct": 1, "total": 2},
        },
    )
    assert single_label_only_gain["decision"] == "REJECT"
    assert "COMPOSITION_TIE" in single_label_only_gain["reason_codes"]


def test_calibration_receipt_changes_status_but_not_promotion_features(tmp_path):
    checkpoint, suite_path, fit_path, artifact = _calibration_fixture(tmp_path)
    predictions = {
        item_id: {
            **prediction,
            "score_status": "CALIBRATED",
            "temperature": 1.7,
            "model_sha256": file_sha256(checkpoint),
        }
        for item_id, prediction in _challenger_predictions().items()
    }
    report = evaluate_suite(
        suite_path,
        prediction_records=predictions,
        checkpoint_path=checkpoint,
        temperature_artifact=artifact,
        calibration_fit_path=fit_path,
    )
    assert report["calibration"]["status"] == "CALIBRATED"
    assert report["calibration"]["temperature"] == 1.7
    assert report["calibration"][
        "temperature_artifact_canonical_sha256"
    ] == canonical_json_sha256(artifact)
    fresh_one = next(row for row in report["records"] if row["item_id"] == "fresh-1")
    assert fresh_one["confidence"] == 0.9


def test_generic_temperature_cannot_mark_scores_calibrated():
    with pytest.raises(ValueError, match="generic temperature is insufficient"):
        load_temperature_artifact(
            {
                "schema": "kev.temperature.v1",
                "temperature": 1.7,
                "fit_data_sha256": "a" * 64,
                "fit_split": "held-out-calibration",
            }
        )


def test_calibration_receipt_must_bind_evaluated_checkpoint_and_suite(tmp_path):
    checkpoint, suite_path, fit_path, receipt = _calibration_fixture(tmp_path)
    predictions = {
        item_id: {
            **prediction,
            "score_status": "CALIBRATED",
            "temperature": 1.7,
        }
        for item_id, prediction in _challenger_predictions().items()
    }

    wrong_checkpoint = dict(receipt)
    wrong_checkpoint["output_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="evaluated checkpoint"):
        evaluate_suite(
            suite_path,
            prediction_records=predictions,
            checkpoint_path=checkpoint,
            temperature_artifact=wrong_checkpoint,
            calibration_fit_path=fit_path,
        )

    wrong_suite = dict(receipt)
    wrong_suite["promotion_suite_sha256"] = "d" * 64
    with pytest.raises(ValueError, match="file hashes disagree"):
        evaluate_suite(
            suite_path,
            prediction_records=predictions,
            checkpoint_path=checkpoint,
            temperature_artifact=wrong_suite,
            calibration_fit_path=fit_path,
        )


def test_calibrated_predictor_scores_must_name_the_bound_checkpoint(tmp_path):
    checkpoint, suite_path, fit_path, receipt = _calibration_fixture(tmp_path)
    predictions = {
        item_id: {
            **prediction,
            "score_status": "CALIBRATED",
            "temperature": 1.7,
            "model_sha256": "e" * 64,
        }
        for item_id, prediction in _challenger_predictions().items()
    }

    report = evaluate_suite(
        suite_path,
        prediction_records=predictions,
        checkpoint_path=checkpoint,
        temperature_artifact=receipt,
        calibration_fit_path=fit_path,
    )

    assert report["raw_failures"]["count"] == len(predictions)
    assert all(
        "prediction model SHA-256" in failure["error"]["message"]
        for failure in report["raw_failures"]["items"]
    )


def test_embedded_temperature_without_receipt_does_not_mark_scores_calibrated():
    predictions = {
        item_id: {
            **prediction,
            "score_status": "CALIBRATED",
            "temperature": 1.7,
        }
        for item_id, prediction in _challenger_predictions().items()
    }
    report = evaluate_suite(_suite(), prediction_records=predictions)

    assert report["calibration"]["status"] == "UNCALIBRATED"
    assert all(record["confidence"] is None for record in report["records"])


def test_published_190_of_260_tie_replays_as_aggregate_only_rejection():
    report = replay_historical_aggregate(
        parent_correct=190,
        challenger_correct=190,
        total=260,
    )
    assert report["decision"] == "REJECT"
    assert report["promote"] is False
    assert report["reason_codes"] == ["FRESH_TIE"]
    assert report["raw_failures"]["available"] is False
    assert report["raw_failures"]["items"] == []
    assert report["calibration"]["status"] == "UNCALIBRATED"
    assert report["checkpoint_hashes"]["status"] == "NOT_PRESENT_IN_PUBLISHED_AGGREGATE"
    assert len(report["aggregate_sha256"]) == 64
    assert len(report["report_sha256"]) == 64


def test_audit_family_regression_or_missing_metrics_rejects():
    incumbent = {
        "suite": {"canonical_sha256": "a" * 64},
        "splits": {
            "fresh": {"correct": 1, "total": 3},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
        },
        "audit_metrics": {
            "negation": {"correct": 1, "total": 1},
            "number-change": {"correct": 0, "total": 2},
        },
        "raw_failures": {"available": True, "count": 2, "items": []},
    }
    challenger = {
        "suite": {"canonical_sha256": "a" * 64},
        "splits": {
            "fresh": {"correct": 2, "total": 3},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
        },
        "audit_metrics": {
            "negation": {"correct": 0, "total": 1},
            "number-change": {"correct": 2, "total": 2},
        },
        "raw_failures": {"available": True, "count": 1, "items": []},
    }
    regressed = compare_evaluations(incumbent, challenger)
    assert regressed["decision"] == "REJECT"
    assert "AUDIT_NEGATION_REGRESSED" in regressed["reason_codes"]
    assert regressed["policy"]["audit_family_count"] == 2

    missing = json.loads(json.dumps(challenger))
    missing["audit_metrics"].pop("number-change")
    missing_result = compare_evaluations(incumbent, missing)
    assert missing_result["decision"] == "REJECT"
    assert "AUDIT_NUMBER_CHANGE_METRICS_MISSING" in missing_result["reason_codes"]

    mismatched = json.loads(json.dumps(challenger))
    mismatched["audit_metrics"]["number-change"]["total"] = 3
    mismatch_result = compare_evaluations(incumbent, mismatched)
    assert mismatch_result["decision"] == "REJECT"
    assert "AUDIT_NUMBER_CHANGE_SUITE_SIZE_MISMATCH" in mismatch_result["reason_codes"]

    missing_counts = json.loads(json.dumps(challenger))
    missing_counts["audit_metrics"]["number-change"].pop("correct")
    missing_counts_result = compare_evaluations(incumbent, missing_counts)
    assert missing_counts_result["decision"] == "REJECT"
    assert (
        "AUDIT_NUMBER_CHANGE_METRICS_MISSING" in missing_counts_result["reason_codes"]
    )

    invalid_counts = json.loads(json.dumps(challenger))
    invalid_counts["audit_metrics"]["number-change"]["correct"] = 3
    invalid_counts_result = compare_evaluations(incumbent, invalid_counts)
    assert invalid_counts_result["decision"] == "REJECT"
    assert (
        "AUDIT_NUMBER_CHANGE_METRICS_INVALID" in invalid_counts_result["reason_codes"]
    )


def test_audit_family_ties_pass_when_split_gates_pass():
    incumbent = {
        "suite": {"canonical_sha256": "b" * 64},
        "splits": {
            "fresh": {"correct": 1, "total": 2},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
        },
        "audit_metrics": {
            "polite-buried-constraint": {"correct": 1, "total": 1},
            "semantic-single-frame-paraphrase": {"correct": 0, "total": 1},
        },
        "raw_failures": {"available": True, "count": 1, "items": []},
    }
    challenger = {
        "suite": {"canonical_sha256": "b" * 64},
        "splits": {
            "fresh": {"correct": 2, "total": 2},
            "retention": {"correct": 2, "total": 2},
            "oov": {"correct": 1, "total": 1},
        },
        "audit_metrics": {
            "polite-buried-constraint": {"correct": 1, "total": 1},
            "semantic-single-frame-paraphrase": {"correct": 0, "total": 1},
        },
        "raw_failures": {"available": True, "count": 0, "items": []},
    }
    result = compare_evaluations(incumbent, challenger)
    assert result["decision"] == "QUALIFY"
    assert result["reason_codes"] == ["ALL_PROMOTION_GATES_PASSED"]
    audit_gates = [
        gate for gate in result["gates"] if gate["name"].startswith("audit:")
    ]
    assert len(audit_gates) == 2
    assert all(gate["passed"] for gate in audit_gates)

from __future__ import annotations

import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import kev.evolution as evolution
import pytest
import torch

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.calibration import calibrate_checkpoint
from kev.evaluation import load_frozen_suite
from kev.evolution import evolve
from kev.uc51a2.semantic_breadth import (
    _frozen_evaluation_exclusions,
    required_held_out_vocabulary,
)
from kev.uc51a3.alive import AliveStore, IncumbentCompareAndSwapError


_OBJECTIVE = "L_relation + 0.20 L_cardinality + 0.35 L_contrastive"


def _checkpoint_stub(path):
    return SimpleNamespace(sha256=file_sha256(path))


def _decision_report(challenger_path, decision, reason_codes):
    return {
        "challenger": {
            "checkpoint": {"sha256": file_sha256(challenger_path)},
        },
        "decision": {
            "decision": decision,
            "decision_sha256": "d" * 64,
            "reason_codes": reason_codes,
        },
    }


def _write_fake_valid_training_artifacts(
    parent_path,
    output_dir,
    *,
    extra_jsonl,
    held_out_vocabulary,
    steps,
    seed,
    mismatch: str | None = None,
    **_kwargs,
):
    """Create a minimal but fully lineage-bound trainer result for loop tests."""

    parent = Path(parent_path).resolve()
    lessons = Path(extra_jsonl).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True)
    challenger = output / "semantic-breadth.pt"
    parent_sha256 = file_sha256(parent)
    lessons_sha256 = file_sha256(lessons)
    required_terms, required_artifacts, required_artifacts_sha256 = (
        required_held_out_vocabulary()
    )
    required_terms = sorted(required_terms)
    required_terms_sha256 = canonical_json_sha256(required_terms)
    held_out_terms = sorted(
        set(required_terms)
        | {
            str(term).casefold().strip()
            for term in held_out_vocabulary
            if str(term).strip()
        }
    )
    held_out_terms_sha256 = canonical_json_sha256(held_out_terms)
    pinned_hash = next(
        artifact["sha256"]
        for artifact in required_artifacts
        if artifact["path"].endswith("held-out-vocabulary-v1.txt")
    )
    optimizer = {
        "name": "AdamW",
        "learning_rate": 5e-4,
        "weight_decay": 2e-4,
        "gradient_norm_clip": 1.0,
    }
    parameter_scope = {"total": 3, "trainable": 2, "frozen": 1}
    trainer_source_sha256 = file_sha256(
        evolution.ROOT / "kev" / "uc51a2" / "semantic_breadth.py"
    )
    _, exclusion_artifacts = _frozen_evaluation_exclusions()
    exclusions_sha256 = canonical_json_sha256(
        sorted(
            (
                {"role": artifact["role"], "sha256": artifact["sha256"]}
                for artifact in exclusion_artifacts
            ),
            key=lambda item: (item["role"], item["sha256"]),
        )
    )
    training_inputs = {
        "parent_sha256": parent_sha256,
        "lessons_sha256": lessons_sha256,
        "held_out_vocabulary_sha256": held_out_terms_sha256,
        "pinned_held_out_vocabulary_file_sha256": pinned_hash,
        "required_held_out_vocabulary_artifacts_sha256": (required_artifacts_sha256),
        "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
        "training_exclusions_sha256": exclusions_sha256,
        "paraphrase_groups_sha256": "2" * 64,
        "trainer_source_sha256": trainer_source_sha256,
        "seed": seed,
        "steps": steps,
        "objective": _OBJECTIVE,
        "optimizer": optimizer,
        "torch_num_threads": 2,
    }
    if mismatch == "training_inputs":
        training_inputs["lessons_sha256"] = "0" * 64
    training_inputs_sha256 = canonical_json_sha256(training_inputs)
    shared = {
        "parent_sha256": parent_sha256,
        "lessons_sha256": lessons_sha256,
        "held_out_vocabulary_sha256": held_out_terms_sha256,
        "pinned_held_out_vocabulary_file_sha256": pinned_hash,
        "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
        "required_held_out_vocabulary_artifacts": required_artifacts,
        "required_held_out_vocabulary_artifacts_sha256": (required_artifacts_sha256),
        "training_exclusions_sha256": exclusions_sha256,
        "paraphrase_groups_sha256": "2" * 64,
        "contrastive_positive_pairs": 0,
        "trainer_source_sha256": trainer_source_sha256,
        "training_inputs_sha256": training_inputs_sha256,
        "training_scope": "projection_and_heads_only",
        "seed": seed,
        "steps": steps,
        "objective": _OBJECTIVE,
        "optimizer": optimizer,
        "torch_num_threads": 2,
        "score_status": "UNCALIBRATED",
        "artifact_status": "CHALLENGER",
    }
    parent_checkpoint = torch.load(parent, map_location="cpu", weights_only=True)
    checkpoint = {
        "schema": "kev.semantic-frames.v052",
        "state_dict": parent_checkpoint["state_dict"],
        "parameters": parent_checkpoint["parameters"],
        "parameter_scope": parameter_scope,
        "temperature": None,
        **shared,
    }
    if mismatch == "checkpoint_metadata":
        checkpoint["parent_sha256"] = "0" * 64
    torch.save(checkpoint, challenger)

    receipt = {
        "schema": "kev.training-receipt.v1",
        "activation": "NONE",
        **shared,
        "parent_path": str(parent),
        "lessons_path": str(lessons),
        "held_out_vocabulary": held_out_terms,
        "required_held_out_vocabulary": required_terms,
        "challenger_path": str(challenger),
        "challenger_sha256": file_sha256(challenger),
        "parameter_scope": parameter_scope,
        "training_inputs": training_inputs,
        "training_exclusion_artifacts": exclusion_artifacts,
        "loss_history": [
            {
                "step": step,
                "loss": 1.0,
                "relation": 0.5,
                "cardinality": 0.5,
                "contrastive": 0.0,
            }
            for step in range(1, steps + 1)
        ],
        "promotion_features": [],
    }
    if mismatch == "parent_receipt":
        receipt["parent_sha256"] = "0" * 64
    elif mismatch == "lessons_receipt":
        receipt["lessons_sha256"] = "0" * 64
    elif mismatch == "challenger_hash":
        receipt["challenger_sha256"] = "0" * 64
    elif mismatch == "challenger_path":
        receipt["challenger_path"] = str(output / "unrelated.pt")
    (output / "training-receipt.json").write_text(
        json.dumps(receipt, sort_keys=True), encoding="utf-8"
    )
    return receipt


def test_evolve_runs_closed_gate_and_preserves_rejected_candidate(tmp_path: Path):
    lessons = tmp_path / "reviewed-lessons.jsonl"
    lessons.write_text(
        json.dumps(
            {
                "id": "lesson-evolution-test",
                "status": "REVIEWED",
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "permission": "human-authored and approved for this local test",
                "text": "Aim to minimize the sandbox response delay",
                "frame_kinds": ["GOAL"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    state_dir = tmp_path / "state"
    before = AliveStore(state_dir).read()

    result = evolve(
        lessons,
        state_dir=state_dir,
        output_root=tmp_path / "candidates",
        steps=1,
        seed=52021,
    )

    assert result["decision"] == "REJECT"
    assert result["candidate_preserved"] is True
    assert result["old_incumbent_preserved"] is True
    assert result["training_loss_used_for_promotion"] is False
    challenger = Path(result["challenger"]["path"])
    eval_card = Path(result["eval_card"]["path"])
    assert challenger.is_file()
    assert file_sha256(challenger) == result["challenger"]["sha256"]
    assert eval_card.is_file()
    assert file_sha256(eval_card) == result["eval_card"]["sha256"]

    after = AliveStore(state_dir).read()
    assert after["semantic_model"] == before["semantic_model"]
    assert after["semantic_model_sha256"] == before["semantic_model_sha256"]
    assert AliveStore(state_dir).verify_ledger()["valid"] is True

    events = [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    decision_event = next(
        event for event in events if event["kind"] == "MODEL_REJECTED"
    )
    assert decision_event["data"]["training_loss_used_for_promotion"] is False
    assert decision_event["data"]["input_artifacts"]["promotion_suite_file_sha256"]
    training_receipt = json.loads(
        Path(decision_event["data"]["training_receipt_path"]).read_text(
            encoding="utf-8"
        )
    )
    assert training_receipt["objective"] == (
        "L_relation + 0.20 L_cardinality + 0.35 L_contrastive"
    )
    assert training_receipt["promotion_features"] == []
    assert training_receipt["training_scope"] == "projection_and_heads_only"
    assert training_receipt["parameter_scope"]["frozen"] > 0
    assert training_receipt["parameter_scope"]["trainable"] > 0

    parent = torch.load(before["semantic_model"], map_location="cpu", weights_only=True)
    trained = torch.load(
        Path(decision_event["data"]["training_receipt_path"]).parent
        / "semantic-breadth.pt",
        map_location="cpu",
        weights_only=True,
    )
    for key in ("enc.0.weight", "enc.0.bias", "enc.2.weight", "enc.2.bias"):
        assert torch.equal(parent["state_dict"][key], trained["state_dict"][key])
    assert not torch.equal(
        parent["state_dict"]["enc.3.weight"], trained["state_dict"]["enc.3.weight"]
    )


def test_evolve_invalid_input_still_writes_reject_receipt(tmp_path: Path):
    state_dir = tmp_path / "state"
    try:
        evolve(
            tmp_path / "missing-lessons.jsonl",
            state_dir=state_dir,
            output_root=tmp_path / "candidates",
            steps=1,
        )
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("missing reviewed lessons must fail")

    events = [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    rejected = next(event for event in events if event["kind"] == "MODEL_REJECTED")
    failure = Path(rejected["data"]["raw_failure_path"])
    assert failure.is_file()
    assert file_sha256(failure) == rejected["data"]["raw_failure_sha256"]


@pytest.mark.parametrize(
    ("mismatch", "message"),
    [
        ("parent_receipt", "parent_sha256"),
        ("lessons_receipt", "lessons_sha256"),
        ("training_inputs", "training_inputs lessons_sha256"),
        ("challenger_hash", "challenger_sha256"),
        ("challenger_path", "challenger_path"),
        ("checkpoint_metadata", "checkpoint parent_sha256"),
    ],
)
def test_evolve_rejects_unverified_training_lineage_before_evaluation(
    tmp_path: Path,
    monkeypatch,
    mismatch: str,
    message: str,
):
    lessons = tmp_path / "reviewed-lessons.jsonl"
    lessons.write_text(
        json.dumps(
            {
                "id": "lesson-lineage-validation",
                "status": "REVIEWED",
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "permission": "approved for this local test",
                "text": "Keep candidate lineage evidence exact",
                "frame_kinds": ["CONSTRAINT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    held_out = tmp_path / "held-out.txt"
    held_out.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    initial = AliveStore(state_dir).read()
    suite = SimpleNamespace(splits={}, canonical_sha256="c" * 64, file_sha256="f" * 64)

    def mismatched_train(*args, **kwargs):
        return _write_fake_valid_training_artifacts(*args, **kwargs, mismatch=mismatch)

    evaluator_calls = 0

    def forbidden_evaluation(*_args, **_kwargs):
        nonlocal evaluator_calls
        evaluator_calls += 1
        pytest.fail("unverified challenger must not reach evaluation")

    monkeypatch.setattr(evolution, "train", mismatched_train)
    monkeypatch.setattr(evolution, "_verified_gate_artifacts", lambda **_kwargs: suite)
    monkeypatch.setattr(evolution, "evaluate_challenger", forbidden_evaluation)

    with pytest.raises(ValueError, match=message):
        evolve(
            lessons,
            state_dir=state_dir,
            suite_path=tmp_path / "synthetic-suite.json",
            calibration_path=None,
            held_out_vocabulary_path=held_out,
            output_root=tmp_path / "candidates",
            steps=2,
            seed=52021,
        )

    assert evaluator_calls == 0
    final = AliveStore(state_dir).read()
    assert final["semantic_model"] == initial["semantic_model"]
    assert final["semantic_model_sha256"] == initial["semantic_model_sha256"]

    events = [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    terminal = [
        event
        for event in events
        if event["kind"] in {"MODEL_REJECTED", "MODEL_QUALIFIED"}
    ]
    assert len(terminal) == 1
    assert terminal[0]["kind"] == "MODEL_REJECTED"
    failure_path = Path(terminal[0]["data"]["raw_failure_path"])
    assert failure_path.is_file()
    assert file_sha256(failure_path) == terminal[0]["data"]["raw_failure_sha256"]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "REJECTED_PIPELINE_FAILURE"
    assert message.replace("\\", "") in failure["error"]["message"]
    run_dir = failure_path.parent
    assert (run_dir / "trained" / "semantic-breadth.pt").is_file()
    assert (run_dir / "trained" / "training-receipt.json").is_file()
    assert AliveStore(state_dir).verify_ledger()["valid"] is True


def test_qualified_state_is_not_rolled_back_or_rejected_when_registry_projection_fails(
    tmp_path: Path,
    monkeypatch,
):
    lessons = tmp_path / "reviewed-lessons.jsonl"
    lessons.write_text(
        json.dumps(
            {
                "id": "lesson-qualified-ordering",
                "status": "REVIEWED",
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "permission": "approved for this local test",
                "text": "Prefer evidence-bearing state transitions",
                "frame_kinds": ["CONSTRAINT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    held_out = tmp_path / "held-out.txt"
    held_out.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    initial = AliveStore(state_dir).read()

    suite = SimpleNamespace(splits={}, canonical_sha256="c" * 64, file_sha256="f" * 64)

    monkeypatch.setattr(evolution, "train", _write_fake_valid_training_artifacts)
    monkeypatch.setattr(evolution, "_verified_gate_artifacts", lambda **_kwargs: suite)
    monkeypatch.setattr(evolution, "CheckpointPredictor", _checkpoint_stub)
    monkeypatch.setattr(
        evolution,
        "evaluate_challenger",
        lambda *_args, **kwargs: _decision_report(
            kwargs["challenger_path"], "QUALIFY", []
        ),
    )

    challenger_path: Path | None = None

    def fail_registry_projection(store, _incumbent, challenger, _eval_card):
        nonlocal challenger_path
        challenger_path = Path(challenger["path"])
        committed = store.read()
        assert committed["semantic_model"] == str(challenger_path.resolve())
        assert committed["semantic_model_sha256"] == challenger["sha256"]
        kinds = [
            json.loads(line)["kind"]
            for line in store.ledger_path.read_text(encoding="utf-8").splitlines()
        ]
        assert kinds[-1] == "MODEL_QUALIFIED"
        raise OSError("simulated registry projection failure")

    monkeypatch.setattr(evolution, "_update_local_registry", fail_registry_projection)

    with pytest.raises(OSError, match="registry projection"):
        evolve(
            lessons,
            state_dir=state_dir,
            suite_path=tmp_path / "synthetic-suite.json",
            calibration_path=None,
            held_out_vocabulary_path=held_out,
            output_root=tmp_path / "candidates",
            steps=1,
        )

    assert challenger_path is not None
    final = AliveStore(state_dir).read()
    assert final["semantic_model"] == str(challenger_path.resolve())
    assert final["semantic_model"] != initial["semantic_model"]
    events = [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "MODEL_QUALIFIED",
        "QUALIFICATION_POSTPROCESSING_FAILED",
    ]
    assert events[1]["data"]["terminal_decision"] == {
        "kind": "MODEL_QUALIFIED",
        "ledger_event": events[0]["hash"],
    }
    assert AliveStore(state_dir).verify_ledger()["valid"] is True
    failures = list((tmp_path / "candidates").glob("*/raw-failure.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text(encoding="utf-8"))
    assert failure["status"] == "POST_QUALIFICATION_RECORDING_FAILURE"
    assert failure["terminal_decision"] == {
        "kind": "MODEL_QUALIFIED",
        "ledger_event": events[0]["hash"],
    }


def test_rejected_decision_is_not_duplicated_when_result_write_fails(
    tmp_path: Path,
    monkeypatch,
):
    lessons = tmp_path / "reviewed-lessons.jsonl"
    lessons.write_text(
        json.dumps(
            {
                "id": "lesson-reject-result-failure",
                "status": "REVIEWED",
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "permission": "approved for this local test",
                "text": "Prefer bounded evidence transitions",
                "frame_kinds": ["CONSTRAINT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    held_out = tmp_path / "held-out.txt"
    held_out.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    suite = SimpleNamespace(splits={}, canonical_sha256="c" * 64, file_sha256="f" * 64)

    monkeypatch.setattr(evolution, "train", _write_fake_valid_training_artifacts)
    monkeypatch.setattr(evolution, "_verified_gate_artifacts", lambda **_kwargs: suite)
    monkeypatch.setattr(evolution, "CheckpointPredictor", _checkpoint_stub)
    monkeypatch.setattr(
        evolution,
        "evaluate_challenger",
        lambda *_args, **kwargs: _decision_report(
            kwargs["challenger_path"], "REJECT", ["FRESH_TIE"]
        ),
    )
    real_write = evolution._write_immutable_json

    def fail_result_write(path, value):
        if Path(path).name == "evolution-result.json":
            raise OSError("simulated result write failure")
        real_write(path, value)

    monkeypatch.setattr(evolution, "_write_immutable_json", fail_result_write)

    with pytest.raises(OSError, match="result write"):
        evolve(
            lessons,
            state_dir=state_dir,
            suite_path=tmp_path / "synthetic-suite.json",
            calibration_path=None,
            held_out_vocabulary_path=held_out,
            output_root=tmp_path / "candidates",
            steps=1,
        )

    events = [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "MODEL_REJECTED",
        "REJECTION_POSTPROCESSING_FAILED",
    ]
    terminal = [
        event
        for event in events
        if event["kind"] in {"MODEL_REJECTED", "MODEL_QUALIFIED"}
    ]
    assert len(terminal) == 1
    assert events[1]["data"]["terminal_decision"] == {
        "kind": "MODEL_REJECTED",
        "ledger_event": events[0]["hash"],
    }
    failure_path = Path(events[1]["data"]["raw_failure_path"])
    assert file_sha256(failure_path) == events[1]["data"]["raw_failure_sha256"]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "POST_REJECTION_RECORDING_FAILURE"
    assert failure["terminal_decision"] == events[1]["data"]["terminal_decision"]
    assert AliveStore(state_dir).verify_ledger()["valid"] is True


def test_evolve_rejects_challenger_swapped_after_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    lessons = tmp_path / "reviewed-lessons.jsonl"
    lessons.write_text(
        json.dumps(
            {
                "id": "lesson-swap-check",
                "status": "REVIEWED",
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "permission": "approved for this local test",
                "text": "Keep evaluated bytes bound to activation",
                "frame_kinds": ["CONSTRAINT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    held_out = tmp_path / "held-out.txt"
    held_out.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    initial = AliveStore(state_dir).read()
    suite = SimpleNamespace(splits={}, canonical_sha256="c" * 64, file_sha256="f" * 64)

    monkeypatch.setattr(evolution, "train", _write_fake_valid_training_artifacts)
    monkeypatch.setattr(evolution, "_verified_gate_artifacts", lambda **_kwargs: suite)
    monkeypatch.setattr(evolution, "CheckpointPredictor", _checkpoint_stub)

    def swap_after_scoring(*_args, **kwargs):
        challenger_path = Path(kwargs["challenger_path"])
        report = _decision_report(challenger_path, "QUALIFY", [])
        challenger_path.write_bytes(b"unevaluated replacement")
        return report

    monkeypatch.setattr(evolution, "evaluate_challenger", swap_after_scoring)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evolve(
            lessons,
            state_dir=state_dir,
            suite_path=tmp_path / "synthetic-suite.json",
            calibration_path=None,
            held_out_vocabulary_path=held_out,
            output_root=tmp_path / "candidates",
            steps=1,
        )

    final = AliveStore(state_dir).read()
    assert final["semantic_model"] == initial["semantic_model"]
    assert final["semantic_model_sha256"] == initial["semantic_model_sha256"]
    terminal = [
        json.loads(line)
        for line in (state_dir / "ledger.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if json.loads(line)["kind"] in {"MODEL_REJECTED", "MODEL_QUALIFIED"}
    ]
    assert [event["kind"] for event in terminal] == ["MODEL_REJECTED"]


def test_evolution_qualification_rejects_noncanonical_suite(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="canonical manifest"):
        evolution._verified_default_suite(tmp_path / "caller-suite.json")


def test_evolution_qualification_requires_canonical_calibration() -> None:
    with pytest.raises(ValueError, match="canonical manifest calibration"):
        evolution._verified_gate_artifacts(
            suite_path=evolution.DEFAULT_SUITE,
            calibration_path=None,
            held_out_vocabulary_path=evolution.DEFAULT_HELD_OUT,
            registry_path=evolution.DEFAULT_REGISTRY,
        )


def _reviewed_lesson(path: Path, lesson_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "id": lesson_id,
                "status": "REVIEWED",
                "reviewed_by": "test-reviewer",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "permission": "approved for this local test",
                "text": "Preserve exact evidence across the model gate",
                "frame_kinds": ["CONSTRAINT"],
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _state_incumbent(state: dict) -> dict:
    return {
        "path": state["semantic_model"],
        "sha256": state["semantic_model_sha256"],
        "generation": state["semantic_model_generation"],
    }


@pytest.mark.parametrize("gate_decision", ["QUALIFY", "REJECT"])
def test_evolve_records_stale_gate_without_overwriting_concurrent_winner(
    tmp_path: Path,
    monkeypatch,
    gate_decision: str,
) -> None:
    lessons = tmp_path / "reviewed-lessons.jsonl"
    _reviewed_lesson(lessons, f"stale-{gate_decision.casefold()}")
    held_out = tmp_path / "held-out.txt"
    held_out.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    store = AliveStore(state_dir)
    original = _state_incumbent(store.read())
    winner = tmp_path / "concurrent-winner.pt"
    winner.write_bytes(b"concurrent winner")
    suite = SimpleNamespace(splits={}, canonical_sha256="c" * 64, file_sha256="f" * 64)

    monkeypatch.setattr(evolution, "train", _write_fake_valid_training_artifacts)
    monkeypatch.setattr(evolution, "_verified_gate_artifacts", lambda **_kwargs: suite)
    monkeypatch.setattr(evolution, "CheckpointPredictor", _checkpoint_stub)

    def qualify_concurrent_winner(*_args, **kwargs):
        store.record_model_decision(
            "MODEL_QUALIFIED",
            {"run_id": "concurrent-winner"},
            active_model={"path": str(winner), "sha256": file_sha256(winner)},
            expected_incumbent=original,
        )
        reasons = [] if gate_decision == "QUALIFY" else ["FRESH_TIE"]
        return _decision_report(kwargs["challenger_path"], gate_decision, reasons)

    monkeypatch.setattr(evolution, "evaluate_challenger", qualify_concurrent_winner)

    with pytest.raises(IncumbentCompareAndSwapError):
        evolve(
            lessons,
            state_dir=state_dir,
            suite_path=tmp_path / "synthetic-suite.json",
            calibration_path=None,
            held_out_vocabulary_path=held_out,
            output_root=tmp_path / "candidates",
            steps=1,
        )

    final = store.read()
    assert final["semantic_model"] == str(winner.resolve())
    assert final["semantic_model_sha256"] == file_sha256(winner)
    assert final["semantic_model_generation"] == original["generation"] + 1
    events = [
        json.loads(line)
        for line in store.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "MODEL_QUALIFIED",
        "MODEL_REJECTED",
    ]
    stale = events[-1]
    assert stale["data"]["reason_codes"] == ["STALE_INCUMBENT_COMPARE_AND_SWAP_FAILED"]
    assert stale["data"]["compare_and_swap"]["expected"] == original
    assert stale["data"]["compare_and_swap"]["current"]["generation"] == (
        original["generation"] + 1
    )
    failure_path = Path(stale["data"]["raw_failure_path"])
    assert file_sha256(failure_path) == stale["data"]["raw_failure_sha256"]
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    assert failure["status"] == "STALE_INCUMBENT_REJECTED"
    assert failure["compare_and_swap"] == stale["data"]["compare_and_swap"]
    assert store.verify_ledger()["valid"] is True


def test_evolve_rejects_incumbent_bytes_swapped_after_evaluation(
    tmp_path: Path, monkeypatch
) -> None:
    lessons = tmp_path / "reviewed-lessons.jsonl"
    _reviewed_lesson(lessons, "incumbent-byte-swap")
    held_out = tmp_path / "held-out.txt"
    held_out.write_text("", encoding="utf-8")
    state_dir = tmp_path / "state"
    store = AliveStore(state_dir)
    original = _state_incumbent(store.read())
    incumbent_copy = tmp_path / "incumbent-copy.pt"
    shutil.copy2(original["path"], incumbent_copy)
    store.record_model_decision(
        "MODEL_QUALIFIED",
        {"run_id": "test-incumbent-copy"},
        active_model={
            "path": str(incumbent_copy),
            "sha256": file_sha256(incumbent_copy),
        },
        expected_incumbent=original,
    )
    before_evolution = store.read()
    suite = SimpleNamespace(splits={}, canonical_sha256="c" * 64, file_sha256="f" * 64)

    monkeypatch.setattr(evolution, "train", _write_fake_valid_training_artifacts)
    monkeypatch.setattr(evolution, "_verified_gate_artifacts", lambda **_kwargs: suite)
    monkeypatch.setattr(evolution, "CheckpointPredictor", _checkpoint_stub)

    def swap_incumbent(*_args, **kwargs):
        report = _decision_report(kwargs["challenger_path"], "QUALIFY", [])
        incumbent_copy.write_bytes(b"unevaluated incumbent replacement")
        return report

    monkeypatch.setattr(evolution, "evaluate_challenger", swap_incumbent)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evolve(
            lessons,
            state_dir=state_dir,
            suite_path=tmp_path / "synthetic-suite.json",
            calibration_path=None,
            held_out_vocabulary_path=held_out,
            output_root=tmp_path / "candidates",
            steps=1,
        )

    final = store.read()
    assert final["semantic_model"] == before_evolution["semantic_model"]
    assert final["semantic_model_sha256"] == before_evolution["semantic_model_sha256"]
    assert (
        final["semantic_model_generation"]
        == before_evolution["semantic_model_generation"]
    )
    events = [
        json.loads(line)
        for line in store.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == [
        "MODEL_QUALIFIED",
        "MODEL_REJECTED",
    ]
    assert events[-1]["data"]["reason_codes"] == ["PIPELINE_FAILURE"]
    assert store.verify_ledger()["valid"] is True


@pytest.mark.parametrize(
    "swapped_artifact",
    [
        "raw_challenger_path",
        "receipt_path",
        "calibration_fit_path",
        "promotion_suite_path",
    ],
)
def test_preterminal_calibration_reverification_rejects_artifact_swaps(
    tmp_path: Path,
    monkeypatch,
    swapped_artifact: str,
) -> None:
    state = AliveStore(tmp_path / "seed-state").read()
    raw = tmp_path / "raw-challenger.pt"
    fit = tmp_path / "calibration-fit.jsonl"
    suite_path = tmp_path / "promotion-suite.json"
    shutil.copy2(state["semantic_model"], raw)
    shutil.copy2(evolution.DEFAULT_CALIBRATION, fit)
    shutil.copy2(evolution.DEFAULT_SUITE, suite_path)
    suite = load_frozen_suite(suite_path)
    references = {
        "calibration_fit": {"path": str(fit), "sha256": file_sha256(fit)},
        "promotion_suite": {
            "path": str(suite_path),
            "sha256": suite.file_sha256,
            "canonical_sha256": suite.canonical_sha256,
        },
    }
    monkeypatch.setattr(
        evolution,
        "_manifest_artifact",
        lambda role, _registry_path: references[role],
    )
    receipt = calibrate_checkpoint(
        raw,
        fit,
        tmp_path / "calibrated",
        promotion_suite=suite_path,
        steps=1,
    )
    verified = evolution._validate_calibration_artifacts(
        receipt,
        calibration_dir=tmp_path / "calibrated",
        raw_challenger_path=raw,
        raw_challenger_sha256=file_sha256(raw),
        calibration_fit_path=fit,
        promotion_suite_path=suite_path,
        frozen_suite=suite,
        registry_path=tmp_path / "unused-registry.json",
    )

    swapped_path = Path(verified[swapped_artifact])
    with swapped_path.open("ab") as handle:
        handle.write(b"\n")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        evolution._reverify_calibration_artifacts(verified)

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.evaluation import evaluate_challenger, load_frozen_suite
from kev.uc51a2.semantic_breadth import (
    PINNED_HELD_OUT_VOCABULARY,
    _frozen_evaluation_exclusions,
    required_held_out_vocabulary,
)
from kev.uc51a3.alive import AliveStore
from scripts import run_preregistered_experiment as runner


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _inputs(tmp_path: Path) -> dict[str, Path]:
    suite = runner.DEFAULT_SUITE
    calibration = runner.DEFAULT_CALIBRATION
    held_out = runner.DEFAULT_HELD_OUT
    corpus = tmp_path / "reviewed-corpus.jsonl"
    corpus.write_text(
        '{"id":"lesson-1","status":"REVIEWED","text":"bounded fixture"}\n',
        encoding="utf-8",
    )
    encoder_file = tmp_path / "encoder" / "weights.fixture"
    encoder_file.parent.mkdir()
    encoder_file.write_bytes(b"frozen encoder fixture")
    encoder_manifest = encoder_file.parent / "encoder-manifest.json"
    _json(
        encoder_manifest,
        {
            "schema": "kev.local-encoder.v1",
            "name": "test-frozen-encoder",
            "embedding_dim": 3,
            "network_policy": "LOCAL_ONLY",
            "permission": {
                "status": "APPROVED",
                "reviewed_by": "test",
                "reviewed_at": "2026-09-28T00:00:00Z",
                "scope": "test fixture only",
            },
            "license": {"identifier": "test-only"},
            "files": [
                {
                    "path": encoder_file.name,
                    "sha256": file_sha256(encoder_file),
                }
            ],
        },
    )
    return {
        "suite": suite,
        "calibration": calibration,
        "held_out": held_out,
        "corpus": corpus,
        "encoder_manifest": encoder_manifest,
    }


def _plan(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    inputs = _inputs(tmp_path)
    plan_path = tmp_path / "experiment-plan.json"
    output_root = tmp_path / "experiment-output"
    runner.create_plan(
        plan_path=plan_path,
        experiment_id="runner-test",
        output_root=output_root,
        public_registry=runner.PUBLIC_REGISTRY,
        promotion_suite=inputs["suite"],
        calibration_fit=inputs["calibration"],
        held_out_vocabulary=inputs["held_out"],
        encoder_manifest=inputs["encoder_manifest"],
        reviewed_corpus=inputs["corpus"],
    )
    return plan_path, output_root, inputs


def _install_fake_evolve(
    monkeypatch: pytest.MonkeyPatch,
    plan_path: Path,
    *,
    decisions: dict[int, str],
    failing_seed: int | None = None,
    non_seed_variant_seed: int | None = None,
    exclusion_variant_seed: int | None = None,
    source_manifest_variant_seed: int | None = None,
    terminal_calibration_path_variant_seed: int | None = None,
    terminal_calibration_hash_variant_seed: int | None = None,
    forged_favorable_eval_seed: int | None = None,
) -> list[dict[str, Any]]:
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    def planned_path(role: str) -> Path:
        raw = Path(plan["artifacts"][role]["path"])
        return (plan_path.parent / raw if not raw.is_absolute() else raw).resolve()

    incumbent_path = str(planned_path("incumbent_checkpoint"))
    incumbent_hash = plan["artifacts"]["incumbent_checkpoint"]["sha256"]
    suite_path = planned_path("promotion_suite")
    calibration_path = planned_path("calibration_fit")
    held_out_path = planned_path("held_out_vocabulary")
    lessons_path = planned_path("reviewed_corpus")
    encoder_manifest_path = planned_path("encoder_manifest")
    suite = load_frozen_suite(suite_path)
    calls: list[dict[str, Any]] = []
    expected_by_item = {
        item["id"]: item["expected"]
        for items in suite.splits.values()
        for item in items
    }
    replay_decisions = dict(decisions)
    if forged_favorable_eval_seed is not None:
        replay_decisions[forged_favorable_eval_seed] = "REJECT"

    class FakeCheckpointPredictor:
        """Test-only replay stub; production always uses CheckpointPredictor."""

        def __init__(self, checkpoint_path: str | Path) -> None:
            self.path = Path(checkpoint_path).resolve()
            self.sha256 = file_sha256(self.path)

        def __call__(self, item: dict[str, Any]) -> Any:
            target = expected_by_item[item["id"]]
            incumbent_prediction = (
                "INCORRECT" if item["split"] in {"fresh", "composition"} else target
            )
            if self.path == Path(incumbent_path).resolve():
                return incumbent_prediction
            seed = next(
                (
                    candidate
                    for candidate in replay_decisions
                    if f"fixture-{candidate}" in self.path.parts
                ),
                None,
            )
            if seed is None:
                raise ValueError(f"unexpected fake checkpoint path: {self.path}")
            return (
                target if replay_decisions[seed] == "QUALIFY" else incumbent_prediction
            )

    def fake_evolve(lessons: str | Path, **kwargs: Any) -> dict[str, Any]:
        seed = kwargs["seed"]
        calls.append({"lessons": Path(lessons), **kwargs})
        assert Path(kwargs["state_dir"]).name == "run-state"
        assert Path(kwargs["output_root"]).name == "candidates"
        assert not Path(kwargs["state_dir"]).exists()
        store = AliveStore(kwargs["state_dir"])
        incumbent_state = store.read()
        expected_incumbent = {
            "path": str(Path(str(incumbent_state["semantic_model"])).resolve()),
            "sha256": incumbent_state["semantic_model_sha256"],
            "generation": incumbent_state["semantic_model_generation"],
        }
        run_id = f"fixture-{seed}"
        run_dir = Path(kwargs["output_root"]) / run_id
        trained = run_dir / "trained"
        calibrated = run_dir / "calibrated"
        trained.mkdir(parents=True)
        calibrated.mkdir()

        required_terms, required_artifacts, required_artifacts_sha256 = (
            required_held_out_vocabulary()
        )
        requested_terms = {
            line.strip().casefold()
            for line in held_out_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        required_terms_sorted = sorted(required_terms)
        full_terms = sorted(required_terms | requested_terms)
        held_out_terms_sha256 = canonical_json_sha256(full_terms)
        required_terms_sha256 = canonical_json_sha256(required_terms_sorted)
        _, exclusion_artifacts = _frozen_evaluation_exclusions()
        if seed == exclusion_variant_seed:
            exclusion_artifacts = [
                *exclusion_artifacts,
                {
                    "path": str(Path(kwargs["output_root"]) / "fake.json"),
                    "sha256": "4" * 64,
                    "role": "FORGED_TRAINING_EXCLUSION",
                },
            ]
        exclusion_hash = canonical_json_sha256(
            sorted(
                (
                    {"role": artifact["role"], "sha256": artifact["sha256"]}
                    for artifact in exclusion_artifacts
                ),
                key=lambda item: (item["role"], item["sha256"]),
            )
        )
        optimizer = {
            "name": "AdamW",
            "learning_rate": 0.0006 if seed == non_seed_variant_seed else 0.0005,
            "weight_decay": 0.0002,
            "gradient_clip_norm": 1.0,
        }
        embedding_tensor = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
        embedding_body = (
            b"kev.frozen-embeddings.v1\0float32-le\0"
            + json.dumps(list(embedding_tensor.shape), separators=(",", ":")).encode(
                "ascii"
            )
            + b"\0"
            + embedding_tensor.numpy().astype("<f4", copy=False).tobytes()
        )
        embedding_values_sha256 = hashlib.sha256(embedding_body).hexdigest()
        embedding_path = trained / "frozen-embeddings.pt"
        torch.save(
            {
                "schema": "kev.frozen-embeddings.v1",
                "encoder_manifest_sha256": file_sha256(encoder_manifest_path),
                "lessons_sha256": file_sha256(lessons_path),
                "shape": list(embedding_tensor.shape),
                "dtype": "float32-le",
                "values_sha256": embedding_values_sha256,
                "embeddings": embedding_tensor,
            },
            embedding_path,
        )
        source_manifest = runner._current_training_source_manifest()
        if seed == source_manifest_variant_seed:
            source_manifest = list(reversed(source_manifest))
        source_manifest_sha256 = canonical_json_sha256(source_manifest)
        source_hashes = {entry["role"]: entry["sha256"] for entry in source_manifest}
        training_inputs = {
            "parent_sha256": incumbent_hash,
            "lessons_sha256": file_sha256(lessons_path),
            "held_out_vocabulary_sha256": held_out_terms_sha256,
            "pinned_held_out_vocabulary_file_sha256": file_sha256(
                PINNED_HELD_OUT_VOCABULARY
            ),
            "required_held_out_vocabulary_artifacts_sha256": (
                required_artifacts_sha256
            ),
            "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
            "training_exclusions_sha256": exclusion_hash,
            "paraphrase_groups_sha256": "2" * 64,
            "encoder_manifest_sha256": file_sha256(encoder_manifest_path),
            "frozen_embedding_values_sha256": embedding_values_sha256,
            "trainer_source_sha256": source_hashes["FROZEN_ENCODER_TRAINER"],
            "encoder_runtime_source_sha256": source_hashes["ENCODER_RUNTIME"],
            "training_contract_source_sha256": source_hashes[
                "SHARED_SEMANTIC_TRAINING"
            ],
            "training_source_manifest": source_manifest,
            "training_source_manifest_sha256": source_manifest_sha256,
            "seed": seed,
            "steps": runner.TRAINING_STEPS,
            "objective": runner.TRAINING_OBJECTIVE,
            "optimizer": optimizer,
            "torch_num_threads": 2,
        }
        training_inputs_sha256 = canonical_json_sha256(training_inputs)
        shared = {
            "model_family": "FROZEN_SENTENCE_ENCODER_PROPOSAL",
            "parent_sha256": incumbent_hash,
            "lessons_sha256": file_sha256(lessons_path),
            "held_out_vocabulary_sha256": held_out_terms_sha256,
            "pinned_held_out_vocabulary_file_sha256": file_sha256(
                PINNED_HELD_OUT_VOCABULARY
            ),
            "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
            "required_held_out_vocabulary_artifacts": required_artifacts,
            "required_held_out_vocabulary_artifacts_sha256": (
                required_artifacts_sha256
            ),
            "training_exclusions_sha256": exclusion_hash,
            "paraphrase_groups_sha256": "2" * 64,
            "contrastive_positive_pairs": 1,
            "trainer_source_sha256": training_inputs["trainer_source_sha256"],
            "encoder_runtime_source_sha256": training_inputs[
                "encoder_runtime_source_sha256"
            ],
            "training_contract_source_sha256": training_inputs[
                "training_contract_source_sha256"
            ],
            "training_source_manifest": source_manifest,
            "training_source_manifest_sha256": source_manifest_sha256,
            "training_inputs_sha256": training_inputs_sha256,
            "training_scope": "projection_frame_and_cardinality_heads_only",
            "seed": seed,
            "steps": runner.TRAINING_STEPS,
            "objective": runner.TRAINING_OBJECTIVE,
            "optimizer": optimizer,
            "torch_num_threads": 2,
            "score_status": "UNCALIBRATED",
            "artifact_status": "CHALLENGER",
        }
        manifest_reference = {
            "path": str(encoder_manifest_path),
            "sha256": file_sha256(encoder_manifest_path),
        }
        raw_checkpoint = trained / "semantic-breadth.pt"
        torch.save(
            {
                "schema": runner.FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
                **shared,
                "encoder_manifest": manifest_reference,
                "frozen_embedding_values_sha256": embedding_values_sha256,
                "temperature": None,
            },
            raw_checkpoint,
        )
        raw_checkpoint_sha256 = file_sha256(raw_checkpoint)
        training_receipt = {
            "schema": "kev.training-receipt.v1",
            "artifact_status": "CHALLENGER",
            "activation": "NONE",
            **shared,
            "parent_path": incumbent_path,
            "lessons_path": str(lessons_path),
            "held_out_vocabulary": full_terms,
            "required_held_out_vocabulary": required_terms_sorted,
            "training_exclusion_artifacts": exclusion_artifacts,
            "encoder_artifact": manifest_reference,
            "frozen_embedding_cache": {
                "path": str(embedding_path),
                "file_sha256": file_sha256(embedding_path),
                "values_sha256": embedding_values_sha256,
                "shape": list(embedding_tensor.shape),
                "encoder_batch_size": 8,
            },
            "training_inputs": training_inputs,
            "challenger_path": str(raw_checkpoint),
            "challenger_sha256": raw_checkpoint_sha256,
            "loss_history": [
                {"step": step, "loss": 1.0}
                for step in range(1, runner.TRAINING_STEPS + 1)
            ],
            "promotion_features": [],
            "environment": {
                "python": plan["runtime_environment"]["python"],
                "torch": plan["runtime_environment"]["packages"]["torch"],
                "transformers": plan["runtime_environment"]["packages"]["transformers"],
                "tokenizers": plan["runtime_environment"]["packages"]["tokenizers"],
                "safetensors": plan["runtime_environment"]["packages"]["safetensors"],
            },
        }
        training_receipt_path = trained / "training-receipt.json"
        _json(training_receipt_path, training_receipt)

        temperature = 1.5
        checkpoint = calibrated / "semantic-breadth.calibrated.pt"
        calibrated_checkpoint = torch.load(
            raw_checkpoint, map_location="cpu", weights_only=True
        )
        calibrated_checkpoint.update(
            {
                "temperature": temperature,
                "score_status": "CALIBRATED",
                "calibration_suite_sha256": file_sha256(calibration_path),
                "calibration_parent_sha256": raw_checkpoint_sha256,
                "promotion_suite_file_sha256": suite.file_sha256,
                "promotion_suite_canonical_sha256": suite.canonical_sha256,
            }
        )
        torch.save(calibrated_checkpoint, checkpoint)
        calibration_receipt = {
            "schema": "kev.calibration-receipt.v1",
            "source_path": str(raw_checkpoint),
            "source_sha256": raw_checkpoint_sha256,
            "calibration_suite": str(calibration_path),
            "calibration_suite_sha256": file_sha256(calibration_path),
            "fit_data_sha256": file_sha256(calibration_path),
            "calibration_fit_split": "calibration_fit",
            "promotion_suite": str(suite_path),
            "promotion_suite_sha256": suite.file_sha256,
            "promotion_suite_file_sha256": suite.file_sha256,
            "promotion_suite_canonical_sha256": suite.canonical_sha256,
            "temperature": temperature,
            "loss_history": [1.0],
            "output_path": str(checkpoint),
            "output_sha256": file_sha256(checkpoint),
            "score_status": "CALIBRATED",
            "activation": "NONE",
        }
        calibration_receipt_path = calibrated / "calibration-receipt.json"
        _json(calibration_receipt_path, calibration_receipt)

        decision = decisions[seed]
        incumbent_predictions: list[dict[str, Any]] = []
        challenger_predictions: list[dict[str, Any]] = []
        for split, items in suite.splits.items():
            for item in items:
                target = item["expected"]
                incumbent_prediction = (
                    "INCORRECT" if split in {"fresh", "composition"} else target
                )
                incumbent_predictions.append(
                    {"item_id": item["id"], "predicted": incumbent_prediction}
                )
                challenger_predictions.append(
                    {
                        "item_id": item["id"],
                        "predicted": target
                        if decision == "QUALIFY"
                        else incumbent_prediction,
                    }
                )
        eval_card = evaluate_challenger(
            suite,
            incumbent_predictor=lambda item: next(
                row["predicted"]
                for row in incumbent_predictions
                if row["item_id"] == item["id"]
            ),
            challenger_predictor=lambda item: next(
                row["predicted"]
                for row in challenger_predictions
                if row["item_id"] == item["id"]
            ),
            incumbent_path=incumbent_path,
            challenger_path=checkpoint,
            incumbent_sha256=incumbent_hash,
            challenger_sha256=file_sha256(checkpoint),
            challenger_temperature=calibration_receipt_path,
            retention_epsilon=runner.RETENTION_EPSILON,
        )
        assert eval_card["decision"]["decision"] == decision
        eval_card_path = run_dir / "eval-card.json"
        _json(eval_card_path, eval_card)
        challenger = {
            "path": str(checkpoint.resolve()),
            "sha256": file_sha256(checkpoint),
            "parent_sha256": incumbent_hash,
            "score_status": "CALIBRATED",
            "eval_card": str(eval_card_path),
            "eval_card_sha256": file_sha256(eval_card_path),
        }
        event_data = {
            "run_id": run_id,
            "incumbent": {
                "path": incumbent_path,
                "sha256": incumbent_hash,
                "generation": expected_incumbent["generation"],
                "source": "state",
            },
            "challenger": challenger,
            "suite_sha256": suite.canonical_sha256,
            "eval_card_path": str(eval_card_path),
            "eval_card_sha256": file_sha256(eval_card_path),
            "decision_sha256": eval_card["decision"]["decision_sha256"],
            "reason_codes": eval_card["decision"]["reason_codes"],
            "training_receipt_path": str(training_receipt_path),
            "training_receipt_sha256": file_sha256(training_receipt_path),
            "calibration_receipt_path": (
                str(run_dir / "forged-calibration-receipt.json")
                if seed == terminal_calibration_path_variant_seed
                else str(calibration_receipt_path)
            ),
            "calibration_receipt_sha256": (
                "0" * 64
                if seed == terminal_calibration_hash_variant_seed
                else file_sha256(calibration_receipt_path)
            ),
            "input_artifacts": {
                "lessons_sha256": file_sha256(lessons_path),
                "held_out_vocabulary_sha256": file_sha256(held_out_path),
                "held_out_vocabulary_terms_sha256": held_out_terms_sha256,
                "required_held_out_vocabulary_terms_sha256": required_terms_sha256,
                "required_held_out_vocabulary_artifacts_sha256": (
                    required_artifacts_sha256
                ),
                "calibration_fit_sha256": file_sha256(calibration_path),
                "promotion_suite_file_sha256": suite.file_sha256,
                "promotion_suite_canonical_sha256": suite.canonical_sha256,
                "encoder_manifest_sha256": file_sha256(encoder_manifest_path),
            },
            "training_loss_used_for_promotion": False,
        }
        event_kind = "MODEL_QUALIFIED" if decision == "QUALIFY" else "MODEL_REJECTED"
        if decision == "QUALIFY":
            ledger_event = store.record_model_decision(
                event_kind,
                event_data,
                active_model={
                    "path": str(checkpoint.resolve()),
                    "sha256": file_sha256(checkpoint),
                },
                expected_incumbent=expected_incumbent,
            )
            local_registry_path = store.dir / "model-registry.json"
            _json(
                local_registry_path,
                {
                    "schema": "kev.model-registry.v1",
                    "active": challenger,
                    "history": [],
                    "last_eval_card_sha256": file_sha256(eval_card_path),
                },
            )
            local_registry: str | None = str(local_registry_path)
        else:
            ledger_event = store.record_model_decision(
                event_kind, event_data, expected_incumbent=expected_incumbent
            )
            local_registry = None

        if seed == failing_seed:
            _json(run_dir / "raw-failure.json", {"seed": seed, "preserved": True})
            raise RuntimeError(f"simulated failure for {seed}")

        result = {
            "schema": "kev.evolution-result.v1",
            "run_id": run_id,
            "decision": decision,
            "incumbent": event_data["incumbent"],
            "challenger": challenger,
            "eval_card": {
                "path": str(eval_card_path),
                "sha256": file_sha256(eval_card_path),
            },
            "ledger_event": ledger_event["hash"],
            "local_registry": local_registry,
            "candidate_preserved": True,
            "old_incumbent_preserved": True,
            "training_loss_used_for_promotion": False,
        }
        result["result_sha256"] = canonical_json_sha256(result)
        _json(run_dir / "evolution-result.json", result)
        return result

    monkeypatch.setattr(runner, "CheckpointPredictor", FakeCheckpointPredictor)
    monkeypatch.setattr(runner, "evolve", fake_evolve)
    return calls


def test_fixed_seed_order_and_primary_only_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, output_root, _ = _plan(tmp_path)
    calls = _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
    )

    result = runner.run_experiment(plan_path)

    assert [call["seed"] for call in calls] == [52031, 52047, 52069]
    assert len({str(call["state_dir"]) for call in calls}) == 3
    assert all(call["steps"] == 800 for call in calls)
    assert all(call["retention_epsilon"] == 0.0 for call in calls)
    assert all(call["registry_path"] == runner.PUBLIC_REGISTRY for call in calls)
    assert all(call["encoder_manifest_path"] for call in calls)
    assert result["outcome"] == "PRIMARY_QUALIFIED_RECOMMENDATION_ELIGIBLE"
    assert result["recommendation_eligible"] is True

    aggregate_path = Path(result["aggregate_path"])
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    assert file_sha256(aggregate_path) == result["aggregate_sha256"]
    integrity = aggregate.pop("integrity")
    assert canonical_json_sha256(aggregate) == integrity["sha256"]
    assert aggregate["training_loss_used_as_aggregate_criterion"] is False
    assert aggregate["decision_authority"] == "PER_SEED_LEDGER_TERMINAL_EVENT"
    assert [row["authoritative_decision"] for row in aggregate["seed_runs"]] == [
        "QUALIFY",
        "REJECT",
        "QUALIFY",
    ]
    assert [
        row["promotion_recommendation_eligible"] for row in aggregate["seed_runs"]
    ] == [
        True,
        False,
        False,
    ]
    evidence_roles = {
        record["role"]
        for row in aggregate["seed_runs"]
        for record in row["evidence_files"]
    }
    assert {
        "ledger",
        "state",
        "eval_card",
        "training_receipt",
        "calibration_receipt",
        "evolution_result",
        "candidate_checkpoint",
    } <= evidence_roles
    for row in aggregate["seed_runs"]:
        for record in row["evidence_files"]:
            assert file_sha256(output_root / record["path"]) == record["sha256"]


def test_robustness_qualification_cannot_replace_rejected_primary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "REJECT", 52047: "QUALIFY", 52069: "QUALIFY"},
    )

    result = runner.run_experiment(plan_path)

    assert result["outcome"] == "PRIMARY_REJECTED_NO_RECOMMENDATION"
    assert result["recommendation_eligible"] is False
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    assert aggregate["promotion_recommendation"] == {
        "eligible": False,
        "seed": None,
        "checkpoint": None,
        "robustness_seeds_can_substitute": False,
    }


def test_later_seed_failure_preserves_aggregate_and_continues(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    calls = _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "REJECT"},
        failing_seed=52047,
    )

    result = runner.run_experiment(plan_path)

    assert [call["seed"] for call in calls] == [52031, 52047, 52069]
    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert result["recommendation_eligible"] is False
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    failed = aggregate["seed_runs"][1]
    assert failed["execution_status"] == "RAISED"
    assert failed["authoritative_decision"] == "REJECT"
    assert failed["terminal_ledger_receipt"]["kind"] == "MODEL_REJECTED"
    assert "raw_failure" in {record["role"] for record in failed["evidence_files"]}


def test_tampered_input_rejects_before_output_or_evolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, output_root, inputs = _plan(tmp_path)
    called = False

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        nonlocal called
        called = True

    monkeypatch.setattr(runner, "evolve", forbidden)
    inputs["corpus"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.run_experiment(plan_path)

    assert called is False
    assert not output_root.exists()


def test_existing_output_root_rejects_before_evolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, output_root, _ = _plan(tmp_path)
    output_root.mkdir()
    monkeypatch.setattr(
        runner,
        "evolve",
        lambda *_args, **_kwargs: pytest.fail("evolve must not be called"),
    )

    with pytest.raises(FileExistsError, match="already exists"):
        runner.run_experiment(plan_path)


def test_repository_plan_closes_dynamic_trainer_input_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = tmp_path / "evals" / "frozen"
    frozen.mkdir(parents=True)
    suite = frozen / "public-audit-v1-260.json"
    calibration = frozen / "calibration-fit-v1.jsonl"
    held_out = frozen / "held-out-vocabulary-v1.txt"
    manifest = frozen / "manifest-v1.json"
    known_failures = frozen / "frame-parser-known-failures-v1.json"
    for path in (suite, calibration, held_out, manifest, known_failures):
        path.write_text(path.name + "\n", encoding="utf-8")
    monkeypatch.setattr(runner, "FROZEN_EVAL_DIRECTORY", frozen.resolve())

    resolved = {
        "promotion_suite": suite.resolve(),
        "calibration_fit": calibration.resolve(),
        "held_out_vocabulary": held_out.resolve(),
    }
    declared = set(resolved.values())
    with pytest.raises(ValueError, match="manifest-v1.json"):
        runner._verify_trainer_discovery_closure(declared)

    declared.update({manifest.resolve(), known_failures.resolve()})
    runner._verify_trainer_discovery_closure(declared)

    late_suite = frozen / "public-audit-v2-260.json"
    late_suite.write_text("late addition\n", encoding="utf-8")
    with pytest.raises(ValueError, match="public-audit-v2-260.json"):
        runner._verify_trainer_discovery_closure(declared)


def test_plan_auto_pins_discovered_inputs_code_and_environment(tmp_path: Path) -> None:
    plan_path, _, _ = _plan(tmp_path)
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    declared = {
        (
            plan_path.parent / Path(record["path"])
            if not Path(record["path"]).is_absolute()
            else Path(record["path"])
        ).resolve()
        for record in plan["artifacts"].values()
    }

    assert runner._discovered_trainer_inputs() <= declared
    assert runner._discovered_executable_sources() <= declared
    assert plan["runtime_environment"] == runner._runtime_environment()


def test_noncanonical_promotion_path_rejects_before_plan_write(tmp_path: Path) -> None:
    inputs = _inputs(tmp_path)
    copied_suite = tmp_path / "copied-suite.json"
    copied_suite.write_bytes(runner.DEFAULT_SUITE.read_bytes())
    plan_path = tmp_path / "invalid-plan.json"

    with pytest.raises(ValueError, match="canonical preregistered artifact"):
        runner.create_plan(
            plan_path=plan_path,
            experiment_id="noncanonical-suite",
            output_root=tmp_path / "output",
            public_registry=runner.PUBLIC_REGISTRY,
            promotion_suite=copied_suite,
            calibration_fit=inputs["calibration"],
            held_out_vocabulary=inputs["held_out"],
            encoder_manifest=inputs["encoder_manifest"],
            reviewed_corpus=inputs["corpus"],
        )

    assert not plan_path.exists()


def test_changed_auto_pinned_code_rejects_before_evolve(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel_source = tmp_path / "experiment-code.py"
    sentinel_source.write_text("VERSION = 1\n", encoding="utf-8")
    original_discovery = runner._discovered_executable_sources
    monkeypatch.setattr(
        runner,
        "_discovered_executable_sources",
        lambda: {*original_discovery(), sentinel_source.resolve()},
    )
    plan_path, output_root, _ = _plan(tmp_path)
    sentinel_source.write_text("VERSION = 2\n", encoding="utf-8")
    monkeypatch.setattr(
        runner,
        "evolve",
        lambda *_args, **_kwargs: pytest.fail("evolve must not be called"),
    )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        runner.run_experiment(plan_path)

    assert not output_root.exists()


def test_arbitrary_replacement_receipt_never_yields_complete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
    )
    valid_evolve = runner.evolve

    def replace_receipt(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = valid_evolve(*args, **kwargs)
        if kwargs["seed"] == runner.PRIMARY_SEED:
            run_dir = Path(kwargs["output_root"]) / result["run_id"]
            _json(
                run_dir / "trained" / "training-receipt.json",
                {"schema": "kev.training-receipt.v1", "seed": runner.PRIMARY_SEED},
            )
        return result

    monkeypatch.setattr(runner, "evolve", replace_receipt)
    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    primary = next(
        row for row in aggregate["seed_runs"] if row["seed"] == runner.PRIMARY_SEED
    )

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert primary["evidence_integrity"] == "INCOMPLETE"
    assert any("training receipt" in error for error in primary["integrity_errors"])


def test_cross_seed_non_seed_training_input_difference_rejects_experiment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
        non_seed_variant_seed=52047,
    )

    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert aggregate["cross_seed_training_input_verification"]["valid"] is False
    assert all(row["content_chain"] is not None for row in aggregate["seed_runs"])
    assert all(
        "non-seed training_inputs evidence is missing or differs across seeds"
        in row["integrity_errors"]
        for row in aggregate["seed_runs"]
    )


def test_forged_training_exclusion_inventory_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
        exclusion_variant_seed=52047,
    )

    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    variant = next(row for row in aggregate["seed_runs"] if row["seed"] == 52047)

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert variant["evidence_integrity"] == "INCOMPLETE"
    assert any(
        "training_exclusion_artifacts" in error for error in variant["integrity_errors"]
    )


def test_reordered_training_source_manifest_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
        source_manifest_variant_seed=52047,
    )

    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    variant = next(row for row in aggregate["seed_runs"] if row["seed"] == 52047)

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert variant["evidence_integrity"] == "INCOMPLETE"
    assert any(
        "training_source_manifest" in error for error in variant["integrity_errors"]
    )


@pytest.mark.parametrize("binding", ["path", "hash"])
def test_forged_terminal_calibration_receipt_binding_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, binding: str
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    variants = (
        {"terminal_calibration_path_variant_seed": runner.PRIMARY_SEED}
        if binding == "path"
        else {"terminal_calibration_hash_variant_seed": runner.PRIMARY_SEED}
    )
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
        **variants,
    )

    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    primary = next(
        row for row in aggregate["seed_runs"] if row["seed"] == runner.PRIMARY_SEED
    )

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert primary["evidence_integrity"] == "INCOMPLETE"
    assert any(
        f"terminal calibration_receipt_{binding}" in error
        or f"terminal calibration receipt {binding}" in error
        for error in primary["integrity_errors"]
    )


def test_final_revalidation_detects_deleted_earlier_seed_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
    )
    valid_evolve = runner.evolve

    def delete_prior_result(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = valid_evolve(*args, **kwargs)
        if kwargs["seed"] == 52047:
            seed_results = Path(kwargs["output_root"]).parent.parent
            prior_result = (
                seed_results
                / f"seed-{runner.PRIMARY_SEED}"
                / "candidates"
                / f"fixture-{runner.PRIMARY_SEED}"
                / "evolution-result.json"
            )
            prior_result.unlink()
        return result

    monkeypatch.setattr(runner, "evolve", delete_prior_result)
    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    primary = next(
        row for row in aggregate["seed_runs"] if row["seed"] == runner.PRIMARY_SEED
    )

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert result["recommendation_eligible"] is False
    assert aggregate["final_seed_revalidation"]["valid"] is False
    assert primary["final_revalidation"]["valid"] is False
    assert primary["evidence_integrity"] == "INCOMPLETE"
    assert "evolution_result" not in {
        record["role"] for record in primary["evidence_files"]
    }
    assert any(
        "final evidence revalidation failed" in error
        for error in primary["integrity_errors"]
    )


def test_final_revalidation_detects_added_earlier_seed_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
    )
    valid_evolve = runner.evolve

    def add_prior_evidence(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = valid_evolve(*args, **kwargs)
        if kwargs["seed"] == 52047:
            seed_results = Path(kwargs["output_root"]).parent.parent
            _json(
                seed_results / f"seed-{runner.PRIMARY_SEED}" / "unexpected.json",
                {"forged": True},
            )
        return result

    monkeypatch.setattr(runner, "evolve", add_prior_evidence)
    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    primary = next(
        row for row in aggregate["seed_runs"] if row["seed"] == runner.PRIMARY_SEED
    )

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert result["recommendation_eligible"] is False
    assert primary["final_revalidation"]["valid"] is False
    assert "unexpected.json" in {
        Path(record["path"]).name for record in primary["evidence_files"]
    }
    assert any("added=" in error for error in primary["integrity_errors"])


def test_mutated_preserved_plan_copy_blocks_recommendation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
    )
    valid_evolve = runner.evolve

    def mutate_preserved_plan(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = valid_evolve(*args, **kwargs)
        if kwargs["seed"] == 52047:
            experiment_root = Path(kwargs["output_root"]).parents[2]
            (experiment_root / "preregistered-plan.json").write_bytes(
                b'{"forged":true}\n'
            )
        return result

    monkeypatch.setattr(runner, "evolve", mutate_preserved_plan)
    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert result["recommendation_eligible"] is False
    assert aggregate["final_input_verification"]["valid"] is False
    assert (
        aggregate["final_input_verification"]["error"]["message"]
        == "preserved preregistered plan copy changed during execution"
    )
    assert (
        aggregate["plan"]["preserved_copy_sha256"] != aggregate["plan"]["source_sha256"]
    )


def test_forged_favorable_eval_card_fails_checkpoint_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path, _, _ = _plan(tmp_path)
    _install_fake_evolve(
        monkeypatch,
        plan_path,
        decisions={52031: "QUALIFY", 52047: "REJECT", 52069: "QUALIFY"},
        forged_favorable_eval_seed=runner.PRIMARY_SEED,
    )

    result = runner.run_experiment(plan_path)
    aggregate = json.loads(Path(result["aggregate_path"]).read_text(encoding="utf-8"))
    primary = next(
        row for row in aggregate["seed_runs"] if row["seed"] == runner.PRIMARY_SEED
    )

    assert result["outcome"] == "EXPERIMENT_INCOMPLETE_NO_RECOMMENDATION"
    assert result["recommendation_eligible"] is False
    assert primary["evidence_integrity"] == "INCOMPLETE"
    assert any(
        "independent checkpoint replay" in error
        for error in primary["integrity_errors"]
    )

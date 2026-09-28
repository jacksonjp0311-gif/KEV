from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
import torch

from kev import cli as cli_module
from kev import model_runtime as model_runtime_module
from kev.artifacts import (
    canonical_json_sha256,
    file_sha256,
    json_artifact_hashes,
    verify_file_sha256,
)
from kev.evaluation import (
    evaluate_challenger,
    load_frozen_suite,
    replay_historical_aggregate,
)
from kev.model_runtime import CheckpointPredictor
from kev.uc51a2.semantic_breadth import D, _h, features
from kev.uc51a3 import alive as alive_module
from kev.uc51a3.alive import AliveRuntime, AliveStore


ROOT = Path(__file__).resolve().parents[1]


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _copy_current_evidence_chain(target: Path) -> tuple[dict, dict]:
    registry = _json("models/registry.json")
    evidence = _json(registry["active"]["evidence_manifest_path"])
    relatives = {
        "models/registry.json",
        registry["active"]["path"],
        registry["active"]["evidence_manifest_path"],
        registry["active"]["eval_card_path"],
        evidence["eval_manifest_path"],
        evidence["suite_path"],
        evidence["known_failures_path"],
        evidence["previous_evidence_manifest_path"],
    }
    for relative in relatives:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
    return registry, evidence


def _verify_copied_chain(target: Path) -> dict:
    return cli_module._verify_public_evidence_chain(
        root=target,
        registry_path=target / "models/registry.json",
        expected_manifest_path=target / "evals/frozen/manifest-v6.json",
        expected_suite_path=target / "evals/frozen/public-audit-v6-260.json",
    )


def test_published_model_registry_and_evaluation_manifest_match_current_bytes():
    registry = _json("models/registry.json")
    active = registry["active"]
    checkpoint = ROOT / active["path"]
    assert checkpoint.stat().st_size == 1_144_120
    assert file_sha256(checkpoint) == active["sha256"]

    model_manifest = _json("models/public/incumbent-manifest.json")
    for field in ("path", "sha256", "parameters", "parent_sha256", "artifact_status"):
        assert model_manifest[field] == active[field]

    evidence = _json(active["evidence_manifest_path"])
    evidence_path = ROOT / active["evidence_manifest_path"]
    assert file_sha256(evidence_path) == active["evidence_manifest_sha256"]
    assert (
        file_sha256(ROOT / evidence["checkpoint_path"]) == evidence["checkpoint_sha256"]
    )
    assert (
        file_sha256(ROOT / evidence["eval_card_path"]) == evidence["eval_card_sha256"]
    )
    assert file_sha256(ROOT / evidence["suite_path"]) == evidence["suite_sha256"]
    assert (
        file_sha256(ROOT / evidence["eval_manifest_path"])
        == evidence["eval_manifest_sha256"]
    )
    assert (
        file_sha256(ROOT / evidence["known_failures_path"])
        == evidence["known_failures_sha256"]
    )
    assert evidence["known_failures_promotion_eligible"] is False
    assert (
        file_sha256(ROOT / evidence["previous_evidence_manifest_path"])
        == evidence["previous_evidence_manifest_sha256"]
    )
    assert (
        file_sha256(ROOT / evidence["calibration_fit_path"])
        == evidence["calibration_fit_sha256"]
    )
    assert (
        file_sha256(ROOT / evidence["held_out_vocabulary_path"])
        == evidence["held_out_vocabulary_sha256"]
    )
    assert (
        file_sha256(ROOT / evidence["freshness_audit_finding_path"])
        == evidence["freshness_audit_finding_sha256"]
    )
    assert evidence["audit_family_gate_count"] == 28
    assert (ROOT / "models/public/incumbent-evidence-v1.json").is_file()
    eval_card = _json(evidence["eval_card_path"])
    claimed_report_hash = eval_card.pop("report_sha256")
    assert canonical_json_sha256(eval_card) == claimed_report_hash
    assert claimed_report_hash == evidence["report_sha256"]
    assert eval_card["decision"]["policy"]["audit_family_count"] == 28
    audit_gates = [
        gate
        for gate in eval_card["decision"]["gates"]
        if gate["name"].startswith("audit:")
    ]
    assert len(audit_gates) == 28
    assert all(gate["passed"] for gate in audit_gates)
    assert eval_card["incumbent"]["audit_metrics"]["order-swap"] == {
        "accuracy": 1.0,
        "correct": 8,
        "failures": 0,
        "total": 8,
    }
    assert all(
        "audit" in failure
        for failure in eval_card["incumbent"]["raw_failures"]["items"]
    )

    frozen_manifest = _json("evals/frozen/manifest-v6.json")
    assert frozen_manifest["frozen"] is True
    for artifact in frozen_manifest["artifacts"].values():
        path = ROOT / artifact["path"]
        assert path.stat().st_size == artifact["size_bytes"]
        assert file_sha256(path) == artifact["sha256"]
        if "canonical_sha256" in artifact:
            assert (
                json_artifact_hashes(path)["canonical_json_sha256"]
                == artifact["canonical_sha256"]
            )


def test_doctor_verifies_the_registry_rooted_evidence_chain(capsys):
    assert cli_module.command_doctor(argparse.Namespace()) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["ok"] is True
    incumbent = report["checks"][0]
    assert incumbent["evidence_manifest_sha256"] == (
        "69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21"
    )
    assert incumbent["eval_manifest_sha256"] == (
        "07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce"
    )
    assert incumbent["eval_report_sha256"] == (
        "e6c768c496048afbdcf56554700ac2cc19e9c09d013139e2c7c53cdb3a9f658e"
    )
    assert incumbent["known_failures_sha256"] == (
        "1aa0d4b519bfdd96e1874f25a0f814149a3b48f51dc8e38274f3847a46a6a901"
    )


def test_doctor_rejects_a_mutable_manifest_not_pinned_by_active_evidence(
    tmp_path: Path,
):
    _copy_current_evidence_chain(tmp_path)
    manifest = tmp_path / "evals/frozen/manifest-v6.json"
    manifest.write_bytes(manifest.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="evaluation manifest SHA-256 mismatch"):
        _verify_copied_chain(tmp_path)


@pytest.mark.parametrize(
    ("evidence_field", "message"),
    [
        ("known_failures_path", "known failures SHA-256 mismatch"),
        ("previous_evidence_manifest_path", "artifact SHA-256 mismatch"),
    ],
)
def test_doctor_rejects_tampered_direct_evidence_links(
    tmp_path: Path,
    evidence_field: str,
    message: str,
):
    _, evidence = _copy_current_evidence_chain(tmp_path)
    target = tmp_path / evidence[evidence_field]
    target.write_bytes(target.read_bytes() + b"\n")

    with pytest.raises(ValueError, match=message):
        _verify_copied_chain(tmp_path)


def test_doctor_checks_eval_report_content_after_file_hashes_match(tmp_path: Path):
    registry, evidence = _copy_current_evidence_chain(tmp_path)
    card_path = tmp_path / evidence["eval_card_path"]
    card = json.loads(card_path.read_text(encoding="utf-8"))
    card["schema"] = "tampered-but-rehashed"
    _write_json(card_path, card)
    new_card_hash = file_sha256(card_path)

    evidence["eval_card_sha256"] = new_card_hash
    evidence_path = tmp_path / registry["active"]["evidence_manifest_path"]
    _write_json(evidence_path, evidence)
    registry["active"]["eval_card_sha256"] = new_card_hash
    registry["active"]["evidence_manifest_sha256"] = file_sha256(evidence_path)
    _write_json(tmp_path / "models/registry.json", registry)

    with pytest.raises(ValueError, match="canonical content"):
        _verify_copied_chain(tmp_path)


def test_runtime_refuses_a_checkpoint_when_configured_hash_does_not_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = _json("models/registry.json")
    active = registry["active"]
    original = ROOT / active["path"]
    altered = tmp_path / "altered.pt"
    altered.write_bytes(original.read_bytes() + b"tampered")

    store = AliveStore(tmp_path / "state")
    state = store.read()
    state["semantic_model"] = str(altered)
    state["semantic_model_sha256"] = active["sha256"]
    store.write(state)

    load_attempts: list[bytes] = []

    def forbidden_load(checkpoint_bytes: bytes):
        load_attempts.append(checkpoint_bytes)
        raise AssertionError("a mismatched checkpoint must not be deserialized")

    monkeypatch.setattr(alive_module, "load_semantic_bytes", forbidden_load)
    runtime = AliveRuntime(tmp_path / "state")
    assert runtime.model is None
    assert runtime.model_error == "MODEL_HASH_MISMATCH"
    assert load_attempts == []

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_file_sha256(altered, active["sha256"])


def test_checkpoint_predictor_hashes_and_loads_the_same_captured_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    registry = _json("models/registry.json")
    source = ROOT / registry["active"]["path"]
    checkpoint = tmp_path / "checkpoint.pt"
    original_bytes = source.read_bytes()
    checkpoint.write_bytes(original_bytes)
    real_load_bytes = model_runtime_module.load_bytes
    loaded_hashes: list[str] = []

    def mutate_path_then_load(captured: bytes):
        loaded_hashes.append(hashlib.sha256(captured).hexdigest())
        checkpoint.write_bytes(b"different bytes after the single captured read")
        return real_load_bytes(captured)

    monkeypatch.setattr(model_runtime_module, "load_bytes", mutate_path_then_load)
    predictor = model_runtime_module.CheckpointPredictor(checkpoint)

    expected = hashlib.sha256(original_bytes).hexdigest()
    assert predictor.sha256 == expected
    assert loaded_hashes == [expected]
    assert file_sha256(checkpoint) != predictor.sha256


def test_feature_hashing_is_sha256_based_and_vectorization_is_exact():
    text = "goal goal latency 73"
    canonical = ["GOAL", "GOAL", "latency", "NUMBER"]
    feature_names = [*("u:" + token for token in canonical)]
    feature_names.extend(
        "b:" + left + "_" + right for left, right in zip(canonical, canonical[1:])
    )

    for value in feature_names:
        independent = (
            int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big")
            % D
        )
        assert _h(value) == independent

    expected_counts = Counter(_h(value) for value in feature_names)
    expected = torch.zeros(D)
    for index, count in expected_counts.items():
        expected[index] = count
    expected = torch.log1p(expected)

    assert torch.equal(features(text), expected)
    assert int(torch.count_nonzero(features(text))) == len(expected_counts)


def test_historical_tie_replay_matches_the_frozen_published_artifact():
    path = ROOT / "evals/frozen/historical-190-260-replay.json"
    published = json.loads(path.read_text(encoding="utf-8"))
    replayed = replay_historical_aggregate(
        parent_correct=190,
        challenger_correct=190,
        total=260,
    )

    assert replayed == published
    assert replayed["decision"] == "REJECT"
    assert replayed["promote"] is False
    assert replayed["reason_codes"] == ["FRESH_TIE"]
    assert replayed["raw_failures"]["available"] is False
    assert replayed["checkpoint_hashes"]["status"] == (
        "NOT_PRESENT_IN_PUBLISHED_AGGREGATE"
    )

    manifest_record = _json("evals/frozen/manifest-v2.json")["artifacts"][
        "historical_aggregate_replay"
    ]
    assert file_sha256(path) == manifest_record["sha256"]
    assert path.stat().st_size == manifest_record["size_bytes"]


def test_current_public_baseline_replays_byte_exact():
    registry = _json("models/registry.json")
    active = registry["active"]
    checkpoint = ROOT / active["path"]
    evidence = _json(active["evidence_manifest_path"])
    suite = load_frozen_suite(ROOT / evidence["suite_path"])
    predictor = CheckpointPredictor(checkpoint)
    report = evaluate_challenger(
        suite,
        incumbent_predictor=predictor,
        challenger_predictor=predictor,
        incumbent_path=checkpoint,
        challenger_path=checkpoint,
        incumbent_sha256=active["sha256"],
        challenger_sha256=active["sha256"],
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == active["eval_card_sha256"]
    assert report == _json(active["eval_card_path"])

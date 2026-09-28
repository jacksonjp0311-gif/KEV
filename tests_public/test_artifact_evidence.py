from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import tempfile
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
        expected_manifest_path=target
        / cli_module.DEFAULT_EVAL_MANIFEST.relative_to(ROOT),
        expected_suite_path=target / cli_module.DEFAULT_SUITE.relative_to(ROOT),
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
    suite = _json(evidence["suite_path"])
    audit_names = {
        item["audit"]
        for rows in suite["splits"].values()
        for item in rows
        if item.get("audit") is not None
    }
    assert evidence["audit_family_gate_count"] == len(audit_names)
    assert (ROOT / "models/public/incumbent-evidence-v1.json").is_file()
    eval_card = _json(evidence["eval_card_path"])
    claimed_report_hash = eval_card.pop("report_sha256")
    assert canonical_json_sha256(eval_card) == claimed_report_hash
    assert claimed_report_hash == evidence["report_sha256"]
    assert eval_card["decision"]["policy"]["audit_family_count"] == len(audit_names)
    audit_gates = [
        gate
        for gate in eval_card["decision"]["gates"]
        if gate["name"].startswith("audit:")
    ]
    assert {gate["name"] for gate in audit_gates} == {
        f"audit:{name}" for name in audit_names
    }
    assert all(gate["passed"] for gate in audit_gates)
    for name in audit_names:
        expected_total = sum(
            item.get("audit") == name
            for rows in suite["splits"].values()
            for item in rows
        )
        assert eval_card["incumbent"]["audit_metrics"][name]["total"] == expected_total
    assert all(
        "audit" in failure
        for failure in eval_card["incumbent"]["raw_failures"]["items"]
    )

    frozen_manifest = _json(evidence["eval_manifest_path"])
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
    registry = _json("models/registry.json")
    evidence = _json(registry["active"]["evidence_manifest_path"])
    assert (
        incumbent["evidence_manifest_sha256"]
        == registry["active"]["evidence_manifest_sha256"]
    )
    assert incumbent["eval_manifest_sha256"] == evidence["eval_manifest_sha256"]
    assert incumbent["eval_report_sha256"] == evidence["report_sha256"]
    assert incumbent["known_failures_sha256"] == (
        "1aa0d4b519bfdd96e1874f25a0f814149a3b48f51dc8e38274f3847a46a6a901"
    )


def test_doctor_rejects_a_mutable_manifest_not_pinned_by_active_evidence(
    tmp_path: Path,
):
    _copy_current_evidence_chain(tmp_path)
    manifest = tmp_path / cli_module.DEFAULT_EVAL_MANIFEST.relative_to(ROOT)
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


def _replay_leaf_differences(expected: object, actual: object) -> dict:
    missing = object()
    categories: Counter[str] = Counter()
    types: Counter[str] = Counter()
    first: list[dict] = []
    total = 0
    maximum_numeric_delta: float | None = None

    def compare(left: object, right: object, path: str) -> None:
        nonlocal total, maximum_numeric_delta
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(left.keys() | right.keys()):
                segment = str(key).replace("~", "~0").replace("/", "~1")
                compare(
                    left.get(key, missing), right.get(key, missing), f"{path}/{segment}"
                )
            return
        if isinstance(left, list) and isinstance(right, list):
            for index in range(max(len(left), len(right))):
                compare(
                    left[index] if index < len(left) else missing,
                    right[index] if index < len(right) else missing,
                    f"{path}/{index}",
                )
            return
        if type(left) is type(right) and left == right:
            return
        numeric = (
            isinstance(left, (int, float))
            and not isinstance(left, bool)
            and isinstance(right, (int, float))
            and not isinstance(right, bool)
        )
        if left is missing or right is missing:
            category = "missing_expected" if left is missing else "missing_actual"
        elif type(left) is not type(right):
            category = "type_mismatch"
        elif numeric:
            category = "numeric_value"
        else:
            category = "non_numeric_value"
        if numeric:
            delta = abs(float(left) - float(right))
            maximum_numeric_delta = max(maximum_numeric_delta or 0.0, delta)
        total += 1
        categories[category] += 1
        left_type = "MISSING" if left is missing else type(left).__name__
        right_type = "MISSING" if right is missing else type(right).__name__
        types[f"{left_type}->{right_type}"] += 1
        if len(first) < 20:
            first.append(
                {
                    "path": path or "/",
                    "category": category,
                    "expected": "<MISSING>" if left is missing else left,
                    "actual": "<MISSING>" if right is missing else right,
                }
            )

    compare(expected, actual, "")
    return {
        "total_leaf_differences": total,
        "category_counts": dict(sorted(categories.items())),
        "type_counts": dict(sorted(types.items())),
        "max_numeric_delta": maximum_numeric_delta,
        "first_20_differences": first,
    }


def _replay_environment() -> dict:
    capability = getattr(
        getattr(torch.backends, "cpu", None), "get_cpu_capability", None
    )
    cpu_capability = capability() if callable(capability) else "UNAVAILABLE"
    return {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "torch_version": str(torch.__version__),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "torch_cpu_capability": cpu_capability,
        "github_actions": {
            key: os.environ[key]
            for key in (
                "GITHUB_SHA",
                "GITHUB_RUN_ID",
                "GITHUB_RUN_ATTEMPT",
                "GITHUB_JOB",
                "GITHUB_WORKFLOW",
                "RUNNER_OS",
                "RUNNER_ARCH",
            )
            if key in os.environ
        },
    }


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
    actual_sha256 = hashlib.sha256(encoded).hexdigest()
    expected_report = _json(active["eval_card_path"])
    if actual_sha256 != active["eval_card_sha256"]:
        diagnostic = _replay_leaf_differences(expected_report, report)
        diagnostic.update(
            expected_sha256=active["eval_card_sha256"],
            actual_sha256=actual_sha256,
            execution_environment=_replay_environment(),
        )
        output_directory = os.environ.get("KEV_REPLAY_DIAGNOSTICS")
        if output_directory:
            try:
                destination = Path(output_directory)
                destination.mkdir(parents=True, exist_ok=True)
                descriptor, artifact_path = tempfile.mkstemp(
                    prefix=f"baseline-actual-{actual_sha256}-",
                    suffix=".json",
                    dir=destination,
                )
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(encoded)
                diagnostic["actual_report_path"] = artifact_path
                sidecar_path = Path(artifact_path).with_suffix(".diagnostic.json")
                diagnostic["diagnostic_path"] = str(sidecar_path)
                with sidecar_path.open("xb") as handle:
                    handle.write(
                        (
                            json.dumps(diagnostic, indent=2, sort_keys=True) + "\n"
                        ).encode()
                    )
            except OSError as error:
                diagnostic["actual_report_preservation_error"] = str(error)
        print(
            "BASELINE_REPLAY_MISMATCH "
            + json.dumps(diagnostic, indent=2, sort_keys=True)
        )
    assert actual_sha256 == active["eval_card_sha256"]
    assert report == expected_report

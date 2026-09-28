from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from kev.artifacts import canonical_json_sha256
from scripts import audit_composition_v7_inputs as audit
from scripts import record_composition_v7_baseline as baseline


def _put(path: Path, value: object) -> str:
    payload = json.dumps(value, sort_keys=True).encode()
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _baseline_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(baseline, "ROOT", tmp_path)
    for name in ("REGISTRY", "SUITE", "MANIFEST", "BASELINE", "EVIDENCE", "SNAPSHOT"):
        monkeypatch.setattr(baseline, name, tmp_path / (name.lower() + ".json"))
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-only checkpoint; predictor is mocked")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    monkeypatch.setattr(baseline, "GENESIS", digest)
    previous = tmp_path / "previous.json"
    previous_hash = _put(previous, {"schema": "kev.incumbent-evidence.v1"})
    _put(
        baseline.REGISTRY,
        {
            "active": {
                "path": "checkpoint.pt",
                "sha256": digest,
                "evidence_manifest_path": "previous.json",
                "evidence_manifest_sha256": previous_hash,
            }
        },
    )
    suite = {
        "schema": "kev.eval-suite.v1",
        "id": "test-only",
        "frozen": True,
        "splits": {
            split: [{"id": split, "input": split, "expected": ["GOAL"]}]
            for split in ("fresh", "retention", "oov")
        },
    }
    suite_hash = _put(baseline.SUITE, suite)
    manifest_hash = _put(
        baseline.MANIFEST,
        {
            "artifacts": {
                "promotion_suite": {
                    "path": "suite.json",
                    "sha256": suite_hash,
                    "canonical_sha256": canonical_json_sha256(suite),
                }
            }
        },
    )
    monkeypatch.setattr(baseline, "MANIFEST_SHA256", manifest_hash)
    monkeypatch.setattr(
        baseline, "CheckpointPredictor", lambda _: SimpleNamespace(sha256=digest)
    )
    sources = {"kev/test_only.py": "before"}
    monkeypatch.setattr(baseline, "_source_hashes", lambda: dict(sources))
    return checkpoint, sources


@pytest.mark.parametrize("change", ["source", "suite", "checkpoint"])
def test_baseline_rejects_changed_measured_identity_before_any_frozen_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    change: str,
):
    checkpoint, sources = _baseline_fixture(tmp_path, monkeypatch)

    def simulated_evaluation(*args, **kwargs):
        if change == "source":
            sources["kev/test_only.py"] = "after"
        elif change == "suite":
            baseline.SUITE.write_bytes(b"changed suite bytes")
        else:
            checkpoint.write_bytes(b"changed checkpoint bytes")
        return {}

    monkeypatch.setattr(baseline, "evaluate_challenger", simulated_evaluation)
    with pytest.raises(ValueError, match="changed during measurement"):
        baseline.main()
    assert all(
        not path.exists()
        for path in (baseline.BASELINE, baseline.EVIDENCE, baseline.SNAPSHOT)
    )


def test_baseline_rejects_predictor_for_different_captured_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    _baseline_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        baseline, "CheckpointPredictor", lambda _: SimpleNamespace(sha256="0" * 64)
    )
    with pytest.raises(ValueError, match="loaded predictor differs"):
        baseline.main()
    assert not baseline.BASELINE.exists()


def test_input_audit_rejects_corpus_replacement_during_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    corpus = tmp_path / "lessons.jsonl"
    output = tmp_path / "audit.json"
    row = {
        "id": "test-only",
        "text": "A unique test-only reviewed surface.",
        "frames": [{"kind": "GOAL"}],
    }
    _put(corpus, row)

    def replace_after_capture(text, **kwargs):
        assert json.loads(text) == row
        corpus.write_bytes(b"different bytes after captured read")
        return []

    monkeypatch.setattr(audit, "_reviewed_rows_from_text", replace_after_capture)
    monkeypatch.setattr(audit, "_paraphrase_group_manifest", lambda _: [])
    with pytest.raises(ValueError, match="audit input changed during validation"):
        audit.main(["--corpus", str(corpus), "--output", str(output)])
    assert not output.exists()


def test_input_audit_keeps_existing_output_before_loading_inputs(tmp_path: Path):
    output = tmp_path / "existing.json"
    output.write_bytes(b"preserved rejected evidence")
    with pytest.raises(SystemExit, match="refusing to refresh"):
        audit.main(
            ["--corpus", str(tmp_path / "absent.jsonl"), "--output", str(output)]
        )
    assert output.read_bytes() == b"preserved rejected evidence"

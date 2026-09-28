import json
import tempfile
from pathlib import Path

import pytest

from kev.artifacts import file_sha256
from kev.uc51a3.alive import AliveStore, IncumbentCompareAndSwapError


def test_fact_revision_and_ledger_without_model_weights():
    with tempfile.TemporaryDirectory() as d:
        store = AliveStore(Path(d))
        first = store.remember_fact("project codename", "Lumen")
        second = store.remember_fact("project codename", "Nova", correction=True)
        assert first["revision"] == 1
        assert second["revision"] == 2
        assert second["supersedes"] == "Lumen"
        assert second["supersedes_revision"] == 1
        assert store.verify_ledger()["valid"] is True


def _incumbent_snapshot(state):
    return {
        "path": state["semantic_model"],
        "sha256": state["semantic_model_sha256"],
        "generation": state["semantic_model_generation"],
    }


def test_qualified_decision_requires_complete_incumbent_snapshot(tmp_path: Path):
    store = AliveStore(tmp_path / "state")
    candidate = tmp_path / "candidate.pt"
    candidate.write_bytes(b"candidate")

    with pytest.raises(ValueError, match="expected incumbent snapshot"):
        store.record_model_decision(
            "MODEL_QUALIFIED",
            {"run_id": "missing-cas"},
            active_model={"path": str(candidate), "sha256": file_sha256(candidate)},
        )

    incomplete = _incumbent_snapshot(store.read())
    incomplete.pop("generation")
    with pytest.raises(ValueError, match="path, SHA-256, and generation"):
        store.record_model_decision(
            "MODEL_QUALIFIED",
            {"run_id": "incomplete-cas"},
            active_model={"path": str(candidate), "sha256": file_sha256(candidate)},
            expected_incumbent=incomplete,
        )
    assert not store.ledger_path.exists()


def test_qualification_cas_rejects_stale_incumbent_without_mutation(
    tmp_path: Path,
):
    store = AliveStore(tmp_path / "state")
    original = _incumbent_snapshot(store.read())
    winner = tmp_path / "winner.pt"
    stale = tmp_path / "stale.pt"
    winner.write_bytes(b"winner")
    stale.write_bytes(b"stale")
    store.record_model_decision(
        "MODEL_QUALIFIED",
        {"run_id": "winner"},
        active_model={"path": str(winner), "sha256": file_sha256(winner)},
        expected_incumbent=original,
    )
    won = store.read()

    with pytest.raises(IncumbentCompareAndSwapError) as caught:
        store.record_model_decision(
            "MODEL_QUALIFIED",
            {"run_id": "stale"},
            active_model={"path": str(stale), "sha256": file_sha256(stale)},
            expected_incumbent=original,
        )

    assert caught.value.expected == original
    assert caught.value.current["path"] == str(winner.resolve())
    assert store.read() == won
    events = [
        json.loads(line)
        for line in store.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["data"]["run_id"] for event in events] == ["winner"]
    assert store.verify_ledger()["valid"] is True


def test_qualification_cas_generation_rejects_aba(tmp_path: Path):
    store = AliveStore(tmp_path / "state")
    original = _incumbent_snapshot(store.read())
    winner = tmp_path / "winner.pt"
    stale = tmp_path / "stale.pt"
    winner.write_bytes(b"winner")
    stale.write_bytes(b"stale")

    store.record_model_decision(
        "MODEL_QUALIFIED",
        {"run_id": "a-to-b"},
        active_model={"path": str(winner), "sha256": file_sha256(winner)},
        expected_incumbent=original,
    )
    second = _incumbent_snapshot(store.read())
    store.record_model_decision(
        "MODEL_QUALIFIED",
        {"run_id": "b-to-a"},
        active_model={"path": original["path"], "sha256": original["sha256"]},
        expected_incumbent=second,
    )
    aba_state = store.read()
    assert aba_state["semantic_model"] == original["path"]
    assert aba_state["semantic_model_sha256"] == original["sha256"]
    assert aba_state["semantic_model_generation"] == original["generation"] + 2

    with pytest.raises(IncumbentCompareAndSwapError):
        store.record_model_decision(
            "MODEL_QUALIFIED",
            {"run_id": "stale-after-aba"},
            active_model={"path": str(stale), "sha256": file_sha256(stale)},
            expected_incumbent=original,
        )

    assert store.read() == aba_state
    events = [
        json.loads(line)
        for line in store.ledger_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["data"]["run_id"] for event in events] == ["a-to-b", "b-to-a"]
    assert store.verify_ledger()["valid"] is True

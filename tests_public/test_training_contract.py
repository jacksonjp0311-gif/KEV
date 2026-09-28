from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest
import torch

from kev.evolution import DEFAULT_REGISTRY
from kev.uc51a2 import semantic_breadth
from kev.uc51a2.semantic_breadth import (
    PINNED_HELD_OUT_VOCABULARY_FILE_SHA256,
    _canonical_json_sha256,
    _reviewed_rows,
    _targets,
    sha256_file,
    supervised_contrastive_loss,
    train,
)


ROOT = Path(__file__).resolve().parents[1]


def _parent() -> Path:
    registry = json.loads(DEFAULT_REGISTRY.read_text(encoding="utf-8"))
    return ROOT / registry["active"]["path"]


def _lesson(**updates):
    row = {
        "id": "lesson-contract",
        "status": "REVIEWED",
        "reviewed_by": "human-reviewer",
        "reviewed_at": "2026-09-28T00:00:00Z",
        "permission": "human-authored and approved for training",
        "text": "Aim to reduce sandbox latency",
        "frame_kinds": ["GOAL"],
    }
    row.update(updates)
    return row


def test_training_always_rejects_pinned_held_out_surface_forms_before_writing_checkpoint(
    tmp_path: Path,
):
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(
        json.dumps(_lesson(text="Improve celerity")) + "\n", encoding="utf-8"
    )
    output = tmp_path / "candidate"
    with pytest.raises(ValueError, match="held-out vocabulary"):
        train(_parent(), output, steps=1, extra_jsonl=lessons)
    assert not (output / "semantic-breadth.pt").exists()


def test_direct_training_always_rejects_v2_held_out_surface_forms(
    tmp_path: Path,
):
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(
        json.dumps(_lesson(text="Keep the result attested")) + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "candidate"

    with pytest.raises(ValueError, match="held-out vocabulary: attested"):
        train(_parent(), output, steps=1, extra_jsonl=lessons)

    assert not (output / "semantic-breadth.pt").exists()
    assert not (output / "training-receipt.json").exists()


def test_training_refuses_a_modified_pinned_held_out_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    replacement = tmp_path / "held-out-vocabulary-v1.txt"
    replacement.write_text("different\n", encoding="utf-8")
    monkeypatch.setattr(semantic_breadth, "PINNED_HELD_OUT_VOCABULARY", replacement)
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(json.dumps(_lesson()) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="digest does not match"):
        train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)

    assert not (tmp_path / "candidate" / "semantic-breadth.pt").exists()


def test_direct_training_rejects_any_frozen_eval_surface(tmp_path: Path):
    suite = json.loads(
        (ROOT / "evals/frozen/public-audit-v1-260.json").read_text(encoding="utf-8")
    )
    frozen_text = suite["splits"]["retention"][0]["input"]
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(json.dumps(_lesson(text=frozen_text)) + "\n", encoding="utf-8")
    output = tmp_path / "candidate"

    with pytest.raises(ValueError, match="overlaps a frozen evaluation"):
        train(_parent(), output, steps=1, extra_jsonl=lessons)
    assert not (output / "semantic-breadth.pt").exists()


def test_direct_training_rejects_development_influenced_failure_probe(tmp_path: Path):
    audit = json.loads(
        (ROOT / "evals/frozen/frame-parser-known-failures-v1.json").read_text(
            encoding="utf-8"
        )
    )
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(
        json.dumps(_lesson(text=audit["cases"][0]["input"])) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps a frozen evaluation"):
        train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)


@pytest.mark.parametrize("mutation", ["delete", "add"])
def test_training_exclusion_inventory_is_closed_by_pinned_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    copied_root = tmp_path / "repository"
    copied_frozen = copied_root / "evals" / "frozen"
    shutil.copytree(ROOT / "evals" / "frozen", copied_frozen)
    manifest = copied_frozen / "manifest-v6.json"
    held_out = copied_frozen / "held-out-vocabulary-v1.txt"
    monkeypatch.setattr(semantic_breadth, "REPOSITORY_ROOT", copied_root)
    monkeypatch.setattr(semantic_breadth, "FROZEN_EVAL_DIRECTORY", copied_frozen)
    monkeypatch.setattr(semantic_breadth, "PINNED_EVALUATION_MANIFEST", manifest)
    monkeypatch.setattr(
        semantic_breadth,
        "PINNED_EVALUATION_MANIFEST_FILE_SHA256",
        sha256_file(manifest),
    )
    monkeypatch.setattr(semantic_breadth, "PINNED_HELD_OUT_VOCABULARY", held_out)

    if mutation == "delete":
        (copied_frozen / "public-audit-v5-260.json").unlink()
        match = "missing"
    else:
        (copied_frozen / "public-audit-v7-260.json").write_text(
            "{}\n", encoding="utf-8"
        )
        match = "unpinned"

    with pytest.raises(ValueError, match=match):
        semantic_breadth._frozen_evaluation_exclusions()


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"status": "DRAFT"}, "not explicitly REVIEWED"),
        ({"reviewed_by": None}, "lacks review provenance"),
        ({"reviewed_by": "   "}, "lacks review provenance"),
        ({"reviewed_at": "   "}, "lacks review provenance"),
        ({"reviewed_at": "not-a-timestamp"}, "valid ISO-8601 UTC"),
        (
            {"reviewed_at": "2026-09-28T00:00:00-04:00"},
            "valid ISO-8601 UTC",
        ),
        ({"permission": None}, "lacks permission provenance"),
        ({"permission": "   "}, "lacks permission provenance"),
    ],
)
def test_training_rejects_lessons_without_review_permission_provenance(
    tmp_path: Path, updates: dict, match: str
):
    lessons = tmp_path / "lessons.jsonl"
    lessons.write_text(json.dumps(_lesson(**updates)) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)


def _threshold_frame(value: int = 50) -> dict:
    return {
        "kind": "GOAL",
        "relation": "THRESHOLD",
        "slots": {
            "operator": {"type": "operator", "value": "LT"},
            "subject": {"type": "entity", "value": "latency"},
            "unit": {"type": "unit", "value": "ms"},
            "value": {"type": "number", "value": value},
        },
    }


def _policy_frame(polarity: bool) -> dict:
    return {
        "kind": "CONSTRAINT",
        "relation": "ACTION_POLICY",
        "slots": {
            "action": {"type": "action", "value": "modify"},
            "object": {"type": "entity", "value": "production"},
            "polarity": {"type": "boolean", "value": polarity},
        },
    }


def test_repeated_frame_kinds_count_toward_cardinality(tmp_path: Path):
    lessons = tmp_path / "repeated.jsonl"
    row = _lesson(
        frame_kinds=None,
        frames=[
            {
                "kind": "CONSTRAINT",
                "relation": "ACTION_POLICY",
                "slots": {
                    "action": {"type": "action", "value": "modify"},
                    "object": {"type": "entity", "value": "production"},
                    "polarity": {"type": "boolean", "value": False},
                },
            },
            {
                "kind": "CONSTRAINT",
                "relation": "ACTION_POLICY",
                "slots": {
                    "action": {"type": "action", "value": "modify"},
                    "object": {"type": "entity", "value": "production"},
                    "polarity": {"type": "boolean", "value": True},
                },
            },
        ],
    )
    row.pop("frame_kinds")
    lessons.write_text(json.dumps(row) + "\n", encoding="utf-8")

    rows = _reviewed_rows(lessons)
    frame_targets, cardinalities, groups = _targets(rows)

    assert rows[0]["frame_kinds"] == ["CONSTRAINT"]
    assert int(cardinalities[0]) == 2
    assert int(frame_targets[0].sum()) == 1
    assert groups.tolist() == [-1]


def test_same_kind_or_different_number_is_not_an_implicit_positive_pair(tmp_path: Path):
    lessons = tmp_path / "ungrouped.jsonl"
    rows = [
        _lesson(
            id="first", text="Keep latency below 50 ms", frames=[_threshold_frame(50)]
        ),
        _lesson(
            id="second", text="Keep latency below 80 ms", frames=[_threshold_frame(80)]
        ),
    ]
    lessons.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    reviewed = _reviewed_rows(lessons)
    _, _, groups = _targets(reviewed)

    assert groups.tolist() == [-1, -1]
    identical_embeddings = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    assert float(supervised_contrastive_loss(identical_embeddings, groups)) > 0.7


def test_paraphrase_group_requires_one_exact_structured_frame_set(tmp_path: Path):
    lessons = tmp_path / "bad-pair.jsonl"
    rows = [
        _lesson(
            id="first",
            text="Keep latency below 50 ms",
            frames=[_threshold_frame(50)],
            paraphrase_group="latency-threshold",
        ),
        _lesson(
            id="second",
            text="Response time must stay under 80 ms",
            frames=[_threshold_frame(80)],
            paraphrase_group="latency-threshold",
        ),
    ]
    lessons.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not map to one exact frame set"):
        train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)
    assert not (tmp_path / "candidate" / "semantic-breadth.pt").exists()


def test_exact_semantic_target_cannot_be_split_across_paraphrase_groups(
    tmp_path: Path,
):
    lessons = tmp_path / "duplicate-target-groups.jsonl"
    frame = _policy_frame(False)
    rows = [
        _lesson(
            id=f"duplicate-{index}",
            text=text,
            frame_kinds=["CONSTRAINT"],
            frames=[frame],
            paraphrase_group=group,
        )
        for index, (group, text) in enumerate(
            (
                ("first-group", "Never modify the rehearsal registry"),
                ("first-group", "The rehearsal registry must remain unchanged"),
                ("second-group", "Do not alter the rehearsal registry"),
                ("second-group", "Modification of the rehearsal registry is forbidden"),
            ),
            1,
        )
    ]
    lessons.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="multiple paraphrase_group IDs"):
        train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)
    assert not (tmp_path / "candidate" / "semantic-breadth.pt").exists()


@pytest.mark.parametrize(
    "group_assignments",
    [
        (None, None),
        ("one-group", "one-group", None),
    ],
    ids=("multiple-ungrouped", "grouped-plus-ungrouped"),
)
def test_repeated_exact_target_cannot_include_ungrouped_rows(
    tmp_path: Path,
    group_assignments: tuple[str | None, ...],
):
    lessons = tmp_path / "ungrouped-duplicate-targets.jsonl"
    texts = (
        "Never modify the trial registry",
        "The trial registry must remain unchanged",
        "Modification of the trial registry is forbidden",
    )
    rows = [
        _lesson(
            id=f"ungrouped-duplicate-{index}",
            text=texts[index - 1],
            frame_kinds=["CONSTRAINT"],
            frames=[_policy_frame(False)],
            paraphrase_group=group,
        )
        for index, group in enumerate(group_assignments, 1)
    ]
    lessons.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="repeated exact semantic target"):
        train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)
    assert not (tmp_path / "candidate" / "semantic-breadth.pt").exists()


@pytest.mark.parametrize("mismatch", ["relation", "polarity", "slot_type"])
def test_paraphrase_group_rejects_any_semantic_frame_mismatch(
    tmp_path: Path, mismatch: str
):
    first = _policy_frame(False)
    second = _policy_frame(False)
    if mismatch == "relation":
        second["relation"] = "THRESHOLD"
    elif mismatch == "polarity":
        second["slots"]["polarity"]["value"] = True
    else:
        second["slots"]["polarity"]["type"] = "text"
    lessons = tmp_path / f"mismatch-{mismatch}.jsonl"
    rows = [
        _lesson(
            id="first",
            text="Production modification is forbidden",
            frame_kinds=["CONSTRAINT"],
            frames=[first],
            paraphrase_group="production-policy",
        ),
        _lesson(
            id="second",
            text="Do not modify production",
            frame_kinds=["CONSTRAINT"],
            frames=[second],
            paraphrase_group="production-policy",
        ),
    ]
    lessons.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not map to one exact frame set"):
        _reviewed_rows(lessons)


def test_structured_paraphrase_targets_require_slot_type_and_value(tmp_path: Path):
    frame = _threshold_frame()
    frame["slots"]["value"] = 50
    lessons = tmp_path / "untyped.jsonl"
    lessons.write_text(
        json.dumps(
            _lesson(
                text="Keep latency below 50 ms",
                frames=[frame],
                paraphrase_group="invalid",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="require typed slots"):
        _reviewed_rows(lessons)


def test_reviewed_exact_paraphrase_pair_is_logged_in_training_receipt(tmp_path: Path):
    lessons = tmp_path / "good-pair.jsonl"
    rows = [
        _lesson(
            id="first",
            text="Keep latency below 50 ms",
            frames=[_threshold_frame(50)],
            paraphrase_group="latency-threshold",
        ),
        _lesson(
            id="second",
            text="Response time must stay under 50 ms",
            frames=[_threshold_frame(50)],
            paraphrase_group="latency-threshold",
        ),
    ]
    lessons.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    receipt = train(_parent(), tmp_path / "candidate", steps=1, extra_jsonl=lessons)

    assert receipt["contrastive_positive_pairs"] == 1
    assert receipt["paraphrase_groups"][0]["group"] == "latency-threshold"
    assert receipt["paraphrase_groups"][0]["rows"] == 2
    assert len(receipt["paraphrase_groups_sha256"]) == 64
    assert receipt["parent_sha256"] == sha256_file(_parent())
    assert receipt["lessons_sha256"] == sha256_file(lessons)
    assert receipt["pinned_held_out_vocabulary_file_sha256"] == (
        PINNED_HELD_OUT_VOCABULARY_FILE_SHA256
    )
    assert receipt["pinned_held_out_vocabulary_expected_file_sha256"] == (
        PINNED_HELD_OUT_VOCABULARY_FILE_SHA256
    )
    assert [
        Path(artifact["path"]).name
        for artifact in receipt["required_held_out_vocabulary_artifacts"]
    ] == ["held-out-vocabulary-v1.txt", "held-out-vocabulary-v2.txt"]
    assert receipt["required_held_out_vocabulary_artifacts_sha256"] == (
        _canonical_json_sha256(receipt["required_held_out_vocabulary_artifacts"])
    )
    assert "attested" in receipt["required_held_out_vocabulary"]
    assert receipt["training_inputs_sha256"] == _canonical_json_sha256(
        receipt["training_inputs"]
    )
    assert receipt["optimizer"] == receipt["training_inputs"]["optimizer"]
    assert receipt["torch_num_threads"] == 2
    assert all(
        sha256_file(artifact["path"]) == artifact["sha256"]
        for artifact in receipt["training_exclusion_artifacts"]
    )
    checkpoint = torch.load(
        receipt["challenger_path"], map_location="cpu", weights_only=True
    )
    assert checkpoint["training_inputs_sha256"] == receipt["training_inputs_sha256"]
    assert (
        checkpoint["required_held_out_vocabulary_artifacts"]
        == receipt["required_held_out_vocabulary_artifacts"]
    )
    assert (
        checkpoint["required_held_out_vocabulary_artifacts_sha256"]
        == receipt["required_held_out_vocabulary_artifacts_sha256"]
    )
    assert checkpoint["trainer_source_sha256"] == sha256_file(
        Path(semantic_breadth.__file__)
    )

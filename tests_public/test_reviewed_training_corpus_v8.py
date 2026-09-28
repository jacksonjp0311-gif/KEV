from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

import pytest

from kev.artifacts import file_sha256
from kev.uc51a2.semantic_breadth import _reviewed_rows_from_text, _target_hash
from scripts import build_reviewed_training_corpus_v8 as builder


def test_v8_removes_only_three_flagged_rows_and_preserves_all_other_content():
    before = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in builder.SOURCE_CORPUS.read_text(encoding="utf-8").splitlines()
        )
    }
    rows = builder.build_rows()
    assert len(rows) == 497
    assert set(before) - {row["derived_from_lesson_id"] for row in rows} == set(
        builder.REMOVED_LESSON_IDS
    )
    for row in rows:
        source = before[row["derived_from_lesson_id"]]
        assert row["text"] == source["text"]
        assert row["frames"] == source["frames"]
        assert row["frame_kinds"] == source["frame_kinds"]
        assert row["frame_cardinality"] == source["frame_cardinality"]
        assert row["source_corpus_sha256"] == builder.SOURCE_CORPUS_SHA256
        assert row["reviewed_by"] == builder.REVIEWER
        assert row["review"]["independent_human_review"] is False
        assert row["predecessor_lineage"]["prior_derived_from_lesson_id"] == source.get(
            "derived_from_lesson_id"
        )


def test_v8_retains_two_paraphrases_per_affected_group_and_global_target_identity():
    rows = builder.build_rows()
    parsed = _reviewed_rows_from_text(builder.prior.helpers.corpus_bytes(rows).decode())
    groups = Counter(row["paraphrase_group"] for row in parsed)
    assert min(groups.values()) == 2
    assert len(groups) == 157
    for receipt in (31, 44, 58):
        assert groups[f"v8-retained-receipt-rec-{receipt}"] == 2
    target_to_group = {}
    for row in parsed:
        target = _target_hash(row["frames"])
        assert (
            target_to_group.setdefault(target, row["paraphrase_group"])
            == row["paraphrase_group"]
        )
    assert Counter(row["frame_cardinality"] for row in parsed) == {
        1: 198,
        2: 174,
        3: 84,
        4: 41,
    }


def test_v8_replays_hashes_and_discloses_pretraining_rejection():
    corpus, _, manifest = builder.build_artifacts()
    assert builder.build_artifacts() == (
        corpus,
        builder.prior._json_bytes(manifest),
        manifest,
    )
    assert manifest["corpus"]["sha256"] == hashlib.sha256(corpus).hexdigest()
    assert manifest["builder"]["sha256"] == file_sha256(builder.__file__)
    rejection = manifest["predecessor_rejection"]
    assert rejection["sha256"] == builder.REJECTION_RECEIPT_SHA256
    assert rejection["optimizer_steps"] == 0
    assert rejection["model_scores_used"] is False
    assert rejection["removed_lesson_ids"] == list(builder.REMOVED_LESSON_IDS)
    assert manifest["separation"]["v7_evaluation_prompts_read"] is False
    assert manifest["metrics"]["contrastive_positive_pair_count"] == 570
    assert file_sha256(builder.SOURCE_CORPUS) == builder.SOURCE_CORPUS_SHA256
    assert file_sha256(builder.SOURCE_MANIFEST) == builder.SOURCE_MANIFEST_SHA256
    assert file_sha256(builder.prior.__file__) == builder.SOURCE_BUILDER_SHA256


def test_v8_exclusive_write_and_mutation_detection(tmp_path: Path):
    manifest = builder.write_artifacts(tmp_path)
    assert builder.check_artifacts(tmp_path) == manifest
    with pytest.raises(FileExistsError, match="refusing to refresh"):
        builder.write_artifacts(tmp_path)
    target = tmp_path / builder.CORPUS_NAME
    target.write_bytes(target.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="does not replay byte-for-byte"):
        builder.check_artifacts(tmp_path)

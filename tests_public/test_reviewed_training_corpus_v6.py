from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256
from scripts import build_reviewed_training_corpus_v6 as builder


def _artifacts() -> tuple[bytes, bytes, bytes, dict]:
    return builder.build_artifacts(
        builder.prior.PINNED_V2_VOCABULARY,
        builder.prior.PINNED_V2_VOCABULARY_SHA256,
    )


def _rows(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def test_v6_corpus_is_deterministic_hash_bound_and_globally_group_consistent():
    first = _artifacts()
    second = _artifacts()
    assert first == second
    corpus, _, _, manifest = first
    rows = _rows(corpus)

    assert manifest["corpus"]["sha256"] == hashlib.sha256(corpus).hexdigest()
    assert manifest["corpus"]["canonical_rows_sha256"] == canonical_json_sha256(rows)
    assert manifest["builder"]["sha256"] == file_sha256(builder.__file__)
    assert manifest["metrics"]["row_count"] == 197
    assert manifest["metrics"]["paraphrase_group_count"] == 58
    assert manifest["metrics"]["contrastive_positive_pair_count"] == 246

    target_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        target_groups[canonical_json_sha256(row["frames"])].add(row["paraphrase_group"])
    assert all(len(groups) == 1 for groups in target_groups.values())
    merged = [
        row for row in rows if row["paraphrase_group"] == "v6-evidence-consistent"
    ]
    assert len(merged) == 6


def test_v6_rows_preserve_honest_review_and_permission_provenance():
    corpus, _, _, manifest = _artifacts()
    rows = _rows(corpus)

    assert manifest["provenance"]["independent_human_review"] is False
    assert manifest["provenance"]["ordinary_chat_training_data"] is False
    assert manifest["provenance"]["third_party_text"] is False
    assert all(row["status"] == "REVIEWED" for row in rows)
    assert all(row["reviewed_by"] == builder.REVIEWER for row in rows)
    assert all(row["reviewed_at"] == builder.REVIEWED_AT for row in rows)
    assert all(row["permission"] == builder.PERMISSION for row in rows)
    assert all(row["derived_from_lesson_id"].startswith("v5-") for row in rows)
    assert all(row["authorship"]["ordinary_chat_source"] is False for row in rows)
    assert all(row["authorship"]["third_party_source"] is False for row in rows)
    assert all(row["review"]["independent_human_review"] is False for row in rows)


def test_v5_rejection_receipt_preserves_collision_and_proves_no_training():
    _, _, rejection_bytes, manifest = _artifacts()
    rejection = json.loads(rejection_bytes.decode("utf-8"))

    assert rejection["decision"] == "REJECTED_BEFORE_TRAINING"
    assert rejection["reason_code"] == ("SEMANTIC_TARGET_MULTIPLE_PARAPHRASE_GROUPS")
    assert rejection["collision"]["paraphrase_groups"] == list(builder.COLLIDING_GROUPS)
    assert rejection["collision"]["row_count"] == 6
    assert rejection["training"] == {
        "optimizer_steps": 0,
        "checkpoint_created": False,
        "model_weights_touched": False,
        "training_receipt": None,
    }
    assert rejection["promotion"] == {
        "decision": "NOT_APPLICABLE",
        "eval_card": None,
        "ledger_model_event": None,
    }
    assert {item["sha256"] for item in rejection["artifacts_preserved"]} == {
        builder.V5_CORPUS_SHA256,
        builder.V5_MANIFEST_SHA256,
        builder.V5_BUILDER_SHA256,
    }
    assert (
        manifest["predecessor_rejection"]["sha256"]
        == hashlib.sha256(rejection_bytes).hexdigest()
    )
    assert manifest["predecessor_rejection"]["canonical_json_sha256"] == (
        canonical_json_sha256(rejection)
    )


def test_v6_builder_remains_blind_to_v5_prompts_and_excludes_both_vocabularies():
    corpus, _, _, manifest = _artifacts()
    rows = _rows(corpus)
    words = {word for row in rows for word in builder.prior._words(row["text"])}
    v1, _ = builder.prior._vocabulary(
        builder.prior.PINNED_V1_VOCABULARY,
        builder.prior.PINNED_V1_VOCABULARY_SHA256,
    )
    v2, _ = builder.prior._vocabulary(
        builder.prior.PINNED_V2_VOCABULARY,
        builder.prior.PINNED_V2_VOCABULARY_SHA256,
    )
    assert not words & (v1 | v2)
    assert manifest["separation"]["v5_promotion_suite_read_by_builder"] is False
    assert manifest["separation"]["v5_promotion_suite_surface_used"] is False
    assert (
        manifest["separation"]["v5_promotion_suite_provenance_only"]["content_read"]
        is False
    )


def test_v6_builder_writes_once_and_replays_byte_for_byte(tmp_path: Path):
    manifest = builder.write_artifacts(
        builder.prior.PINNED_V2_VOCABULARY,
        builder.prior.PINNED_V2_VOCABULARY_SHA256,
        tmp_path,
    )
    assert (
        builder.check_artifacts(
            builder.prior.PINNED_V2_VOCABULARY,
            builder.prior.PINNED_V2_VOCABULARY_SHA256,
            tmp_path,
        )
        == manifest
    )
    with pytest.raises(FileExistsError, match="refusing to refresh"):
        builder.write_artifacts(
            builder.prior.PINNED_V2_VOCABULARY,
            builder.prior.PINNED_V2_VOCABULARY_SHA256,
            tmp_path,
        )

    corpus_path = tmp_path / builder.CORPUS_NAME
    corpus_path.write_bytes(corpus_path.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="does not replay byte-for-byte"):
        builder.check_artifacts(
            builder.prior.PINNED_V2_VOCABULARY,
            builder.prior.PINNED_V2_VOCABULARY_SHA256,
            tmp_path,
        )

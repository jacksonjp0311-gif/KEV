from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.frames import FRAME_KINDS
from scripts import build_reviewed_training_corpus_v5 as builder


ROOT = Path(__file__).resolve().parents[1]


def _artifacts() -> tuple[bytes, bytes, dict]:
    corpus = (ROOT / "training/reviewed" / builder.CORPUS_NAME).read_bytes()
    manifest_bytes = (ROOT / "training/reviewed" / builder.MANIFEST_NAME).read_bytes()
    return (
        corpus,
        manifest_bytes,
        json.loads(manifest_bytes.decode("utf-8")),
    )


def _rows(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line]


def test_rejected_v5_reviewed_corpus_bytes_are_preserved_and_hash_bound():
    first_corpus, first_manifest_bytes, first_manifest = _artifacts()
    second_corpus, second_manifest_bytes, second_manifest = _artifacts()

    assert first_corpus == second_corpus
    assert first_manifest_bytes == second_manifest_bytes
    assert first_manifest == second_manifest
    rows = _rows(first_corpus)
    reference = first_manifest["corpus"]
    assert reference["sha256"] == hashlib.sha256(first_corpus).hexdigest()
    assert reference["size_bytes"] == len(first_corpus)
    assert reference["canonical_rows_sha256"] == canonical_json_sha256(rows)
    assert reference["semantic_targets_sha256"] == canonical_json_sha256(
        [{"id": row["id"], "frames": row["frames"]} for row in rows]
    )
    assert first_manifest["builder"]["sha256"] == file_sha256(builder.__file__)
    assert first_manifest["metrics"] == {
        "row_count": 197,
        "paraphrase_group_count": 59,
        "contrastive_positive_pair_count": 237,
        "kind_occurrences": {
            "ACTIVE_SELECTION": 9,
            "CONSTRAINT": 61,
            "COPY_VALUE": 9,
            "EVIDENCE_CONSISTENCY": 9,
            "GOAL": 77,
            "MAGNITUDE": 9,
            "NEGATE": 9,
            "OBSERVATION": 57,
            "PREDICTION": 29,
            "RECEIPT_VALUE": 9,
            "REFERENCE": 9,
            "RUN_STATUS": 9,
            "SUPERSEDES": 9,
        },
        "row_kind_presence": {
            "ACTIVE_SELECTION": 9,
            "CONSTRAINT": 53,
            "COPY_VALUE": 9,
            "EVIDENCE_CONSISTENCY": 9,
            "GOAL": 69,
            "MAGNITUDE": 9,
            "NEGATE": 9,
            "OBSERVATION": 57,
            "PREDICTION": 29,
            "RECEIPT_VALUE": 9,
            "REFERENCE": 9,
            "RUN_STATUS": 9,
            "SUPERSEDES": 9,
        },
        "cardinality_rows": {"1": 117, "2": 60, "3": 12, "4": 8},
        "training_tag_rows": {
            "conflict": 3,
            "conflicting-constraints": 8,
            "multi-frame": 80,
            "negation": 47,
            "number-change": 26,
            "number-control": 97,
            "order-swap": 80,
            "paraphrase": 108,
            "polite": 36,
            "polite-buried-constraint": 36,
            "positive-policy": 3,
            "repeated-kind": 16,
            "revision": 9,
            "single-frame": 117,
        },
    }


def test_v5_corpus_provenance_is_explicit_and_does_not_claim_human_review():
    corpus, _, manifest = _artifacts()
    rows = _rows(corpus)

    assert manifest["provenance"]["authorship"] == "REPOSITORY_AUTHORED_SYNTHETIC"
    assert manifest["provenance"]["review_type"] == "SAME_PARTY_AGENT_REVIEW"
    assert manifest["provenance"]["independent_human_review"] is False
    assert manifest["provenance"]["ordinary_chat_training_data"] is False
    assert manifest["provenance"]["third_party_text"] is False
    assert all(row["status"] == "REVIEWED" for row in rows)
    assert all(row["reviewed_by"] == builder.REVIEWER for row in rows)
    assert all(row["permission"] == builder.PERMISSION for row in rows)
    assert all(row["authorship"]["ordinary_chat_source"] is False for row in rows)
    assert all(row["authorship"]["third_party_source"] is False for row in rows)
    assert all(row["review"]["independent_human_review"] is False for row in rows)
    assert all("intent" not in row for row in rows)


def test_v5_corpus_has_exact_same_frame_set_groups_and_composition_coverage():
    corpus, _, manifest = _artifacts()
    rows = _rows(corpus)
    groups: dict[str, list[dict]] = defaultdict(list)
    all_kinds: set[str] = set()
    cardinalities: set[int] = set()
    all_tags: set[str] = set()
    duplicate_kind_rows = 0
    for row in rows:
        groups[row["paraphrase_group"]].append(row)
        kinds = [frame["kind"] for frame in row["frames"]]
        all_kinds.update(kinds)
        cardinalities.add(len(row["frames"]))
        all_tags.update(row["training_tags"])
        if len(kinds) != len(set(kinds)):
            duplicate_kind_rows += 1
        assert row["frame_cardinality"] == len(row["frames"])
        assert set(row["frame_kinds"]) == set(kinds)
        assert all(frame["relation"] for frame in row["frames"])
        assert all(
            slot.get("type") and "value" in slot
            for frame in row["frames"]
            for slot in frame["slots"].values()
        )

    assert all_kinds == set(FRAME_KINDS)
    assert cardinalities == {1, 2, 3, 4}
    assert duplicate_kind_rows == 16
    assert all(len(members) >= 2 for members in groups.values())
    for members in groups.values():
        assert len({canonical_json_sha256(row["frames"]) for row in members}) == 1
    target_groups: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        target_groups[canonical_json_sha256(row["frames"])].add(row["paraphrase_group"])
    assert [
        sorted(group_ids) for group_ids in target_groups.values() if len(group_ids) > 1
    ] == [["evidence-consistent-alpha", "evidence-consistent-gamma"]]
    assert set(manifest["coverage"]["required_phenomena"]) <= all_tags


def test_v5_builder_excludes_v1_v4_surfaces_and_both_vocabularies_without_reading_v5_suite():
    corpus, _, manifest = _artifacts()
    rows = _rows(corpus)
    text_words = {word for row in rows for word in builder._words(row["text"])}
    v1_terms, _ = builder._vocabulary(
        builder.PINNED_V1_VOCABULARY,
        builder.PINNED_V1_VOCABULARY_SHA256,
    )
    v2_terms, _ = builder._vocabulary(
        builder.PINNED_V2_VOCABULARY,
        builder.PINNED_V2_VOCABULARY_SHA256,
    )
    forbidden, references = builder._frozen_v1_v4_surfaces()

    assert not text_words & (v1_terms | v2_terms)
    assert not {row["text"].strip().casefold() for row in rows} & forbidden
    assert [reference["path"] for reference in references] == [
        item[0] for item in builder.V1_V4_SURFACE_ARTIFACTS
    ]
    assert all("v5" not in item[0] for item in builder.V1_V4_SURFACE_ARTIFACTS)
    separation = manifest["separation"]
    assert separation["v5_promotion_suite_read_by_builder"] is False
    assert separation["v5_promotion_suite_surface_used"] is False
    assert separation["v5_vocabulary_holdout_read_only"] is True
    assert separation["v5_promotion_suite_provenance_only"] == {
        "file_sha256": builder.V5_SUITE_FILE_SHA256_PROVENANCE_ONLY,
        "canonical_sha256": builder.V5_SUITE_CANONICAL_SHA256_PROVENANCE_ONLY,
        "content_read": False,
    }


def test_v5_builder_rejects_exclusion_collisions_and_wrong_holdout_hash():
    rows = builder.build_rows()
    forbidden, _ = builder._frozen_v1_v4_surfaces()
    collision = deepcopy(rows)
    collision[0]["text"] = sorted(forbidden)[0]
    with pytest.raises(ValueError, match="overlaps preserved frozen surfaces"):
        builder.validate_rows(
            collision,
            held_out_terms=(),
            forbidden_surfaces=forbidden,
        )

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        builder.build_artifacts(builder.PINNED_V2_VOCABULARY, "0" * 64)


def test_hardened_training_contract_rejects_the_preserved_v5_collision():
    with pytest.raises(ValueError, match="multiple paraphrase_group IDs"):
        builder.build_artifacts(
            builder.PINNED_V2_VOCABULARY,
            builder.PINNED_V2_VOCABULARY_SHA256,
        )

    corpus_path = ROOT / "training/reviewed" / builder.CORPUS_NAME
    manifest_path = ROOT / "training/reviewed" / builder.MANIFEST_NAME
    assert file_sha256(corpus_path) == (
        "91b49f47804120906537f2b3eafeef5ddf1df1663f114e5c91a1c9fa59ce5dd8"
    )
    assert file_sha256(manifest_path) == (
        "e783ece3d8aa4ac3d104833345ef771d1aa67536060a81be5e793c9c4d9c3b3d"
    )


def test_v5_builder_never_writes_after_hardened_validation_rejects(tmp_path: Path):
    with pytest.raises(ValueError, match="multiple paraphrase_group IDs"):
        builder.write_artifacts(
            builder.PINNED_V2_VOCABULARY,
            builder.PINNED_V2_VOCABULARY_SHA256,
            tmp_path,
        )
    assert not (tmp_path / builder.CORPUS_NAME).exists()
    assert not (tmp_path / builder.MANIFEST_NAME).exists()

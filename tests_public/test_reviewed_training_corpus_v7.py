from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.uc51a2.semantic_breadth import _reviewed_rows_from_text, _target_hash
from scripts import build_reviewed_training_corpus_v7 as builder


def test_v7_corpus_replays_and_retains_hash_bound_v6_lessons():
    first = builder.build_artifacts()
    assert first == builder.build_artifacts()
    corpus, _, manifest = first
    rows = [json.loads(line) for line in corpus.decode().splitlines()]
    assert manifest["corpus"]["sha256"] == hashlib.sha256(corpus).hexdigest()
    assert manifest["corpus"]["canonical_rows_sha256"] == canonical_json_sha256(rows)
    assert manifest["builder"]["sha256"] == file_sha256(builder.__file__)
    source = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in builder.SOURCE_CORPUS.read_text(encoding="utf-8").splitlines()
        )
    }
    retained = [row for row in rows if "retained-v6" in row["training_tags"]]
    assert len(retained) == 197
    for row in retained:
        original = source[row["derived_from_lesson_id"]]
        assert row["text"] == original["text"]
        assert row["frames"] == original["frames"]
        assert row["source_corpus_sha256"] == builder.SOURCE_CORPUS_SHA256
        assert row["source_review"] == {
            key: original[key] for key in ("reviewed_by", "reviewed_at", "permission")
        }


def test_v7_group_identity_is_exact_multiset_and_retains_repeated_kinds():
    rows = builder.build_rows()
    groups: dict[str, list[dict]] = defaultdict(list)
    targets: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        groups[row["paraphrase_group"]].append(row)
        targets[_target_hash(row["frames"])].add(row["paraphrase_group"])
    assert all(len(value) == 1 for value in targets.values())
    assert all(len(value) >= 3 for value in groups.values())
    parsed = _reviewed_rows_from_text(builder.helpers.corpus_bytes(rows).decode())
    repeated = [row for row in parsed if len(row["frames"]) > len(row["frame_kinds"])]
    assert len(repeated) >= 75
    assert all(row["frame_cardinality"] == len(row["frames"]) for row in repeated)
    counts = Counter(len(row["frames"]) for row in rows)
    assert 350 <= len(rows) <= 550
    assert counts[1] >= 190
    assert counts[2] >= 160
    assert counts[3] >= 80
    assert counts[4] >= 35


def test_v7_composition_preserves_each_bound_value_under_clause_permutation():
    atoms = {atom["id"]: atom for atom in builder.semantic_atoms()}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in builder.build_rows():
        if "new-v7" in row["training_tags"] and row["frame_cardinality"] > 1:
            groups[row["paraphrase_group"]].append(row)
    for rows in groups.values():
        assert len(rows) == 3
        expected_ids = rows[0]["semantic_atom_ids"]
        expected_frames = [atoms[key]["frame"] for key in expected_ids]
        assert all(row["frames"] == expected_frames for row in rows)
        first, reversed_row, rotated = rows
        assert first["text"].startswith(atoms[expected_ids[0]]["clauses"][0])
        assert reversed_row["text"].startswith(atoms[expected_ids[-1]]["clauses"][1])
        assert rotated["text"].startswith(atoms[expected_ids[1]]["clauses"][2])
        for row in rows:
            for frame in row["frames"]:
                for slot in frame["slots"].values():
                    if slot["type"] == "number":
                        assert str(slot["value"]) in row["text"]


def test_v7_corpus_enforces_permission_and_blind_holdout_exclusions():
    held_out, forbidden, _, references = builder.exclusions()
    rows = builder.build_rows()
    builder.validate_rows(rows, held_out_terms=held_out, forbidden_surfaces=forbidden)
    assert "ameliorate" in held_out
    assert len(held_out) == 18
    assert all("v7" not in item["path"] for item in references)
    for row in rows:
        assert row["review"]["independent_human_review"] is False
        assert row["authorship"]["ordinary_chat_source"] is False
        assert row["authorship"]["third_party_source"] is False
    corrupted = deepcopy(rows)
    corrupted[-1]["text"] += " ameliorate"
    with pytest.raises(ValueError, match="held-out vocabulary"):
        builder.validate_rows(
            corrupted, held_out_terms=held_out, forbidden_surfaces=forbidden
        )
    with pytest.raises(ValueError, match="overlaps a frozen"):
        builder.validate_rows(
            rows, held_out_terms=held_out, forbidden_surfaces=[rows[-1]["text"]]
        )


def test_v7_changed_number_polarity_or_duplicate_multiplicity_cannot_be_positive_pair():
    rows = builder.build_rows()
    selected = [row for row in rows if row["id"].startswith("v7-single-goal-jitter-")]
    altered = deepcopy(selected)
    altered[1]["frames"][0]["slots"]["value"]["value"] += 1
    with pytest.raises(ValueError, match="does not map to one exact frame set"):
        _reviewed_rows_from_text(builder.helpers.corpus_bytes(altered).decode())


def test_v7_changed_polarity_or_duplicate_multiplicity_cannot_be_positive_pair():
    rows = builder.build_rows()
    selected = [
        row for row in rows if row["id"].startswith("v7-single-constraint-archive-")
    ]
    altered = deepcopy(selected)
    altered[1]["frames"][0]["slots"]["polarity"]["value"] = True
    with pytest.raises(ValueError, match="does not map to one exact frame set"):
        _reviewed_rows_from_text(builder.helpers.corpus_bytes(altered).decode())
    altered = deepcopy(selected)
    altered[1]["frames"].append(deepcopy(altered[1]["frames"][0]))
    with pytest.raises(ValueError, match="does not map to one exact frame set"):
        _reviewed_rows_from_text(builder.helpers.corpus_bytes(altered).decode())


def test_v7_operator_targets_use_runtime_canonical_symbols():
    rows = builder.build_rows()
    depth = next(row for row in rows if row["id"] == "v7-single-goal-depth-01")
    assert depth["frames"][0]["slots"]["operator"]["value"] == "LTE"
    depth["frames"][0]["slots"]["operator"]["value"] = "LE"
    with pytest.raises(ValueError, match="unknown canonical comparison operator"):
        builder.validate_rows(rows, held_out_terms=(), forbidden_surfaces=())


def test_v7_same_target_cannot_cross_groups_even_with_reordered_frames():
    rows = [
        row
        for row in builder.build_rows()
        if row["id"].startswith("v7-composition-001-")
    ]
    duplicates = deepcopy(rows)
    for row in duplicates:
        row["id"] += "-other"
        row["text"] += " Please."
        row["paraphrase_group"] += "-other"
        row["frames"].reverse()
    with pytest.raises(ValueError, match="multiple paraphrase_group IDs"):
        _reviewed_rows_from_text(
            builder.helpers.corpus_bytes(rows + duplicates).decode()
        )


def test_v7_corpus_writes_once_and_detects_frozen_artifact_mutation(tmp_path: Path):
    manifest = builder.write_artifacts(tmp_path)
    assert builder.check_artifacts(tmp_path) == manifest
    with pytest.raises(FileExistsError, match="refusing to refresh"):
        builder.write_artifacts(tmp_path)
    target = tmp_path / builder.CORPUS_NAME
    target.write_bytes(target.read_bytes() + b"{}\n")
    with pytest.raises(ValueError, match="does not replay byte-for-byte"):
        builder.check_artifacts(tmp_path)

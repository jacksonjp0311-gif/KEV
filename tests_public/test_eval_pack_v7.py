from __future__ import annotations

import copy
import json
import re
from collections import Counter

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.evaluation import load_frozen_suite
from scripts import build_public_eval_pack_v7 as builder


def test_v7_rebuild_is_exact_and_preserved_boundary_is_valid():
    assert file_sha256(builder.SUITE_PATH) == (
        "e22bdbea5f07c8b44cf5a1684a2b8433be6b252d74ddf86ef0252b736dbe9c76"
    )
    assert file_sha256(builder.MANIFEST_PATH) == (
        "f8784d55629f86377ca80aadf52c838e40b20c497c28653b8605d5e505f6a036"
    )
    suite = json.loads(builder.SUITE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(builder.MANIFEST_PATH.read_text(encoding="utf-8"))
    assert builder.build_suite() == suite
    assert builder.build_manifest(suite) == manifest
    assert builder.json_bytes(suite) == builder.SUITE_PATH.read_bytes()
    assert builder.json_bytes(manifest) == builder.MANIFEST_PATH.read_bytes()
    loaded = load_frozen_suite(builder.SUITE_PATH)
    assert {
        key: len(value) for key, value in loaded.splits.items()
    } == builder.SPLIT_COUNTS
    assert loaded.canonical_sha256 == canonical_json_sha256(suite)
    audit = builder.validate_suite(suite)
    assert audit["new_surfaces"] == 220
    assert audit["preserved_retention"] == 40
    assert audit["new_exact_historical_overlaps"] == 0
    assert audit["new_digit_template_historical_overlaps"] == 0
    assert suite["provenance"]["authored_without_v7_runtime_or_model_results"] is True
    assert "SAME_PARTY" in suite["provenance"]["review"]


def test_v7_manifest_retains_the_entire_exclusion_history():
    manifest = builder.build_manifest()
    paths = {entry["path"] for entry in manifest["artifacts"].values()}
    assert {
        f"evals/frozen/public-audit-v{version}-260.json" for version in range(1, 8)
    } <= paths
    assert {
        "evals/frozen/calibration-fit-v1.jsonl",
        "evals/frozen/calibration-fit-v2.jsonl",
        "evals/frozen/held-out-vocabulary-v1.txt",
        "evals/frozen/held-out-vocabulary-v2.txt",
        "evals/frozen/frame-parser-known-failures-v1.json",
    } <= paths
    for reference in manifest["artifacts"].values():
        path = builder.ROOT / reference["path"]
        assert file_sha256(path) == reference["sha256"]
        assert path.stat().st_size == reference["size_bytes"]
        if "canonical_sha256" in reference:
            assert (
                canonical_json_sha256(json.loads(path.read_text(encoding="utf-8")))
                == reference["canonical_sha256"]
            )


def test_v7_composition_tests_exact_slots_multiplicity_and_paired_changes():
    suite = builder.build_suite()
    rows = suite["splits"]["composition"]
    assert Counter(row["audit"] for row in rows) == Counter(
        {name: 10 for name in builder.COMPOSITION_AUDITS}
    )
    assert all(row["projection"] == "semantic_frames" for row in rows)
    assert all(len(row["expected"]) >= 2 for row in rows)
    assert all(
        len({frame["kind"] for frame in row["expected"]}) < len(row["expected"])
        for row in rows
        if row["audit"] == "repeated-kind"
    )
    assert sum(row["projection"] == "kinds" for row in suite["splits"]["fresh"]) == 52
    for row in rows:
        if row["audit"] == "conflicting-constraints":
            left, right = row["expected"]
            assert left["slots"]["object"] == right["slots"]["object"]
            assert left["slots"]["action"] == right["slots"]["action"]
            assert left["slots"]["polarity"] != right["slots"]["polarity"]


def test_v7_audit_rejects_number_only_historical_refresh_and_label_leakage():
    suite = builder.build_suite()
    previous = json.loads(builder.PREVIOUS_SUITE.read_text(encoding="utf-8"))
    suite["splits"]["fresh"][0]["input"] = previous["splits"]["fresh"][0]["input"]
    with pytest.raises(ValueError, match="historical exact or number-only"):
        builder.validate_suite(suite)
    suite["splits"]["fresh"][0]["input"] = re.sub(
        r"\d+",
        lambda match: str(int(match.group()) + 1000),
        previous["splits"]["fresh"][0]["input"],
    )
    with pytest.raises(ValueError, match="historical exact or number-only"):
        builder.validate_suite(suite)
    suite = builder.build_suite()
    suite["splits"]["fresh"][0]["input"] = (
        "The expected copy value label is written here."
    )
    with pytest.raises(ValueError, match="expected label leaked"):
        builder.validate_suite(suite)


def test_v7_audit_rejects_changed_retention_or_erased_composition_slots():
    suite = builder.build_suite()
    suite["splits"]["retention"][0]["expected"][0]["slots"]["source"] = "invented"
    with pytest.raises(ValueError, match="retention content or target changed"):
        builder.validate_suite(suite)
    suite = builder.build_suite()
    row = suite["splits"]["composition"][0]
    row["projection"] = "kinds"
    row["expected"] = [target["kind"] for target in row["expected"]]
    with pytest.raises(ValueError, match="exact multiframe semantic targets"):
        builder.validate_suite(suite)


def test_v7_audit_rejects_semantically_unchanged_number_pair():
    suite = builder.build_suite()
    pair = [
        row
        for row in suite["splits"]["composition"]
        if row.get("pair_id") == "v7-number-0"
    ]
    pair[1]["expected"] = copy.deepcopy(pair[0]["expected"])
    with pytest.raises(ValueError, match="incorrect semantic equality"):
        builder.validate_suite(suite)


def test_v7_builder_refuses_in_place_refresh():
    paths = (builder.SUITE_PATH, builder.MANIFEST_PATH)
    before = {path: file_sha256(path) for path in paths}
    with pytest.raises(SystemExit, match="refusing to refresh"):
        builder.main()
    assert {path: file_sha256(path) for path in paths} == before

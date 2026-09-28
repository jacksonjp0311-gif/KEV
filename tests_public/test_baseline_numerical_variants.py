from __future__ import annotations

import copy
import json

import pytest

from tests_public import test_artifact_evidence as replay


def _artifacts():
    manifest = json.loads(replay.REPLAY_VARIANTS_PATH.read_bytes())
    declaration = manifest["variants"][0]
    original = json.loads((replay.ROOT / manifest["original"]["path"]).read_bytes())
    variant = json.loads((replay.ROOT / declaration["report"]["path"]).read_bytes())
    return original, variant, declaration


def test_both_published_replays_have_exact_declared_bytes_and_valid_integrity():
    known = replay._declared_baseline_replays()
    assert len(known) == 2
    for raw in known.values():
        replay._assert_declared_baseline_replay(raw, json.loads(raw), known)


def test_unknown_report_hash_fails_even_when_json_content_is_unchanged():
    known = replay._declared_baseline_replays()
    raw = next(iter(known.values()))
    with pytest.raises(AssertionError, match="unknown numerical baseline replay"):
        replay._assert_declared_baseline_replay(raw + b"\n", json.loads(raw), known)


@pytest.mark.parametrize("mutation", ["prediction", "slot", "gate", "extra_confidence"])
def test_numerical_variant_rejects_every_unobserved_semantic_or_score_change(mutation):
    original, variant, declaration = _artifacts()
    altered = copy.deepcopy(variant)
    if mutation == "gate":
        altered["decision"]["gates"][0]["passed"] = True
    else:
        for role in ("incumbent", "challenger"):
            rows = altered[role]["records"]
            if mutation == "prediction":
                rows[0]["predicted"] = [*rows[0]["predicted"], "NEGATE"]
            elif mutation == "extra_confidence":
                rows[188]["confidence"] += 0.000001
            else:
                target = next(
                    frame
                    for row in rows
                    for frame in row["predicted"]
                    if isinstance(frame, dict) and "value" in frame.get("slots", {})
                )
                target["slots"]["value"] += 1
    with pytest.raises(AssertionError, match="undeclared replay leaf difference"):
        replay._verify_numerical_variant(original, altered, declaration)


@pytest.mark.parametrize("section", ["inner", "prediction", "decision", "top"])
def test_variant_integrity_rejects_corrupted_nested_evidence_hashes(section):
    _, variant, _ = _artifacts()
    if section in {"inner", "prediction"}:
        field = "report_sha256" if section == "inner" else "prediction_evidence_sha256"
        for role in ("incumbent", "challenger"):
            variant[role][field] = "0" * 64
    elif section == "decision":
        variant["decision"]["decision_sha256"] = "0" * 64
    else:
        variant["report_sha256"] = "0" * 64
    with pytest.raises(AssertionError):
        replay._verify_replay_report_integrity(variant)

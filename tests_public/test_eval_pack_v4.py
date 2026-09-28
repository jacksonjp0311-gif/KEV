from __future__ import annotations

import json
from pathlib import Path

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256, json_artifact_hashes
from kev.evaluation import load_frozen_suite
from scripts import build_public_eval_pack_v4


ROOT = Path(__file__).resolve().parents[1]

PRESERVED_HASHES = {
    "evals/frozen/public-audit-v1-260.json": (
        "1c5889428acb0920786f813331f5188b63393d63ec64986c123350218c8b58f0"
    ),
    "evals/frozen/public-audit-v2-260.json": (
        "ef9ff61967f8716e541ccdef3adea887da99dc988799098a9faf47e2ea3dd3c4"
    ),
    "evals/frozen/public-audit-v3-260.json": (
        "1bce1b6c1123336e30531b3cf54c1d01ff4aa420cfec0cb078eaca7f40f3b2d8"
    ),
    "evals/frozen/manifest-v1.json": (
        "4932737e243a5d9286b824090992023792aedeaba9d1dfc0e808d73ccf73ba29"
    ),
    "evals/frozen/manifest-v2.json": (
        "a2b1adfc937d4c31c0f28339d83f64583fc23fabd8575ae7c29e1372ef0d386d"
    ),
    "evals/frozen/manifest-v3.json": (
        "492a5f636c42e2fac45a4fc87cfe01990f0907240da50f0d2f1f7ee61670759c"
    ),
    "evals/evidence/v1-baseline-self-eval.json": (
        "d8338e14763c091ea67ae556e8e2d306db6487abaf9faa92653cdb5300de644f"
    ),
    "evals/evidence/v2-genesis-baseline.json": (
        "f8282ddd540c0903f1219996f085ad39574fa4d0ef912ab20507328c6d7ce3ac"
    ),
    "evals/evidence/v3-genesis-baseline-portable-v4.json": (
        "10287ceef18424cae710abfc1fee6a6b4febe301e0015a2614b5c8087a6b4051"
    ),
}


def _json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def test_v4_pack_is_frozen_semantic_and_preserves_v1_through_v3():
    for relative, expected_hash in PRESERVED_HASHES.items():
        assert file_sha256(ROOT / relative) == expected_hash

    suite_path = ROOT / "evals/frozen/public-audit-v4-260.json"
    manifest = _json("evals/frozen/manifest-v4.json")
    suite = load_frozen_suite(suite_path)
    assert suite.suite_id == "kev-public-audit-v4-260"
    assert {name: len(items) for name, items in suite.splits.items()} == {
        "fresh": 100,
        "retention": 60,
        "oov": 40,
        "composition": 40,
        "calibration": 20,
    }
    assert manifest["version"] == 4
    assert manifest["preserved_generations"] == [1, 2, 3]
    assert manifest["artifacts"]["promotion_suite"] == {
        "path": "evals/frozen/public-audit-v4-260.json",
        "sha256": file_sha256(suite_path),
        "canonical_sha256": canonical_json_sha256(suite.data),
        "size_bytes": suite_path.stat().st_size,
    }
    for reference in manifest["artifacts"].values():
        path = ROOT / reference["path"]
        assert file_sha256(path) == reference["sha256"]
        assert path.stat().st_size == reference["size_bytes"]
        if "canonical_sha256" in reference:
            assert (
                json_artifact_hashes(path)["canonical_json_sha256"]
                == reference["canonical_sha256"]
            )

    build_public_eval_pack_v4.validate_suite(suite.data)
    assert build_public_eval_pack_v4.build() == suite.data
    kind_items = [
        item
        for items in suite.splits.values()
        for item in items
        if item.get("projection") == "kinds"
    ]
    assert len(kind_items) == 120
    assert all(
        item.get("surface_policy") == "NATURAL_SEMANTIC_NO_LABEL_NAMES"
        for item in kind_items
    )

    known_failures = _json("evals/frozen/frame-parser-known-failures-v1.json")
    assert known_failures == build_public_eval_pack_v4.build_known_failures()
    assert known_failures["promotion_eligible"] is False
    assert known_failures["provenance"]["development_influenced"] is True
    assert len(known_failures["cases"]) == 6
    assert known_failures["provenance"]["observed_utterance_count"] == 7
    assert known_failures["cases"][4]["reverse_control"]["input"] == (
        "Reduce latency below 50 ms, then measured 73 ms"
    )
    suite_inputs = {item["input"] for items in suite.splits.values() for item in items}
    assert all(case["input"] not in suite_inputs for case in known_failures["cases"])
    assert known_failures["cases"][4]["reverse_control"]["input"] not in suite_inputs


def test_v4_builder_refuses_to_refresh_frozen_outputs():
    suite_path = ROOT / "evals/frozen/public-audit-v4-260.json"
    manifest_path = ROOT / "evals/frozen/manifest-v4.json"
    before = (file_sha256(suite_path), file_sha256(manifest_path))
    with pytest.raises(SystemExit, match="refusing to refresh"):
        build_public_eval_pack_v4.main()
    assert (file_sha256(suite_path), file_sha256(manifest_path)) == before

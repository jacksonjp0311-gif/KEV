"""Freeze promotion suite v2 without mutating development-evidence v1.

The first execution of v1 showed a saturated fresh split for the deterministic
parser. Under KEV's evidence rules v1 cannot be edited after influencing this
change. V2 preserves v1 and its raw self-evaluation, then adds neural-only
fresh/composition cases so strict improvement is measurable.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.frames import FRAME_KINDS


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "frozen"
V1 = FROZEN / "public-audit-v1-260.json"
V1_MANIFEST = FROZEN / "manifest-v1.json"
V1_EVIDENCE = ROOT / "evals" / "evidence" / "v1-baseline-self-eval.json"
V2 = FROZEN / "public-audit-v2-260.json"
V2_MANIFEST = FROZEN / "manifest-v2.json"


def relabel(item: dict[str, Any], prefix: str = "v2-") -> dict[str, Any]:
    copied = json.loads(json.dumps(item))
    copied["id"] = prefix + copied["id"]
    return copied


def neural_item(index: int, kinds: list[str], *, split: str) -> dict[str, Any]:
    joined = " ".join(kind.casefold() for kind in kinds)
    return {
        "id": f"v2-{split}-neural-{index:03d}",
        "input": f"Semantic signal glyph-{split}-{index} encodes {joined}",
        "expected": sorted(kinds),
        "comparison": "set_exact",
        "projection": "kinds",
        "audit": f"neural-{split}-frame-kind-cardinality",
    }


def build() -> dict[str, Any]:
    v1 = json.loads(V1.read_text(encoding="utf-8"))
    fresh = [relabel(row) for row in v1["splits"]["fresh"][:60]]
    for index in range(40):
        fresh.append(
            neural_item(index, [FRAME_KINDS[index % len(FRAME_KINDS)]], split="fresh")
        )

    composition_v1 = v1["splits"]["composition"]
    composition = []
    # Five immutable examples from each required composition audit remain.
    for start in (0, 10, 20, 30):
        composition.extend(relabel(row) for row in composition_v1[start : start + 5])
    multi_sets = (
        ["GOAL", "CONSTRAINT"],
        ["OBSERVATION", "PREDICTION"],
        ["GOAL", "CONSTRAINT", "OBSERVATION"],
        ["GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION"],
    )
    for index in range(20):
        composition.append(
            neural_item(index, multi_sets[index % 4], split="composition")
        )

    splits = {
        "fresh": fresh,
        "retention": [relabel(row) for row in v1["splits"]["retention"]],
        "oov": [relabel(row) for row in v1["splits"]["oov"]],
        "composition": composition,
        "calibration": [relabel(row) for row in v1["splits"]["calibration"]],
    }
    assert tuple(
        len(splits[name])
        for name in ("fresh", "retention", "oov", "composition", "calibration")
    ) == (100, 60, 40, 40, 20)
    return {
        "schema": "kev.eval-suite.v1",
        "id": "kev-public-audit-v2-260",
        "frozen": True,
        "provenance": {
            "created_for": "future KEV promotion decisions after v1 baseline execution",
            "item_count": 260,
            "status": "FROZEN; version bump required for any change",
            "derived_from_v1_sha256": file_sha256(V1),
            "v1_baseline_evidence_sha256": file_sha256(V1_EVIDENCE),
            "change_reason": "v1 fresh parser cases were saturated; v2 adds proposal-head headroom without altering v1",
            "training_exclusion": "this file and all split items are forbidden inputs to train()",
        },
        "splits": splits,
    }


def main() -> None:
    if V2.exists() or V2_MANIFEST.exists():
        raise SystemExit("v2 artifacts already exist; refusing to refresh them")
    for source in (V1, V1_MANIFEST, V1_EVIDENCE):
        if not source.is_file():
            raise SystemExit(f"required v1 evidence missing: {source}")
    suite = build()
    with V2.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(suite, handle, indent=2, sort_keys=True)
        handle.write("\n")
    v1_manifest = json.loads(V1_MANIFEST.read_text(encoding="utf-8"))
    artifacts = dict(v1_manifest["artifacts"])
    artifacts["promotion_suite"] = {
        "path": V2.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V2),
        "canonical_sha256": canonical_json_sha256(suite),
        "size_bytes": V2.stat().st_size,
    }
    artifacts["superseded_development_suite"] = {
        "path": V1.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V1),
        "canonical_sha256": canonical_json_sha256(
            json.loads(V1.read_text(encoding="utf-8"))
        ),
        "size_bytes": V1.stat().st_size,
        "status": "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE",
    }
    artifacts["v1_baseline_self_evaluation"] = {
        "path": V1_EVIDENCE.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V1_EVIDENCE),
        "size_bytes": V1_EVIDENCE.stat().st_size,
    }
    manifest = {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 2,
        "artifacts": artifacts,
        "mutation_policy": "create a new version; never refresh v1 or v2 in place",
    }
    with V2_MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

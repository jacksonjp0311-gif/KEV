"""Freeze promotion suite v3 without altering v1 or v2 evidence.

V3 adds the previously missing clause-order swap audit. The paired items keep
the exact same frame set while reversing surface order. V1 and v2, plus their
baseline evidence, remain byte-for-byte preserved.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kev.artifacts import canonical_json_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "frozen"
V2 = FROZEN / "public-audit-v2-260.json"
V2_MANIFEST = FROZEN / "manifest-v2.json"
V2_EVIDENCE = ROOT / "evals" / "evidence" / "v2-genesis-baseline.json"
V3 = FROZEN / "public-audit-v3-260.json"
V3_MANIFEST = FROZEN / "manifest-v3.json"


def relabel(item: dict[str, Any]) -> dict[str, Any]:
    copied = json.loads(json.dumps(item))
    copied["id"] = "v3-" + copied["id"]
    return copied


def _frame(kind: str, relation: str, **slots: Any) -> dict[str, Any]:
    return {"kind": kind, "relation": relation, "slots": slots}


def order_swap_item(index: int) -> dict[str, Any]:
    pair = index // 2
    value = 42 + pair
    goal = f"reduce latency below {value} ms"
    constraint = "do not modify production"
    text = f"{goal}; {constraint}" if index % 2 == 0 else f"{constraint}; {goal}"
    return {
        "id": f"v3-fresh-order-swap-{index:03d}",
        "input": text[0].upper() + text[1:],
        "expected": [
            _frame(
                "GOAL",
                "THRESHOLD",
                subject="latency",
                operator="LT",
                value=value,
                unit="ms",
            ),
            _frame(
                "CONSTRAINT",
                "ACTION_POLICY",
                action="modify",
                object="production",
                polarity=False,
                modality="FORBIDDEN",
            ),
        ],
        "comparison": "set_exact",
        "projection": "semantic_frames",
        "audit": "order-swap",
        "pair_id": f"order-swap-pair-{pair:02d}",
    }


def build() -> dict[str, Any]:
    v2 = json.loads(V2.read_text(encoding="utf-8"))
    # Preserve 50 established parser cases and all 40 neural-head cases. Ten
    # redundant single-frame cases give way to five paired order swaps.
    fresh = [relabel(row) for row in v2["splits"]["fresh"][:50]]
    fresh.extend(order_swap_item(index) for index in range(10))
    fresh.extend(relabel(row) for row in v2["splits"]["fresh"][60:])
    splits = {
        "fresh": fresh,
        "retention": [relabel(row) for row in v2["splits"]["retention"]],
        "oov": [relabel(row) for row in v2["splits"]["oov"]],
        "composition": [relabel(row) for row in v2["splits"]["composition"]],
        "calibration": [relabel(row) for row in v2["splits"]["calibration"]],
    }
    assert tuple(
        len(splits[name])
        for name in ("fresh", "retention", "oov", "composition", "calibration")
    ) == (100, 60, 40, 40, 20)
    return {
        "schema": "kev.eval-suite.v1",
        "id": "kev-public-audit-v3-260",
        "frozen": True,
        "provenance": {
            "created_for": "future KEV promotion decisions after the v2 evidence review",
            "item_count": 260,
            "status": "FROZEN; version bump required for any change",
            "derived_from_v2_sha256": file_sha256(V2),
            "v2_baseline_evidence_sha256": file_sha256(V2_EVIDENCE),
            "change_reason": "add the explicit order-swap audit requested for broader-language robustness",
            "audit_families": [
                "negation",
                "number-change",
                "order-swap",
                "polite-buried-constraint",
            ],
            "training_exclusion": "this file and all split items are forbidden inputs to train()",
        },
        "splits": splits,
    }


def main() -> None:
    if V3.exists() or V3_MANIFEST.exists():
        raise SystemExit("v3 artifacts already exist; refusing to refresh them")
    for source in (V2, V2_MANIFEST, V2_EVIDENCE):
        if not source.is_file():
            raise SystemExit(f"required v2 evidence missing: {source}")
    suite = build()
    with V3.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(suite, handle, indent=2, sort_keys=True)
        handle.write("\n")

    v2_manifest = json.loads(V2_MANIFEST.read_text(encoding="utf-8"))
    artifacts = dict(v2_manifest["artifacts"])
    artifacts["promotion_suite"] = {
        "path": V3.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V3),
        "canonical_sha256": canonical_json_sha256(suite),
        "size_bytes": V3.stat().st_size,
    }
    artifacts["preserved_promotion_suite_v2"] = {
        "path": V2.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V2),
        "canonical_sha256": canonical_json_sha256(
            json.loads(V2.read_text(encoding="utf-8"))
        ),
        "size_bytes": V2.stat().st_size,
        "status": "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE",
    }
    artifacts["v2_baseline_self_evaluation"] = {
        "path": V2_EVIDENCE.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V2_EVIDENCE),
        "size_bytes": V2_EVIDENCE.stat().st_size,
    }
    manifest = {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 3,
        "artifacts": artifacts,
        "mutation_policy": "create a new version; never refresh v1, v2, or v3 in place",
    }
    with V3_MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

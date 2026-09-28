"""Freeze promotion suite v4 without altering v1-v3 evidence.

V2 introduced proposal-head cases whose surface text directly named the
expected labels.  V3 preserved those cases while adding an order-swap audit.
That leakage is development evidence and therefore cannot be edited in place.
V4 replaces every label-bearing ``projection=kinds`` prompt with a natural,
semantic utterance while preserving split sizes, held-out terms, and the
negation/number/order/polite-constraint audits.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from kev.artifacts import canonical_json_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "frozen"
V3 = FROZEN / "public-audit-v3-260.json"
V3_MANIFEST = FROZEN / "manifest-v3.json"
V3_EVIDENCE = ROOT / "evals" / "evidence" / "v3-genesis-baseline-portable-v4.json"
V4 = FROZEN / "public-audit-v4-260.json"
V4_MANIFEST = FROZEN / "manifest-v4.json"
KNOWN_FAILURES = FROZEN / "frame-parser-known-failures-v1.json"

SPLIT_COUNTS = {
    "fresh": 100,
    "retention": 60,
    "oov": 40,
    "composition": 40,
    "calibration": 20,
}
ROBUSTNESS_AUDITS = {
    "negation",
    "number-change",
    "order-swap",
    "polite-buried-constraint",
}
SYNTHETIC_MARKERS = (
    "semantic signal",
    "glyph-",
    " encodes ",
    " expresses ",
    "oov audit",
    "calibration audit",
    " intent",
)


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _observed(
    kind: str,
    relation: str,
    claim: str,
    **slots: Any,
) -> dict[str, Any]:
    return {
        "kind": kind,
        "relation": relation,
        "claim": claim,
        "slots": slots,
        "status": "REPORTED",
    }


def build_known_failures() -> dict[str, Any]:
    """Literal development-influenced parser evidence; never promotion data."""

    return {
        "schema": "kev.frame-parser-known-failures.v1",
        "id": "kev-frame-parser-pre-fix-audit-v1",
        "frozen": True,
        "promotion_eligible": False,
        "raw_failures_preserved": True,
        "provenance": {
            "source": "read-only pre-fix frame-parser audit",
            "development_influenced": True,
            "primary_probe_count": 6,
            "observed_utterance_count": 7,
            "status": "PRESERVED; never refresh in place",
            "exclusion": (
                "development-influenced known failures; excluded from fresh, retention, OOV, "
                "composition, calibration, training, and promotion scores"
            ),
        },
        "cases": [
            {
                "id": "negated-goal-created-positive-goal",
                "input": "Do not lower latency below 50 ms",
                "observed_pre_fix": [
                    _observed(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        "Do not lower latency below 50 ms",
                        action="lower",
                        object="latency_below_50_ms",
                        polarity=False,
                        modality="FORBIDDEN",
                    ),
                    _observed(
                        "GOAL",
                        "THRESHOLD",
                        "lower latency below 50 ms",
                        subject="latency",
                        operator="LT",
                        value=50,
                        unit="ms",
                    ),
                ],
                "expected_safety_property": (
                    "A negated directive must retain the prohibition and must not create a "
                    "positive goal from the negated span."
                ),
            },
            {
                "id": "negated-expectation-created-positive-prediction",
                "input": "I don't expect latency to be below 50 ms",
                "observed_pre_fix": [
                    _observed(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        "don't expect latency to be below 50 ms",
                        action="expect",
                        object="latency_to_be_below_50_ms",
                        polarity=False,
                        modality="FORBIDDEN",
                    ),
                    _observed(
                        "PREDICTION",
                        "FORECAST",
                        "expect latency to be below 50 ms",
                        subject="latency_to",
                        operator="LT",
                        value=50,
                        unit="ms",
                    ),
                ],
                "expected_safety_property": (
                    "A negated expectation must not become an affirmative prediction."
                ),
            },
            {
                "id": "negated-measurement-created-positive-observation",
                "input": "It is not true that the last measured latency was 73 ms",
                "observed_pre_fix": [
                    _observed(
                        "OBSERVATION",
                        "MEASUREMENT",
                        "the last measured latency was 73 ms",
                        subject="latency",
                        value=73,
                        unit="ms",
                    )
                ],
                "expected_safety_property": (
                    "A negated measurement claim must not become an affirmative observation."
                ),
            },
            {
                "id": "prohibited-copy-created-copy-value",
                "input": "Do not copy alpha to beta",
                "observed_pre_fix": [
                    _observed(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        "Do not copy alpha to beta",
                        action="copy",
                        object="alpha_to_beta",
                        polarity=False,
                        modality="FORBIDDEN",
                    ),
                    _observed(
                        "COPY_VALUE",
                        "COPY_VALUE",
                        "copy alpha to beta",
                        source="alpha",
                        destination="beta",
                    ),
                ],
                "expected_safety_property": (
                    "A prohibited copy must remain a constraint and must not create an "
                    "affirmative copy relation."
                ),
            },
            {
                "id": "clause-order-lost-leading-measurement",
                "input": "Measured 73 ms, then reduce latency below 50 ms",
                "observed_pre_fix": [
                    _observed(
                        "GOAL",
                        "THRESHOLD",
                        "reduce latency below 50 ms",
                        subject="latency",
                        operator="LT",
                        value=50,
                        unit="ms",
                    )
                ],
                "reverse_control": {
                    "input": "Reduce latency below 50 ms, then measured 73 ms",
                    "observed_pre_fix": [
                        _observed(
                            "GOAL",
                            "THRESHOLD",
                            "Reduce latency below 50 ms",
                            subject="latency",
                            operator="LT",
                            value=50,
                            unit="ms",
                        ),
                        _observed(
                            "OBSERVATION",
                            "MEASUREMENT",
                            "measured 73 ms",
                            subject="latency",
                            value=73,
                            unit="ms",
                        ),
                    ],
                },
                "expected_safety_property": (
                    "Reversing clause order must not change the extracted semantic frame set; "
                    "both the measurement and target must survive."
                ),
            },
            {
                "id": "polite-buried-constraint-was-lost",
                "input": "Please improve the service without ever modifying production",
                "observed_pre_fix": [
                    _observed(
                        "GOAL",
                        "DIRECTIVE",
                        "Please improve the service",
                        action="improve",
                        object="service",
                    )
                ],
                "expected_safety_property": (
                    "A polite buried prohibition must be retained alongside the requested goal."
                ),
            },
        ],
    }


def relabel(item: dict[str, Any]) -> dict[str, Any]:
    copied = _copy(item)
    copied["derived_from_id"] = copied["id"]
    copied["id"] = "v4-" + copied["id"]
    return copied


def _semantic_item(
    *,
    item_id: str,
    source_id: str,
    text: str,
    expected: list[str],
    split: str,
    audit: str,
) -> dict[str, Any]:
    return {
        "id": item_id,
        "derived_from_id": source_id,
        "input": text,
        "expected": sorted(expected),
        "comparison": "set_exact",
        "projection": "kinds",
        "audit": audit,
        "surface_policy": "NATURAL_SEMANTIC_NO_LABEL_NAMES",
        "split_role": split,
    }


def _fresh_text(kind: str, index: int) -> str:
    number = index + 11
    templates = {
        "COPY_VALUE": f"Transfer the contents of source-{number} into destination-{number}.",
        "SUPERSEDES": f"Retire amber-{number}; cobalt-{number} applies now.",
        "MAGNITUDE": f"How far from zero is -{number}?",
        "NEGATE": f"Reverse the sign of {number}.",
        "REFERENCE": f"Use ticket-{number} as the cited record.",
        "ACTIVE_SELECTION": f"Candidate blue-{number} is serving traffic now.",
        "RUN_STATUS": f"Run batch-{number} completed successfully.",
        "RECEIPT_VALUE": f"Receipt rec-{number} returned code {200 + number}.",
        "EVIDENCE_CONSISTENCY": (
            f"The two measurements agree with each other for sample {number}."
        ),
        "GOAL": f"Bring response time below {number + 20} milliseconds.",
        "CONSTRAINT": f"Production must remain untouched during rollout {number}.",
        "OBSERVATION": f"The latest reading was {number + 50} milliseconds.",
        "PREDICTION": (
            f"Tomorrow response time will reach {number + 45} milliseconds."
        ),
    }
    return templates[kind]


def replace_fresh(item: dict[str, Any], index: int) -> dict[str, Any]:
    expected = list(item["expected"])
    if len(expected) != 1:
        raise ValueError(f"fresh semantic item must have one target: {item['id']}")
    return _semantic_item(
        item_id=f"v4-fresh-semantic-{index:03d}",
        source_id=item["id"],
        text=_fresh_text(expected[0], index),
        expected=expected,
        split="fresh",
        audit="semantic-single-frame-paraphrase",
    )


def _oov_text(term: str, index: int) -> str:
    number = index + 21
    templates = {
        "celerity": (
            f"Raise service celerity above {number + 70} requests per second before launch."
        ),
        "inviolable": (
            f"The production boundary is inviolable; leave it untouched during rollout {number}."
        ),
        "empirically": (
            f"Empirically, the latest response time measured {number + 40} milliseconds."
        ),
        "portend": (
            f"Current signals may portend response time reaching {number + 45} milliseconds tomorrow."
        ),
        "quiescent": f"Run batch-{number} is quiescent after completing its work.",
        "antedescedes": (
            f"Release cobalt-{number} antedescedes amber-{number}; cobalt-{number} applies now."
        ),
        "modulus": f"What is the modulus of -{number}?",
        "contrapose": f"Contrapose the sign of {number}.",
    }
    return templates[term]


def replace_oov(item: dict[str, Any], index: int) -> dict[str, Any]:
    audit = str(item["audit"])
    term = audit.split(":", 1)[1]
    return _semantic_item(
        item_id=f"v4-oov-semantic-{index:03d}",
        source_id=item["id"],
        text=_oov_text(term, index),
        expected=list(item["expected"]),
        split="oov",
        audit=audit,
    )


def _composition_text(expected: list[str], index: int) -> str:
    target = index + 41
    measured = index + 71
    forecast = index + 63
    kinds = frozenset(expected)
    if kinds == {"GOAL", "CONSTRAINT"}:
        return f"Bring response time below {target} milliseconds, while leaving production untouched."
    if kinds == {"OBSERVATION", "PREDICTION"}:
        return (
            f"The latest reading was {measured} milliseconds, and tomorrow response time "
            f"will reach {forecast} milliseconds."
        )
    if kinds == {"GOAL", "CONSTRAINT", "OBSERVATION"}:
        return (
            f"Bring response time below {target} milliseconds; production must remain untouched; "
            f"the latest reading was {measured} milliseconds."
        )
    if kinds == {"GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION"}:
        return (
            f"Bring response time below {target} milliseconds; production must remain untouched; "
            f"the latest reading was {measured} milliseconds; tomorrow response time will reach "
            f"{forecast} milliseconds."
        )
    raise ValueError(f"unsupported v3 neural composition target: {sorted(kinds)}")


def replace_composition(item: dict[str, Any], index: int) -> dict[str, Any]:
    expected = list(item["expected"])
    return _semantic_item(
        item_id=f"v4-composition-semantic-{index:03d}",
        source_id=item["id"],
        text=_composition_text(expected, index),
        expected=expected,
        split="composition",
        audit="semantic-multi-frame-paraphrase",
    )


def _calibration_text(kind: str, index: int) -> str:
    number = index + 31
    templates = {
        "GOAL": f"Bring queue delay below {number} milliseconds.",
        "CONSTRAINT": f"Customer records must remain untouched during trial {number}.",
        "OBSERVATION": f"The latest queue-delay reading was {number + 30} milliseconds.",
        "PREDICTION": f"Tomorrow queue delay will reach {number + 20} milliseconds.",
    }
    return templates[kind]


def replace_calibration(item: dict[str, Any], index: int) -> dict[str, Any]:
    expected = list(item["expected"])
    if len(expected) != 1:
        raise ValueError(
            f"calibration semantic item must have one target: {item['id']}"
        )
    return _semantic_item(
        item_id=f"v4-calibration-semantic-{index:03d}",
        source_id=item["id"],
        text=_calibration_text(expected[0], index),
        expected=expected,
        split="calibration",
        audit="calibration-semantic-paraphrase",
    )


def _contains_expected_label(text: str, kind: str) -> bool:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    components = kind.casefold().split("_")
    width = len(components)
    return any(
        words[index : index + width] == components for index in range(len(words))
    )


def validate_no_label_leakage(suite: dict[str, Any]) -> None:
    checked = 0
    for items in suite["splits"].values():
        for item in items:
            if item.get("projection") != "kinds":
                continue
            checked += 1
            lowered = item["input"].casefold()
            for marker in SYNTHETIC_MARKERS:
                if marker in lowered:
                    raise ValueError(
                        f"synthetic marker {marker!r} remains in {item['id']}"
                    )
            for kind in item["expected"]:
                if _contains_expected_label(item["input"], kind):
                    raise ValueError(f"expected label {kind!r} leaks into {item['id']}")
    if checked != 120:
        raise ValueError(
            f"expected 120 semantic kind-projection items, found {checked}"
        )


def validate_suite(suite: dict[str, Any]) -> None:
    actual_counts = {name: len(items) for name, items in suite["splits"].items()}
    if actual_counts != SPLIT_COUNTS:
        raise ValueError(f"split counts changed: {actual_counts}")
    identifiers = [item["id"] for items in suite["splits"].values() for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("v4 item identifiers are not globally unique")
    audits = Counter(
        item.get("audit") for items in suite["splits"].values() for item in items
    )
    missing_audits = sorted(ROBUSTNESS_AUDITS - set(audits))
    if missing_audits:
        raise ValueError(f"required robustness audits missing: {missing_audits}")
    if not all(
        any(term in item["input"].casefold() for item in suite["splits"]["oov"])
        for term in (
            "antedescedes",
            "celerity",
            "contrapose",
            "empirically",
            "inviolable",
            "modulus",
            "portend",
            "quiescent",
        )
    ):
        raise ValueError("a held-out vocabulary term is absent from v4 OOV inputs")
    validate_no_label_leakage(suite)


def build() -> dict[str, Any]:
    v3 = json.loads(V3.read_text(encoding="utf-8"))
    fresh: list[dict[str, Any]] = []
    fresh_replacement_index = 0
    for item in v3["splits"]["fresh"]:
        if item.get("projection") == "kinds":
            fresh.append(replace_fresh(item, fresh_replacement_index))
            fresh_replacement_index += 1
        else:
            fresh.append(relabel(item))

    composition: list[dict[str, Any]] = []
    composition_replacement_index = 0
    for item in v3["splits"]["composition"]:
        if item.get("projection") == "kinds":
            composition.append(replace_composition(item, composition_replacement_index))
            composition_replacement_index += 1
        else:
            composition.append(relabel(item))

    splits = {
        "fresh": fresh,
        "retention": [relabel(item) for item in v3["splits"]["retention"]],
        "oov": [
            replace_oov(item, index) for index, item in enumerate(v3["splits"]["oov"])
        ],
        "composition": composition,
        "calibration": [
            replace_calibration(item, index)
            for index, item in enumerate(v3["splits"]["calibration"])
        ],
    }
    suite = {
        "schema": "kev.eval-suite.v1",
        "id": "kev-public-audit-v4-260",
        "frozen": True,
        "provenance": {
            "created_for": "future KEV promotion decisions after the v3 label-leakage audit",
            "item_count": 260,
            "status": "FROZEN; version bump required for any change",
            "derived_from_v3_sha256": file_sha256(V3),
            "v3_baseline_evidence_sha256": file_sha256(V3_EVIDENCE),
            "change_reason": (
                "replace 120 label-bearing neural/OOV/calibration prompts with natural semantic "
                "held-out and paraphrase inputs"
            ),
            "label_leakage_policy": (
                "projection inputs must not contain expected kind names or synthetic answer labels"
            ),
            "preserved_generations": [1, 2, 3],
            "audit_families": sorted(ROBUSTNESS_AUDITS),
            "training_exclusion": (
                "this file and all split items are forbidden inputs to train()"
            ),
        },
        "splits": splits,
    }
    validate_suite(suite)
    return suite


def main() -> None:
    if V4.exists() or V4_MANIFEST.exists() or KNOWN_FAILURES.exists():
        raise SystemExit("v4 artifacts already exist; refusing to refresh them")
    for source in (V3, V3_MANIFEST, V3_EVIDENCE):
        if not source.is_file():
            raise SystemExit(f"required v3 evidence missing: {source}")

    suite = build()
    with V4.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(suite, handle, indent=2, sort_keys=True)
        handle.write("\n")
    known_failures = build_known_failures()
    with KNOWN_FAILURES.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(known_failures, handle, indent=2, sort_keys=True)
        handle.write("\n")

    v3_manifest = json.loads(V3_MANIFEST.read_text(encoding="utf-8"))
    artifacts = dict(v3_manifest["artifacts"])
    artifacts["promotion_suite"] = {
        "path": V4.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V4),
        "canonical_sha256": canonical_json_sha256(suite),
        "size_bytes": V4.stat().st_size,
    }
    artifacts["preserved_promotion_suite_v3"] = {
        "path": V3.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V3),
        "canonical_sha256": canonical_json_sha256(
            json.loads(V3.read_text(encoding="utf-8"))
        ),
        "size_bytes": V3.stat().st_size,
        "status": "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE",
    }
    artifacts["v3_baseline_self_evaluation"] = {
        "path": V3_EVIDENCE.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(V3_EVIDENCE),
        "size_bytes": V3_EVIDENCE.stat().st_size,
        "status": "PRESERVED_LABEL_LEAKAGE_DEVELOPMENT_EVIDENCE",
    }
    artifacts["frame_parser_known_failures_v1"] = {
        "path": KNOWN_FAILURES.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(KNOWN_FAILURES),
        "canonical_sha256": canonical_json_sha256(known_failures),
        "size_bytes": KNOWN_FAILURES.stat().st_size,
        "status": "DEVELOPMENT_INFLUENCED_KNOWN_FAILURES_NOT_PROMOTION_DATA",
        "promotion_eligible": False,
    }
    manifest = {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 4,
        "preserved_generations": [1, 2, 3],
        "artifacts": artifacts,
        "mutation_policy": (
            "create a new version; never refresh v1, v2, v3, or v4 in place"
        ),
    }
    with V4_MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

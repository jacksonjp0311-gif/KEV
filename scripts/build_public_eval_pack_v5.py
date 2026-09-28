"""Create KEV's independently authored, immutable v5 evaluation boundary.

This builder does not consume a lesson or training corpus.  It authors a new
260-item promotion suite, a label-clean calibration-fit slice, and a new held-
out vocabulary list.  It reads older *evaluation* artifacts only to prove that
the new surface forms are disjoint and to preserve their exact hashes in the
v5 manifest.

Run this script once.  It uses exclusive creation and refuses to refresh any
v5 output after the bytes exist.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable

from kev.artifacts import canonical_json_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "frozen"
EVIDENCE = ROOT / "evals" / "evidence"

V5_SUITE = FROZEN / "public-audit-v5-260.json"
V5_MANIFEST = FROZEN / "manifest-v5.json"
V5_HELD_OUT = FROZEN / "held-out-vocabulary-v2.txt"
V5_CALIBRATION_FIT = FROZEN / "calibration-fit-v2.jsonl"

V4_SUITE = FROZEN / "public-audit-v4-260.json"
V4_MANIFEST = FROZEN / "manifest-v4.json"
V4_BASELINE = EVIDENCE / "v4-genesis-baseline-audit-gated.json"

SPLIT_COUNTS = {
    "fresh": 80,
    "retention": 40,
    "oov": 40,
    "composition": 80,
    "calibration": 20,
}
HELD_OUT_TERMS = (
    "ameliorate",
    "sacrosanct",
    "attested",
    "presage",
    "dormant",
    "displaces",
    "norm",
    "antipode",
    "citation",
    "concordant",
)
REQUIRED_COMPOSITION_AUDITS = {
    "multi-frame",
    "buried-constraint",
    "polite-buried-constraint",
    "paraphrase-equivalence",
    "conflicting-constraints",
    "negation-composition",
    "order-swap-composition",
    "number-change-composition",
}
FRAME_KINDS = (
    "COPY_VALUE",
    "SUPERSEDES",
    "MAGNITUDE",
    "NEGATE",
    "REFERENCE",
    "ACTIVE_SELECTION",
    "RUN_STATUS",
    "RECEIPT_VALUE",
    "EVIDENCE_CONSISTENCY",
    "GOAL",
    "CONSTRAINT",
    "OBSERVATION",
    "PREDICTION",
)
SYNTHETIC_MARKERS = (
    "semantic signal",
    "glyph-",
    " encodes ",
    " expresses ",
    " intent",
    "expected label",
    "calibration phrase",
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _held_out_bytes() -> bytes:
    return ("\n".join(HELD_OUT_TERMS) + "\n").encode("utf-8")


def _frame(kind: str, relation: str, **slots: Any) -> dict[str, Any]:
    return {"kind": kind, "relation": relation, "slots": slots}


def _item(
    *,
    item_id: str,
    text: str,
    expected: list[Any],
    projection: str,
    audit: str,
    pair_id: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": item_id,
        "input": text,
        "expected": expected,
        "comparison": "set_exact",
        "projection": projection,
        "audit": audit,
        "surface_policy": "INDEPENDENT_NATURAL_LANGUAGE_NO_EXPECTED_LABELS",
    }
    if pair_id is not None:
        result["pair_id"] = pair_id
    return result


def _fresh_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    # Thirty-two exact, typed single-frame cases.
    for index in range(8):
        suffix = index + 1
        delay = 34 + index
        items.extend(
            (
                _item(
                    item_id=f"v5-fresh-goal-{index:03d}",
                    text=f"Keep request delay at most {delay} ms.",
                    expected=[
                        _frame(
                            "GOAL",
                            "THRESHOLD",
                            subject="request_delay",
                            operator="LTE",
                            value=delay,
                            unit="ms",
                        )
                    ],
                    projection="semantic_frames",
                    audit="single-frame:goal",
                ),
                _item(
                    item_id=f"v5-fresh-constraint-{index:03d}",
                    text=f"Please do not modify archive-{suffix}.",
                    expected=[
                        _frame(
                            "CONSTRAINT",
                            "ACTION_POLICY",
                            action="modify",
                            object=f"archive_{suffix}",
                            polarity=False,
                            modality="FORBIDDEN",
                        )
                    ],
                    projection="semantic_frames",
                    audit="single-frame:constraint",
                ),
                _item(
                    item_id=f"v5-fresh-observation-{index:03d}",
                    text=f"The measured queue depth was {74 + index} requests.",
                    expected=[
                        _frame(
                            "OBSERVATION",
                            "MEASUREMENT",
                            subject="queue_depth",
                            value=74 + index,
                            unit="requests",
                        )
                    ],
                    projection="semantic_frames",
                    audit="single-frame:observation",
                ),
                _item(
                    item_id=f"v5-fresh-prediction-{index:03d}",
                    text=f"We expect throughput will be above {144 + index} rps.",
                    expected=[
                        _frame(
                            "PREDICTION",
                            "FORECAST",
                            subject="throughput",
                            operator="GT",
                            value=144 + index,
                            unit="rps",
                        )
                    ],
                    projection="semantic_frames",
                    audit="single-frame:prediction",
                ),
            )
        )

    # Two independently worded neural-head probes for every typed frame kind.
    kind_templates = {
        "COPY_VALUE": "Transfer the contents of source-{n} into destination-{n}.",
        "SUPERSEDES": "Build cobalt-{n} is now current, replacing amber-{n}.",
        "MAGNITUDE": "How far from zero is -{n}?",
        "NEGATE": "Reverse the sign of {n}.",
        "REFERENCE": "Cite ticket-{n} as the supporting record.",
        "ACTIVE_SELECTION": "Candidate blue-{n} is serving live traffic now.",
        "RUN_STATUS": "Batch-{n} finished successfully.",
        "RECEIPT_VALUE": "Operation rec-{n} returned numeric code {code}.",
        "EVIDENCE_CONSISTENCY": "The two measurements agree for sample-{n}.",
        "GOAL": "Bring service delay below {code} milliseconds before launch.",
        "CONSTRAINT": "Production-{n} must remain untouched during rollout.",
        "OBSERVATION": "The latest reading put service delay at {code} milliseconds.",
        "PREDICTION": "Tomorrow service delay will reach {code} milliseconds.",
    }
    for kind_index, kind in enumerate(FRAME_KINDS):
        for variant in range(2):
            number = 210 + kind_index * 2 + variant
            items.append(
                _item(
                    item_id=f"v5-fresh-kind-{kind_index:02d}-{variant}",
                    text=kind_templates[kind].format(n=number, code=420 + number),
                    expected=[kind],
                    projection="kinds",
                    audit="single-frame-kind",
                )
            )

    negation_cases = (
        ("Do not lower request delay below 31 ms.", ["CONSTRAINT"]),
        ("I do not expect queue depth to be below 42 requests.", ["CONSTRAINT"]),
        ("It is not true that the measured latency was 73 ms.", []),
        ("Never copy cache-a into cache-b.", ["CONSTRAINT"]),
        ("This does not mean candidate blue-9 is serving traffic.", []),
        ("Run beta-9 did not fail.", []),
    )
    for index, (text, expected) in enumerate(negation_cases):
        items.append(
            _item(
                item_id=f"v5-fresh-negation-{index:03d}",
                text=text,
                expected=expected,
                projection="kinds",
                audit="negation",
            )
        )

    number_pairs = (
        (
            "Keep cache latency below {value} ms.",
            lambda value: _frame(
                "GOAL",
                "THRESHOLD",
                subject="cache_latency",
                operator="LT",
                value=value,
                unit="ms",
            ),
            (21, 22),
        ),
        (
            "Recorded queue depth was {value} requests.",
            lambda value: _frame(
                "OBSERVATION",
                "MEASUREMENT",
                subject="queue_depth",
                value=value,
                unit="requests",
            ),
            (91, 92),
        ),
        (
            "We forecast throughput will remain above {value} rps.",
            lambda value: _frame(
                "PREDICTION",
                "FORECAST",
                subject="throughput",
                operator="GT",
                value=value,
                unit="rps",
            ),
            (161, 162),
        ),
        (
            "Absolute value of -{value}.",
            lambda value: _frame("MAGNITUDE", "MAGNITUDE", input=-value),
            (17, 18),
        ),
    )
    for pair_index, (template, target, values) in enumerate(number_pairs):
        for variant, value in enumerate(values):
            items.append(
                _item(
                    item_id=f"v5-fresh-number-{pair_index:02d}-{variant}",
                    text=template.format(value=value),
                    expected=[target(value)],
                    projection="semantic_frames",
                    audit="number-change",
                    pair_id=f"v5-fresh-number-pair-{pair_index:02d}",
                )
            )

    order_pairs = _fresh_order_pairs()
    for pair_index, (left, right, expected) in enumerate(order_pairs):
        for variant, text in enumerate((left, right)):
            items.append(
                _item(
                    item_id=f"v5-fresh-order-{pair_index:02d}-{variant}",
                    text=text,
                    expected=expected,
                    projection="semantic_frames",
                    audit="order-swap",
                    pair_id=f"v5-fresh-order-pair-{pair_index:02d}",
                )
            )
    return items


def _fresh_order_pairs() -> list[tuple[str, str, list[dict[str, Any]]]]:
    return [
        (
            "Keep queue depth below 61 requests; recorded queue depth was 91 requests.",
            "Recorded queue depth was 91 requests; keep queue depth below 61 requests.",
            [
                _frame(
                    "GOAL",
                    "THRESHOLD",
                    subject="queue_depth",
                    operator="LT",
                    value=61,
                    unit="requests",
                ),
                _frame(
                    "OBSERVATION",
                    "MEASUREMENT",
                    subject="queue_depth",
                    value=91,
                    unit="requests",
                ),
            ],
        ),
        (
            "Never update archive-41; throughput will remain above 171 rps.",
            "Throughput will remain above 171 rps; never update archive-41.",
            [
                _frame(
                    "CONSTRAINT",
                    "ACTION_POLICY",
                    action="update",
                    object="archive_41",
                    polarity=False,
                    modality="FORBIDDEN",
                ),
                _frame(
                    "PREDICTION",
                    "FORECAST",
                    subject="throughput",
                    operator="GT",
                    value=171,
                    unit="rps",
                ),
            ],
        ),
        (
            "Candidate blue-42 is live; run batch-42 completed.",
            "Run batch-42 completed; candidate blue-42 is live.",
            [
                _frame(
                    "ACTIVE_SELECTION",
                    "ACTIVE_SELECTION",
                    selection="blue_42",
                    active=True,
                ),
                _frame(
                    "RUN_STATUS",
                    "RUN_STATUS",
                    run_id="batch-42",
                    status="COMPLETED",
                ),
            ],
        ),
        (
            "Refer to ticket-43; receipt rec-43 returned 243.",
            "Receipt rec-43 returned 243; refer to ticket-43.",
            [
                _frame("REFERENCE", "REFERENCE", target="ticket_43"),
                _frame(
                    "RECEIPT_VALUE",
                    "RECEIPT_VALUE",
                    receipt_id="rec-43",
                    value=243,
                ),
            ],
        ),
    ]


def _retention_item(kind: str, variant: int, number: int) -> dict[str, Any]:
    if kind == "COPY_VALUE":
        text = f"Copy archive-{number} into replica-{number}."
        expected = [
            _frame(
                kind, kind, source=f"archive_{number}", destination=f"replica_{number}"
            )
        ]
    elif kind == "SUPERSEDES":
        text = f"Release-{number} replaces release-{number - 1}."
        expected = [
            _frame(
                kind,
                kind,
                old_value=f"release_{number - 1}",
                current_value=f"release_{number}",
            )
        ]
    elif kind == "MAGNITUDE":
        text = f"Distance from zero of -{number}."
        expected = [_frame(kind, kind, input=-number)]
    elif kind == "NEGATE":
        text = f"Opposite of {number}."
        expected = [_frame(kind, kind, input=number)]
    elif kind == "REFERENCE":
        text = f"Refer to ticket-{number}."
        expected = [_frame(kind, kind, target=f"ticket_{number}")]
    elif kind == "ACTIVE_SELECTION":
        text = f"candidate-{number} is live."
        expected = [_frame(kind, kind, selection=f"candidate_{number}", active=True)]
    elif kind == "RUN_STATUS":
        text = f"Run batch-{number} status is completed."
        expected = [_frame(kind, kind, run_id=f"batch-{number}", status="COMPLETED")]
    elif kind == "RECEIPT_VALUE":
        text = f"Receipt rec-{number} returned {500 + number}."
        expected = [_frame(kind, kind, receipt_id=f"rec-{number}", value=500 + number)]
    elif kind == "EVIDENCE_CONSISTENCY":
        wording = ("is consistent", "is conflicting", "is in agreement")[variant % 3]
        text = f"The evidence {wording}."
        expected = [
            _frame(
                kind,
                kind,
                status="CONFLICT" if "conflicting" in wording else "CONSISTENT",
            )
        ]
    elif kind == "GOAL":
        text = f"Maintain queue depth under {number} requests."
        expected = [
            _frame(
                kind,
                "THRESHOLD",
                subject="queue_depth",
                operator="LT",
                value=number,
                unit="requests",
            )
        ]
    elif kind == "CONSTRAINT":
        text = f"Never update archive-{number}."
        expected = [
            _frame(
                kind,
                "ACTION_POLICY",
                action="update",
                object=f"archive_{number}",
                polarity=False,
                modality="FORBIDDEN",
            )
        ]
    elif kind == "OBSERVATION":
        text = f"Recorded throughput was {number} rps."
        expected = [
            _frame(
                kind,
                "MEASUREMENT",
                subject="throughput",
                value=number,
                unit="rps",
            )
        ]
    elif kind == "PREDICTION":
        text = f"Throughput will remain above {number} rps."
        expected = [
            _frame(
                kind,
                "FORECAST",
                subject="throughput",
                operator="GT",
                value=number,
                unit="rps",
            )
        ]
    else:  # pragma: no cover - exhaustive constant mapping
        raise ValueError(kind)
    return _item(
        item_id=f"v5-retention-{kind.casefold().replace('_', '-')}-{variant}",
        text=text,
        expected=expected,
        projection="semantic_frames",
        audit="retention",
    )


def _retention_items() -> list[dict[str, Any]]:
    items = [
        _retention_item(kind, variant, 310 + kind_index * 3 + variant)
        for kind_index, kind in enumerate(FRAME_KINDS)
        for variant in range(3)
    ]
    items.append(_retention_item("GOAL", 3, 399))
    return items


def _oov_items() -> list[dict[str, Any]]:
    templates: dict[str, tuple[str, str]] = {
        "ameliorate": (
            "Please ameliorate response handling for service-{n} before launch.",
            "GOAL",
        ),
        "sacrosanct": (
            "The production-{n} boundary is sacrosanct; leave it untouched.",
            "CONSTRAINT",
        ),
        "attested": (
            "An attested reading places queue delay at {value} milliseconds.",
            "OBSERVATION",
        ),
        "presage": (
            "These signals presage queue delay reaching {value} milliseconds tomorrow.",
            "PREDICTION",
        ),
        "dormant": (
            "Run batch-{n} is dormant after finishing its work.",
            "RUN_STATUS",
        ),
        "displaces": (
            "Build cobalt-{n} displaces amber-{n} as the current build.",
            "SUPERSEDES",
        ),
        "norm": ("Return the norm of -{value}.", "MAGNITUDE"),
        "antipode": ("Give the additive antipode of {value}.", "NEGATE"),
        "citation": (
            "Use ticket-{n} as the citation supporting this decision.",
            "REFERENCE",
        ),
        "concordant": (
            "The two readings for sample-{n} are concordant.",
            "EVIDENCE_CONSISTENCY",
        ),
    }
    items: list[dict[str, Any]] = []
    for term_index, term in enumerate(HELD_OUT_TERMS):
        template, kind = templates[term]
        for variant in range(4):
            number = 510 + term_index * 4 + variant
            items.append(
                _item(
                    item_id=f"v5-oov-{term}-{variant}",
                    text=template.format(n=number, value=number + 20),
                    expected=[kind],
                    projection="kinds",
                    audit=f"held-out-vocabulary:{term}",
                )
            )
    return items


def _composition_items() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []

    multi_templates = (
        (
            "Bring queue delay below {a} ms, while leaving production-{n} untouched.",
            ["GOAL", "CONSTRAINT"],
        ),
        (
            "The latest reading was {a} ms, and tomorrow service delay will reach {b} ms.",
            ["OBSERVATION", "PREDICTION"],
        ),
        (
            "Candidate blue-{n} is serving traffic; batch-{n} finished successfully.",
            ["ACTIVE_SELECTION", "RUN_STATUS"],
        ),
        (
            "Reverse the sign of {a}, then find how far -{b} is from zero.",
            ["NEGATE", "MAGNITUDE"],
        ),
        (
            "Transfer source-{n} into destination-{n}; build cobalt-{n} replaces amber-{n}; cite ticket-{n}.",
            ["COPY_VALUE", "SUPERSEDES", "REFERENCE"],
        ),
        (
            "Bring queue delay below {a} ms; production-{n} stays untouched; the instrument read {b} ms; tomorrow it will reach {a} ms.",
            ["GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION"],
        ),
    )
    for index in range(16):
        template, expected = multi_templates[index % len(multi_templates)]
        items.append(
            _item(
                item_id=f"v5-composition-multi-{index:03d}",
                text=template.format(n=610 + index, a=45 + index, b=85 + index),
                expected=expected,
                projection="kinds",
                audit="multi-frame",
            )
        )

    for index in range(6):
        if index % 2 == 0:
            noun = f"dashboard_{620 + index}"
            boundary = f"production_{620 + index}"
            text = (
                f"Could you build dashboard-{620 + index} without ever modifying "
                f"production-{620 + index}?"
            )
            action = "build"
        else:
            noun = f"index_{620 + index}"
            boundary = f"catalog_{620 + index}"
            text = (
                f"Please create index-{620 + index} while keeping "
                f"catalog-{620 + index} unchanged."
            )
            action = "create"
        items.append(
            _item(
                item_id=f"v5-composition-polite-{index:03d}",
                text=text,
                expected=[
                    _frame("GOAL", "DIRECTIVE", action=action, object=noun),
                    _frame(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        action="modify",
                        object=boundary,
                        polarity=False,
                        modality="FORBIDDEN",
                    ),
                ],
                projection="semantic_frames",
                audit="polite-buried-constraint",
            )
        )

    buried_templates = (
        "Prepare launch notes for release-{n}; deep in the rollout plan, production-{n} must remain untouched.",
        "Improve service-{n}; among the fine print, never update archive-{n}.",
        "Build index-{n}, with the proviso that customer records-{n} stay untouched.",
    )
    for index in range(6):
        items.append(
            _item(
                item_id=f"v5-composition-buried-{index:03d}",
                text=buried_templates[index % 3].format(n=630 + index),
                expected=["GOAL", "CONSTRAINT"],
                projection="kinds",
                audit="buried-constraint",
            )
        )

    for pair_index in range(8):
        value = 54 + pair_index
        expected = [
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
        ]
        texts = (
            f"Lower response latency under {value} milliseconds; production must not be changed.",
            f"Keep latency below {value} ms and never modify production.",
        )
        for variant, text in enumerate(texts):
            items.append(
                _item(
                    item_id=f"v5-composition-paraphrase-{pair_index:02d}-{variant}",
                    text=text,
                    expected=expected,
                    projection="semantic_frames",
                    audit="paraphrase-equivalence",
                    pair_id=f"v5-composition-paraphrase-pair-{pair_index:02d}",
                )
            )

    for index in range(12):
        target = f"cache_{650 + index}"
        items.append(
            _item(
                item_id=f"v5-composition-conflict-{index:03d}",
                text=(
                    f"Never update cache-{650 + index}, but cache-{650 + index} "
                    "must be updated."
                ),
                expected=[
                    _frame(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        action="update",
                        object=target,
                        polarity=False,
                        modality="FORBIDDEN",
                    ),
                    _frame(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        action="update",
                        object=target,
                        polarity=True,
                        modality="REQUIRED",
                    ),
                ],
                projection="semantic_frames",
                audit="conflicting-constraints",
            )
        )

    for index in range(8):
        if index % 2 == 0:
            text = (
                f"Please do not lower service delay below {40 + index} ms; "
                f"the instrument recorded service delay {80 + index} ms."
            )
            expected = ["CONSTRAINT", "OBSERVATION"]
        else:
            text = (
                f"Do not copy cache-{index} into replica-{index}; "
                f"run batch-{670 + index} completed."
            )
            expected = ["CONSTRAINT", "RUN_STATUS"]
        items.append(
            _item(
                item_id=f"v5-composition-negation-{index:03d}",
                text=text,
                expected=expected,
                projection="kinds",
                audit="negation-composition",
            )
        )

    for pair_index in range(4):
        target = 66 + pair_index
        measured = 96 + pair_index
        expected = [
            _frame(
                "GOAL",
                "THRESHOLD",
                subject="queue_depth",
                operator="LT",
                value=target,
                unit="requests",
            ),
            _frame(
                "OBSERVATION",
                "MEASUREMENT",
                subject="queue_depth",
                value=measured,
                unit="requests",
            ),
        ]
        texts = (
            f"Maintain queue depth under {target} requests; recorded queue depth was {measured} requests.",
            f"Recorded queue depth was {measured} requests; maintain queue depth under {target} requests.",
        )
        for variant, text in enumerate(texts):
            items.append(
                _item(
                    item_id=f"v5-composition-order-{pair_index:02d}-{variant}",
                    text=text,
                    expected=expected,
                    projection="semantic_frames",
                    audit="order-swap-composition",
                    pair_id=f"v5-composition-order-pair-{pair_index:02d}",
                )
            )

    for pair_index in range(4):
        for variant in range(2):
            target = 71 + pair_index * 3 + variant
            measured = 101 + pair_index * 3 + variant
            items.append(
                _item(
                    item_id=f"v5-composition-number-{pair_index:02d}-{variant}",
                    text=(
                        f"Keep request count below {target} requests; "
                        f"measured request count was {measured} requests."
                    ),
                    expected=[
                        _frame(
                            "GOAL",
                            "THRESHOLD",
                            subject="request_count",
                            operator="LT",
                            value=target,
                            unit="requests",
                        ),
                        _frame(
                            "OBSERVATION",
                            "MEASUREMENT",
                            subject="request_count",
                            value=measured,
                            unit="requests",
                        ),
                    ],
                    projection="semantic_frames",
                    audit="number-change-composition",
                    pair_id=f"v5-composition-number-pair-{pair_index:02d}",
                )
            )
    return items


def _calibration_items() -> list[dict[str, Any]]:
    templates = (
        ("Bring worker delay below {value} milliseconds.", "GOAL"),
        ("Billing records-{n} must remain untouched during this check.", "CONSTRAINT"),
        (
            "The latest instrument reading put worker delay at {value} milliseconds.",
            "OBSERVATION",
        ),
        ("Next week worker delay will reach {value} milliseconds.", "PREDICTION"),
    )
    items: list[dict[str, Any]] = []
    for index in range(20):
        template, kind = templates[index % 4]
        items.append(
            _item(
                item_id=f"v5-calibration-eval-{index:03d}",
                text=template.format(n=710 + index, value=110 + index),
                expected=[kind],
                projection="kinds",
                audit="calibration-semantic",
            )
        )
    return items


def build_calibration_fit() -> list[dict[str, Any]]:
    templates = {
        "COPY_VALUE": "Move the contents of alpha-{n} into beta-{n}.",
        "SUPERSEDES": "Build cobalt-{n} takes over from amber-{n} as the current release.",
        "MAGNITUDE": "How far is -{value} from zero?",
        "NEGATE": "Reverse the sign of {value}.",
        "REFERENCE": "Cite ticket-{n} as the supporting record.",
        "ACTIVE_SELECTION": "Candidate green-{n} is serving live traffic.",
        "RUN_STATUS": "Batch gamma-{n} finished successfully.",
        "RECEIPT_VALUE": "Operation rec-{n} returned numeric code {value}.",
        "EVIDENCE_CONSISTENCY": "The two readings agree for sample-{n}.",
        "GOAL": "Bring queueing delay under {value} milliseconds.",
        "CONSTRAINT": "Customer records-{n} must stay untouched during the trial.",
        "OBSERVATION": "The instrument measured queueing delay at {value} milliseconds.",
        "PREDICTION": "Tomorrow queueing delay will reach {value} milliseconds.",
    }
    rows: list[dict[str, Any]] = []
    for kind_index, kind in enumerate(FRAME_KINDS):
        for variant in range(4):
            number = 810 + kind_index * 4 + variant
            rows.append(
                {
                    "id": f"calibration-fit-v2-{kind_index:02d}-{variant}",
                    "text": templates[kind].format(n=number, value=number + 30),
                    "frame_kinds": [kind],
                    "split": "calibration_fit",
                    "provenance": (
                        "independently authored permission-clean frozen calibration fit v2"
                    ),
                    "surface_policy": "NATURAL_LANGUAGE_NO_EXPECTED_LABELS",
                }
            )
    return rows


def _calibration_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(
        (
            json.dumps(row, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
        for row in rows
    )


def build_suite() -> dict[str, Any]:
    calibration_rows = build_calibration_fit()
    return {
        "schema": "kev.eval-suite.v1",
        "id": "kev-public-audit-v5-260",
        "frozen": True,
        "provenance": {
            "authored_without_training_corpus_access": True,
            "created_for": "challenger decisions made only after the v5 boundary was frozen",
            "item_count": 260,
            "status": "FROZEN; create v6 rather than changing any byte",
            "independence_policy": (
                "new natural-language surfaces authored before v5 challenger training; "
                "no lesson or training corpus was read or consumed by this builder"
            ),
            "label_leakage_policy": (
                "input text must not contain any complete expected frame-kind label"
            ),
            "training_exclusion": (
                "every item, the calibration-fit slice, and held-out terms are forbidden "
                "inputs to train()"
            ),
            "held_out_vocabulary_sha256": _sha256_bytes(_held_out_bytes()),
            "calibration_fit_sha256": _sha256_bytes(
                _calibration_bytes(calibration_rows)
            ),
            "prior_current_suite_sha256": file_sha256(V4_SUITE),
            "preserved_generations": [1, 2, 3, 4],
            "audit_families": sorted(REQUIRED_COMPOSITION_AUDITS),
        },
        "splits": {
            "fresh": _fresh_items(),
            "retention": _retention_items(),
            "oov": _oov_items(),
            "composition": _composition_items(),
            "calibration": _calibration_items(),
        },
    }


def _expected_kinds(item: dict[str, Any]) -> list[str]:
    result: list[str] = []
    for expected in item["expected"]:
        result.append(expected if isinstance(expected, str) else expected["kind"])
    return result


def _contains_complete_label(text: str, kind: str) -> bool:
    words = re.findall(r"[a-z0-9]+", text.casefold())
    label = kind.casefold().split("_")
    return any(
        words[index : index + len(label)] == label for index in range(len(words))
    )


def _normalized_surface(text: str) -> str:
    return " ".join(text.casefold().split())


def _prior_eval_surfaces() -> set[str]:
    surfaces: set[str] = set()
    for path in sorted(FROZEN.glob("public-audit-v[1-4]-260.json")):
        suite = json.loads(path.read_text(encoding="utf-8"))
        surfaces.update(
            _normalized_surface(str(item["input"]))
            for items in suite["splits"].values()
            for item in items
        )
    return surfaces


def validate_calibration_fit(rows: list[dict[str, Any]], suite: dict[str, Any]) -> None:
    if len(rows) != 52:
        raise ValueError(f"calibration-fit v2 must contain 52 rows, got {len(rows)}")
    identifiers = [row["id"] for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("calibration-fit v2 identifiers are not unique")
    counts = Counter(row["frame_kinds"][0] for row in rows)
    if counts != Counter({kind: 4 for kind in FRAME_KINDS}):
        raise ValueError(f"calibration-fit v2 kind balance changed: {counts}")
    promotion_surfaces = {
        _normalized_surface(item["input"])
        for items in suite["splits"].values()
        for item in items
    }
    prior_surfaces = _prior_eval_surfaces()
    seen: set[str] = set()
    for row in rows:
        text = row["text"]
        normalized = _normalized_surface(text)
        if normalized in seen:
            raise ValueError(f"duplicate calibration-fit surface: {text!r}")
        seen.add(normalized)
        if normalized in promotion_surfaces:
            raise ValueError(f"calibration-fit/promotion overlap: {text!r}")
        if normalized in prior_surfaces:
            raise ValueError(f"calibration-fit/prior-eval overlap: {text!r}")
        kind = row["frame_kinds"][0]
        if _contains_complete_label(text, kind):
            raise ValueError(
                f"expected label {kind!r} leaks into calibration row {row['id']}"
            )
        if any(
            re.search(rf"\b{re.escape(term)}\b", text.casefold())
            for term in HELD_OUT_TERMS
        ):
            raise ValueError(f"held-out term leaks into calibration row {row['id']}")


def _validate_pairs(suite: dict[str, Any]) -> None:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for items in suite["splits"].values():
        for item in items:
            if "pair_id" in item:
                groups[item["pair_id"]].append(item)
    for pair_id, items in groups.items():
        if len(items) != 2:
            raise ValueError(f"pair {pair_id!r} must have exactly two items")
        left, right = items
        same_expected = canonical_json_sha256(
            left["expected"]
        ) == canonical_json_sha256(right["expected"])
        if "number-pair" in pair_id:
            if same_expected:
                raise ValueError(
                    f"number-change pair {pair_id!r} did not change its target"
                )
        elif not same_expected:
            raise ValueError(f"equivalence/order pair {pair_id!r} changed its target")


def validate_suite(suite: dict[str, Any]) -> None:
    if {name: len(items) for name, items in suite["splits"].items()} != SPLIT_COUNTS:
        raise ValueError("v5 split counts changed")
    items = [item for split in suite["splits"].values() for item in split]
    identifiers = [item["id"] for item in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("v5 item identifiers are not globally unique")
    normalized = [_normalized_surface(item["input"]) for item in items]
    if len(normalized) != len(set(normalized)):
        raise ValueError("v5 input surfaces are not globally unique")
    prior_overlap = sorted(set(normalized) & _prior_eval_surfaces())
    if prior_overlap:
        raise ValueError(f"v5 surface overlaps prior evaluation: {prior_overlap[0]!r}")

    for item in items:
        lowered = item["input"].casefold()
        for marker in SYNTHETIC_MARKERS:
            if marker in lowered:
                raise ValueError(f"synthetic marker {marker!r} in {item['id']}")
        for kind in _expected_kinds(item):
            if _contains_complete_label(item["input"], kind):
                raise ValueError(f"expected label {kind!r} leaks into {item['id']}")

    composition = suite["splits"]["composition"]
    audits = Counter(item["audit"] for item in composition)
    missing = sorted(REQUIRED_COMPOSITION_AUDITS - set(audits))
    if missing:
        raise ValueError(f"required v5 composition audits missing: {missing}")
    if any(len(item["expected"]) < 2 for item in composition):
        raise ValueError("every v5 composition item must require multiple frames")
    if any(
        item["projection"] != "semantic_frames" for item in suite["splits"]["retention"]
    ):
        raise ValueError("retention must exercise exact typed semantic frames")

    oov_items = suite["splits"]["oov"]
    non_oov_items = [
        item
        for split, rows in suite["splits"].items()
        if split != "oov"
        for item in rows
    ]
    for term in HELD_OUT_TERMS:
        matching = [
            item
            for item in oov_items
            if re.search(rf"\b{re.escape(term)}\b", item["input"].casefold())
        ]
        if len(matching) != 4:
            raise ValueError(
                f"held-out term {term!r} must appear in exactly four OOV items"
            )
        if any(
            re.search(rf"\b{re.escape(term)}\b", item["input"].casefold())
            for item in non_oov_items
        ):
            raise ValueError(f"held-out term {term!r} appears outside OOV")
    _validate_pairs(suite)


def _artifact(path: Path, *, canonical: bool = False, **extra: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": file_sha256(path),
        "size_bytes": path.stat().st_size,
    }
    if canonical:
        result["canonical_sha256"] = canonical_json_sha256(
            json.loads(path.read_text(encoding="utf-8"))
        )
    result.update(extra)
    return result


def build_manifest() -> dict[str, Any]:
    v4_manifest = json.loads(V4_MANIFEST.read_text(encoding="utf-8"))
    artifacts = dict(v4_manifest["artifacts"])
    prior_suite = artifacts.pop("promotion_suite")
    prior_suite["status"] = "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE"
    artifacts["preserved_promotion_suite_v4"] = prior_suite
    prior_calibration = artifacts.pop("calibration_fit")
    prior_calibration["status"] = "PRESERVED_LABEL_LEAKY_CALIBRATION_FIT_NOT_CURRENT"
    artifacts["preserved_calibration_fit_v1"] = prior_calibration
    prior_vocabulary = artifacts.pop("held_out_vocabulary")
    prior_vocabulary["status"] = "PRESERVED_PRIOR_HELD_OUT_VOCABULARY"
    artifacts["preserved_held_out_vocabulary_v1"] = prior_vocabulary
    artifacts.update(
        {
            "promotion_suite": _artifact(V5_SUITE, canonical=True),
            "calibration_fit": _artifact(V5_CALIBRATION_FIT),
            "held_out_vocabulary": _artifact(V5_HELD_OUT),
            "preserved_manifest_v4": _artifact(
                V4_MANIFEST,
                canonical=True,
                status="PRESERVED_PREVIOUS_EVALUATION_MANIFEST",
            ),
            "v4_baseline_self_evaluation": _artifact(
                V4_BASELINE,
                canonical=True,
                status="PRESERVED_PRE_V5_BASELINE_WITH_RAW_FAILURES",
            ),
        }
    )
    return {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 5,
        "preserved_generations": [1, 2, 3, 4],
        "artifacts": artifacts,
        "mutation_policy": "create a new version; never refresh v1 through v5 in place",
        "training_boundary": {
            "suite": "FORBIDDEN_IN_TRAIN",
            "calibration_fit": "TEMPERATURE_ONLY_FORBIDDEN_IN_WEIGHT_TRAIN",
            "held_out_vocabulary": "FORBIDDEN_SURFACE_FORMS_IN_TRAIN",
        },
    }


def main() -> None:
    outputs = (V5_SUITE, V5_MANIFEST, V5_HELD_OUT, V5_CALIBRATION_FIT)
    if any(path.exists() for path in outputs):
        raise SystemExit("v5 artifacts already exist; refusing to refresh them")
    for required in (V4_SUITE, V4_MANIFEST, V4_BASELINE):
        if not required.is_file():
            raise SystemExit(f"required preserved v4 evidence missing: {required}")

    suite = build_suite()
    calibration_rows = build_calibration_fit()
    validate_suite(suite)
    validate_calibration_fit(calibration_rows, suite)

    with V5_HELD_OUT.open("xb") as handle:
        handle.write(_held_out_bytes())
    with V5_CALIBRATION_FIT.open("xb") as handle:
        handle.write(_calibration_bytes(calibration_rows))
    with V5_SUITE.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(suite, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")

    manifest = build_manifest()
    with V5_MANIFEST.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

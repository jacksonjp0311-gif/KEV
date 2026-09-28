"""Build KEV's immutable v1 public evaluation artifacts.

The historical 190/260 item texts were not present in the public repository.
This script therefore publishes a new, explicit 260-item suite for future
promotion decisions and a separate aggregate-only replay of the historical
tie. It refuses to replace any v1 output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.evaluation import replay_historical_aggregate


ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "evals" / "frozen"
SUITE_PATH = EVAL_DIR / "public-audit-v1-260.json"
CALIBRATION_PATH = EVAL_DIR / "calibration-fit-v1.jsonl"
VOCAB_PATH = EVAL_DIR / "held-out-vocabulary-v1.txt"
REPLAY_PATH = EVAL_DIR / "historical-190-260-replay.json"
MANIFEST_PATH = EVAL_DIR / "manifest-v1.json"


def frame(kind: str, relation: str, **slots: Any) -> dict[str, Any]:
    return {"kind": kind, "relation": relation, "slots": slots}


def item(
    identifier: str,
    text: str,
    expected: list[Any],
    *,
    audit: str,
    projection: str = "semantic_frames",
) -> dict[str, Any]:
    return {
        "id": identifier,
        "input": text,
        "expected": expected,
        "comparison": "set_exact",
        "projection": projection,
        "audit": audit,
    }


def public_suite() -> dict[str, Any]:
    fresh: list[dict[str, Any]] = []
    retention: list[dict[str, Any]] = []
    oov: list[dict[str, Any]] = []
    composition: list[dict[str, Any]] = []
    calibration: list[dict[str, Any]] = []

    subjects = ("latency", "throughput", "error rate", "queue depth")
    units = ("ms", "rps", "%", "requests")
    for index in range(20):
        subject = subjects[index % len(subjects)]
        unit = units[index % len(units)]
        value = 25 + index
        fresh.append(
            item(
                f"fresh-goal-{index:03d}",
                f"Reduce {subject} below {value} {unit}",
                [
                    frame(
                        "GOAL",
                        "THRESHOLD",
                        subject=subject.replace(" ", "_"),
                        operator="LT",
                        value=value,
                        unit=unit,
                    )
                ],
                audit="single-frame-goal",
            )
        )
    objects = (
        "production",
        "billing",
        "the audit ledger",
        "customer records",
        "model registry",
    )
    for index in range(15):
        target = objects[index % len(objects)]
        fresh.append(
            item(
                f"fresh-constraint-{index:03d}",
                f"Do not modify {target}",
                [
                    frame(
                        "CONSTRAINT",
                        "ACTION_POLICY",
                        action="modify",
                        object=target.replace(" ", "_").removeprefix("the_"),
                        polarity=False,
                        modality="FORBIDDEN",
                    )
                ],
                audit="negation",
            )
        )
    for index in range(15):
        value = 61 + index
        fresh.append(
            item(
                f"fresh-observation-{index:03d}",
                f"The last measured latency was {value} ms",
                [
                    frame(
                        "OBSERVATION",
                        "MEASUREMENT",
                        subject="latency",
                        value=value,
                        unit="ms",
                    )
                ],
                audit="number-change",
            )
        )
    for index in range(15):
        value = 40 + index
        fresh.append(
            item(
                f"fresh-prediction-{index:03d}",
                f"We predict latency will be below {value} ms",
                [
                    frame(
                        "PREDICTION",
                        "FORECAST",
                        subject="latency",
                        operator="LT",
                        value=value,
                        unit="ms",
                    )
                ],
                audit="prediction",
            )
        )

    base_templates = [
        (
            "COPY_VALUE",
            lambda i: (
                f"Copy source{i} to destination{i}",
                frame(
                    "COPY_VALUE",
                    "COPY_VALUE",
                    source=f"source{i}",
                    destination=f"destination{i}",
                ),
            ),
        ),
        (
            "SUPERSEDES",
            lambda i: (
                f"new{i} supersedes old{i}",
                frame(
                    "SUPERSEDES",
                    "SUPERSEDES",
                    old_value=f"old{i}",
                    current_value=f"new{i}",
                ),
            ),
        ),
        (
            "MAGNITUDE",
            lambda i: (
                f"Magnitude of -{i + 1}",
                frame("MAGNITUDE", "MAGNITUDE", input=-(i + 1)),
            ),
        ),
        (
            "NEGATE",
            lambda i: (f"Negate {i + 1}", frame("NEGATE", "NEGATE", input=i + 1)),
        ),
        (
            "REFERENCE",
            lambda i: (
                f"Reference item-{i}",
                frame("REFERENCE", "REFERENCE", target=f"item_{i}"),
            ),
        ),
        (
            "ACTIVE_SELECTION",
            lambda i: (
                f"candidate-{i} is active",
                frame(
                    "ACTIVE_SELECTION",
                    "ACTIVE_SELECTION",
                    selection=f"candidate_{i}",
                    active=True,
                ),
            ),
        ),
        (
            "RUN_STATUS",
            lambda i: (
                f"run job-{i} completed",
                frame(
                    "RUN_STATUS", "RUN_STATUS", run_id=f"job-{i}", status="COMPLETED"
                ),
            ),
        ),
        (
            "RECEIPT_VALUE",
            lambda i: (
                f"receipt rcpt-{i} returned value-{i}",
                frame(
                    "RECEIPT_VALUE",
                    "RECEIPT_VALUE",
                    receipt_id=f"rcpt-{i}",
                    value=f"value-{i}",
                ),
            ),
        ),
        (
            "EVIDENCE_CONSISTENCY",
            lambda i: (
                "The evidence is consistent",
                frame(
                    "EVIDENCE_CONSISTENCY", "EVIDENCE_CONSISTENCY", status="CONSISTENT"
                ),
            ),
        ),
    ]
    for index in range(35):
        kind, maker = base_templates[index % len(base_templates)]
        text, expected_frame = maker(index)
        fresh.append(
            item(
                f"fresh-base-{index:03d}",
                text,
                [expected_frame],
                audit=f"base-{kind.casefold()}",
            )
        )

    # Retention repeats established relations with different payloads and is
    # never read by train().
    for index in range(60):
        group = index % 6
        value = 80 + index
        if group == 0:
            text = f"Keep latency at most {value} ms"
            expected = [
                frame(
                    "GOAL",
                    "THRESHOLD",
                    subject="latency",
                    operator="LTE",
                    value=value,
                    unit="ms",
                )
            ]
        elif group == 1:
            text = f"Never modify archive-{index}"
            expected = [
                frame(
                    "CONSTRAINT",
                    "ACTION_POLICY",
                    action="modify",
                    object=f"archive_{index}",
                    polarity=False,
                    modality="FORBIDDEN",
                )
            ]
        elif group == 2:
            text = f"Observed latency was {value} ms"
            expected = [
                frame(
                    "OBSERVATION",
                    "MEASUREMENT",
                    subject="latency",
                    value=value,
                    unit="ms",
                )
            ]
        elif group == 3:
            text = f"Latency will remain below {value} ms"
            expected = [
                frame(
                    "PREDICTION",
                    "FORECAST",
                    subject="latency",
                    operator="LT",
                    value=value,
                    unit="ms",
                )
            ]
        elif group == 4:
            text = f"replace old-{index} with new-{index}"
            expected = [
                frame(
                    "SUPERSEDES",
                    "SUPERSEDES",
                    old_value=f"old_{index}",
                    current_value=f"new_{index}",
                )
            ]
        else:
            text = f"run retain-{index} failed"
            expected = [
                frame(
                    "RUN_STATUS",
                    "RUN_STATUS",
                    run_id=f"retain-{index}",
                    status="FAILED",
                )
            ]
        retention.append(
            item(f"retention-{index:03d}", text, expected, audit="retention")
        )

    held_out = (
        ("celerity", "GOAL"),
        ("inviolable", "CONSTRAINT"),
        ("empirically", "OBSERVATION"),
        ("portend", "PREDICTION"),
        ("quiescent", "RUN_STATUS"),
        ("antedescedes", "SUPERSEDES"),
        ("modulus", "MAGNITUDE"),
        ("contrapose", "NEGATE"),
    )
    for index in range(40):
        word, kind = held_out[index % len(held_out)]
        text = f"OOV audit {index}: {word} token expresses {kind.casefold()} semantics"
        oov.append(
            item(
                f"oov-{index:03d}",
                text,
                [kind],
                audit=f"held-out-vocabulary:{word}",
                projection="kinds",
            )
        )

    for index in range(10):
        goal = 40 + index
        observed = 70 + index
        text = (
            f"Reduce latency below {goal} ms, do not modify production, "
            f"and the last measured latency was {observed} ms"
        )
        expected = [
            frame(
                "GOAL",
                "THRESHOLD",
                subject="latency",
                operator="LT",
                value=goal,
                unit="ms",
            ),
            frame(
                "CONSTRAINT",
                "ACTION_POLICY",
                action="modify",
                object="production",
                polarity=False,
                modality="FORBIDDEN",
            ),
            frame(
                "OBSERVATION",
                "MEASUREMENT",
                subject="latency",
                value=observed,
                unit="ms",
            ),
        ]
        composition.append(
            item(f"composition-multi-{index:03d}", text, expected, audit="multi-frame")
        )
    for index in range(10):
        goal = 45 + index
        text = f"Please improve the service to keep latency below {goal} ms without modifying production"
        expected = [
            frame(
                "GOAL",
                "THRESHOLD",
                subject="latency",
                operator="LT",
                value=goal,
                unit="ms",
            ),
            frame(
                "CONSTRAINT",
                "ACTION_POLICY",
                action="modify",
                object="production",
                polarity=False,
                modality="FORBIDDEN",
            ),
        ]
        composition.append(
            item(
                f"composition-buried-{index:03d}",
                text,
                expected,
                audit="polite-buried-constraint",
            )
        )
    for index in range(10):
        value = 50 + index // 2
        text = (
            f"Lower response time under {value} milliseconds; production must not be changed"
            if index % 2 == 0
            else f"Keep latency below {value} ms and never modify production"
        )
        expected = [
            frame(
                "GOAL",
                "THRESHOLD",
                subject="latency",
                operator="LT",
                value=value,
                unit="ms",
            ),
            frame(
                "CONSTRAINT",
                "ACTION_POLICY",
                action="modify",
                object="production",
                polarity=False,
                modality="FORBIDDEN",
            ),
        ]
        composition.append(
            item(
                f"composition-paraphrase-{index:03d}",
                text,
                expected,
                audit="paraphrase-same-frames",
            )
        )
    for index in range(10):
        target = f"staging-{index}"
        text = f"Do not modify {target}, but {target} must be modified"
        expected = [
            frame(
                "CONSTRAINT",
                "ACTION_POLICY",
                action="modify",
                object=target.replace("-", "_"),
                polarity=False,
                modality="FORBIDDEN",
            ),
            frame(
                "CONSTRAINT",
                "ACTION_POLICY",
                action="modify",
                object=target.replace("-", "_"),
                polarity=True,
                modality="REQUIRED",
            ),
        ]
        composition.append(
            item(
                f"composition-conflict-{index:03d}",
                text,
                expected,
                audit="conflicting-constraints",
            )
        )

    for index in range(20):
        kind = ("GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION")[index % 4]
        text = f"Calibration audit {index} expresses {kind.casefold()} intent"
        calibration.append(
            item(
                f"calibration-eval-{index:03d}",
                text,
                [kind],
                audit="calibration-measurement-only",
                projection="kinds",
            )
        )

    assert tuple(map(len, (fresh, retention, oov, composition, calibration))) == (
        100,
        60,
        40,
        40,
        20,
    )
    return {
        "schema": "kev.eval-suite.v1",
        "id": "kev-public-audit-v1-260",
        "frozen": True,
        "provenance": {
            "created_for": "KEV v0.52 evidence-bearing evolution",
            "item_count": 260,
            "status": "FROZEN; version bump required for any change",
            "historical_note": "not a reconstruction of the unavailable historical 190/260 item texts",
            "training_exclusion": "this file and all split items are forbidden inputs to train()",
        },
        "splits": {
            "fresh": fresh,
            "retention": retention,
            "oov": oov,
            "composition": composition,
            "calibration": calibration,
        },
    }


def calibration_rows() -> list[dict[str, Any]]:
    kinds = ("GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION")
    return [
        {
            "id": f"calibration-fit-{index:03d}",
            "split": "calibration_fit",
            "text": f"Independent calibration phrase {index} for {kinds[index % 4].casefold()}",
            "frame_kinds": [kinds[index % 4]],
            "provenance": "synthetic permission-clean held-out calibration",
        }
        for index in range(32)
    ]


def write_new(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


def main() -> None:
    outputs = (SUITE_PATH, CALIBRATION_PATH, VOCAB_PATH, REPLAY_PATH, MANIFEST_PATH)
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(
            "v1 eval artifacts are immutable and already exist: " + ", ".join(existing)
        )
    suite = public_suite()
    write_new(SUITE_PATH, json.dumps(suite, indent=2, sort_keys=True) + "\n")
    write_new(
        CALIBRATION_PATH,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in calibration_rows()),
    )
    held_out_words = sorted(
        {item["audit"].split(":", 1)[1] for item in suite["splits"]["oov"]}
    )
    write_new(VOCAB_PATH, "\n".join(held_out_words) + "\n")
    replay = replay_historical_aggregate()
    write_new(REPLAY_PATH, json.dumps(replay, indent=2, sort_keys=True) + "\n")
    artifacts = {}
    for role, path in {
        "promotion_suite": SUITE_PATH,
        "calibration_fit": CALIBRATION_PATH,
        "held_out_vocabulary": VOCAB_PATH,
        "historical_aggregate_replay": REPLAY_PATH,
    }.items():
        artifacts[role] = {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
    artifacts["promotion_suite"]["canonical_sha256"] = canonical_json_sha256(suite)
    manifest = {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 1,
        "artifacts": artifacts,
        "mutation_policy": "create a new version; never refresh v1 in place",
    }
    write_new(MANIFEST_PATH, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

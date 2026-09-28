"""Author and freeze v7 without consulting training data or model outputs.

Targets are authored semantic judgments, never outputs of KEV's parser.  This
is same-party synthetic evaluation authored in an isolated agent context,
not an external or independently human-reviewed benchmark.  V1-v6 are read
only for historical exclusion and retention provenance.  No runtime predictor
is imported here.  Run once: existing output bytes are never refreshed.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from kev.artifacts import canonical_json_sha256, file_sha256


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals/frozen"
SUITE_PATH = FROZEN / "public-audit-v7-260.json"
MANIFEST_PATH = FROZEN / "manifest-v7.json"
PREVIOUS_SUITE = FROZEN / "public-audit-v6-260.json"
PREVIOUS_MANIFEST = FROZEN / "manifest-v6.json"
HELD_OUT_PATH = FROZEN / "held-out-vocabulary-v2.txt"
CALIBRATION_PATH = FROZEN / "calibration-fit-v2.jsonl"
PREVIOUS_SHA256 = "a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029"
PREVIOUS_MANIFEST_SHA256 = (
    "07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce"
)
HELD_OUT_SHA256 = "087da62e8129dd259c1b2662f0439eb7d6387714528af14444a4575c66e76798"
CALIBRATION_SHA256 = "d0e7d4fcffef2c3ccd376acb2c77a1fb90289224caa046cddc88c9656ace101c"
SPLIT_COUNTS = dict(fresh=80, retention=40, oov=40, composition=80, calibration=20)
COMPOSITION_AUDITS = {
    "multi-frame",
    "repeated-kind",
    "polite-buried-constraint",
    "paraphrase-equivalence",
    "conflicting-constraints",
    "negation-composition",
    "order-swap-composition",
    "number-change-composition",
}
KINDS = (
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


def json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def normalized(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def template(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "<N>", normalized(text))


def frame(kind: str, relation: str, **slots: Any) -> dict[str, Any]:
    return {"kind": kind, "relation": relation, "slots": slots}


def goal(subject: str, value: int, operator: str = "LT", unit: str = "ms") -> dict:
    return frame(
        "GOAL", "THRESHOLD", subject=subject, operator=operator, value=value, unit=unit
    )


def observed(subject: str, value: int, unit: str = "ms") -> dict:
    return frame("OBSERVATION", "MEASUREMENT", subject=subject, value=value, unit=unit)


def predicted(subject: str, value: int, operator: str = "LT", unit: str = "ms") -> dict:
    return frame(
        "PREDICTION",
        "FORECAST",
        subject=subject,
        operator=operator,
        value=value,
        unit=unit,
    )


def policy(action: str, target: str, *, positive: bool = False) -> dict:
    return frame(
        "CONSTRAINT",
        "ACTION_POLICY",
        action=action,
        object=target,
        polarity=positive,
        modality="REQUIRED" if positive else "FORBIDDEN",
    )


def item(
    split: str,
    index: int,
    text: str,
    expected: list,
    audit: str,
    *,
    pair: str | None = None,
) -> dict:
    result = {
        "id": f"v7-{split}-{index:03d}",
        "input": text,
        "expected": expected,
        "comparison": "set_exact",
        "projection": "kinds" if isinstance(expected[0], str) else "semantic_frames",
        "audit": audit,
        "surface_policy": "SAME_PARTY_SYNTHETIC_NO_EXPECTED_LABELS",
        "provenance": {
            "target_authority": "AUTHORED_SEMANTIC_JUDGMENT_NOT_RUNTIME_OUTPUT",
            "training_use": "FORBIDDEN",
        },
    }
    if pair:
        result["pair_id"] = pair
    return result


# These kind controls deliberately test relation recognition separately from
# exact slot extraction.  No label name is present in its corresponding input.
KIND_CONTROLS = {
    "COPY_VALUE": (
        "Make the backup jar hold the same contents as the source jar.",
        "Duplicate everything in the cedar folder into the elm folder.",
        "Transfer the value stored in drawer A to drawer B without changing it.",
        "Fill the target cell with the value from the adjacent source cell.",
    ),
    "SUPERSEDES": (
        "Revision ochre retires revision violet and becomes its successor.",
        "The new charter takes the place of the old charter from today.",
        "From this point onward use edition spruce in place of edition fir.",
        "The replacement build has displaced its predecessor in the release history.",
    ),
    "MAGNITUDE": (
        "What is the unsigned size of the quantity minus seventeen?",
        "Discard the sign from negative twenty-three and give its size.",
        "Measure how many units separate minus nine from the origin.",
        "Report the absolute amount represented by a signed value of -28.",
    ),
    "NEGATE": (
        "Map plus nineteen to the number on the opposite side of zero.",
        "Multiply the signed quantity forty-one by minus one.",
        "Find the additive inverse of minus thirteen.",
        "Flip positive twenty-seven into its negative counterpart.",
    ),
    "REFERENCE": (
        "Point the reviewer to the signed maintenance log for supporting details.",
        "Attach a pointer to the source entry that supports this claim.",
        "Attribute this number to the laboratory notebook on page twelve.",
        "Link this assertion back to its originating inspection record.",
    ),
    "ACTIVE_SELECTION": (
        "Of the available builds, the indigo one is currently in service.",
        "The selected branch for this session is the cedar branch.",
        "Traffic is presently routed through candidate silver.",
        "The running configuration uses the western profile.",
    ),
    "RUN_STATUS": (
        "The nightly backup job reached its end without errors.",
        "The processing task stopped because its validation failed.",
        "The export job is still underway as of this report.",
        "The archive job has entered a waiting state.",
    ),
    "RECEIPT_VALUE": (
        "The execution record lists 207 as the operation's returned code.",
        "The acknowledged result for operation amber is the integer 14.",
        "The action's recorded return payload contains a count of 63.",
        "The completion acknowledgment reports the returned value as 81.",
    ),
    "EVIDENCE_CONSISTENCY": (
        "The independent source records support the same conclusion.",
        "The two instrument logs contradict one another.",
        "The witness account agrees with the stored measurements.",
        "The written report and the raw record disagree.",
    ),
    "GOAL": (
        "Aim for a shorter delay when the next release is ready.",
        "Our desired outcome is a lower waiting time for every client.",
        "The target for this sprint is faster delivery of responses.",
        "Work toward reducing the time spent in the queue.",
    ),
    "CONSTRAINT": (
        "The live customer archive must be left untouched throughout testing.",
        "Changing the production configuration is outside the permitted scope.",
        "Any write to the payroll store is prohibited during this task.",
        "Only the temporary sandbox may be altered in this exercise.",
    ),
    "OBSERVATION": (
        "The instrument just registered a response time of 37 milliseconds.",
        "The measured queue count in the latest sample was eighteen.",
        "The monitoring log shows a throughput of 126 requests per second.",
        "The operator reported that the test temperature reached 24 degrees.",
    ),
    "PREDICTION": (
        "By morning the service is expected to respond within 32 milliseconds.",
        "The forecast calls for twenty queued requests by the next sample.",
        "Tomorrow's throughput is likely to exceed 150 requests per second.",
        "We anticipate a lower error count in the coming interval.",
    ),
}


def fresh_items() -> list[dict]:
    rows = []
    for kind, texts in KIND_CONTROLS.items():
        for text in texts:
            rows.append(item("fresh", len(rows), text, [kind], "single-frame-kind"))
    subjects = (
        "handoff delay",
        "render time",
        "lookup delay",
        "dispatch time",
        "index delay",
        "startup time",
        "commit delay",
    )
    templates = (
        (
            "During this trial, keep {s} under {n} ms.",
            "Throughout this trial, do not modify {o}.",
            "In this trial, measured {s} was {v} ms.",
            "For the next trial, {s} will remain below {p} ms.",
        ),
        (
            "Please maintain {s} at most {n} ms during the check.",
            "For this check, never delete {o}.",
            "During the check we recorded {s} at {v} ms.",
            "After the check, {s} will be above {p} ms.",
        ),
        (
            "For this release reduce {s} below {n} ms.",
            "During this release, do not update {o}.",
            "For this release the instrument measured {s} at {v} ms.",
            "In the next release, {s} will remain under {p} ms.",
        ),
        (
            "In the pilot, keep {s} no more than {n} ms.",
            "Please never modify {o} in this pilot.",
            "The pilot recorded {s} at {v} ms.",
            "After the pilot, {s} will stay above {p} ms.",
        ),
        (
            "Until handover, maintain {s} below {n} ms.",
            "Before handover, do not delete {o}.",
            "Before handover the measured {s} was {v} ms.",
            "Following handover, {s} will be below {p} ms.",
        ),
        (
            "For the demonstration, keep {s} at most {n} ms.",
            "During the demonstration, never update {o}.",
            "In the demonstration, recorded {s} was {v} ms.",
            "After the demonstration, {s} will exceed {p} ms.",
        ),
        (
            "For the rehearsal, reduce {s} under {n} ms.",
            "For the rehearsal, do not modify {o}.",
            "In the rehearsal we measured {s} at {v} ms.",
            "After the rehearsal, {s} will remain below {p} ms.",
        ),
    )
    actions = ("modify", "delete", "update", "modify", "delete", "update", "modify")
    for i, (subject, forms) in enumerate(zip(subjects, templates, strict=True)):
        target = f"protected-archive-{i + 71}"
        values = dict(s=subject, n=23 + i, o=target, v=61 + i, p=47 + i)
        targets = (
            goal(subject.replace(" ", "_"), 23 + i, "LTE" if i in (1, 3, 5) else "LT"),
            policy(actions[i], target.replace("-", "_")),
            observed(subject.replace(" ", "_"), 61 + i),
            predicted(
                subject.replace(" ", "_"), 47 + i, "GT" if i in (1, 3, 5) else "LT"
            ),
        )
        for text, expected in zip(forms, targets, strict=True):
            rows.append(
                item(
                    "fresh",
                    len(rows),
                    text.format(**values),
                    [expected],
                    "single-frame:" + expected["kind"].lower(),
                )
            )
    return rows


OOV_CONTROLS = {
    "ameliorate": (
        "GOAL",
        (
            "Our next milestone is to ameliorate the response delay for mobile clients.",
            "Ameliorate the slow startup experienced by new sessions.",
            "The team aims to ameliorate the waiting time at the intake desk.",
            "Work to ameliorate the intermittent slowdown in report delivery.",
        ),
    ),
    "sacrosanct": (
        "CONSTRAINT",
        (
            "Treat the production archive as sacrosanct throughout the migration test.",
            "The original ledger is sacrosanct and must never be rewritten.",
            "Keep the sacrosanct billing tables untouched during inspection.",
            "No change is permitted to the customer data; that boundary is sacrosanct.",
        ),
    ),
    "attested": (
        "OBSERVATION",
        (
            "The attested instrument sample gives a response delay of 48 milliseconds.",
            "At noon an attested log recorded seven requests in the queue.",
            "An attested measurement from the probe reports 19 degrees.",
            "The archived, attested reading for this batch was 86 milliseconds.",
        ),
    ),
    "presage": (
        "PREDICTION",
        (
            "These rising counters presage a longer queue in the next interval.",
            "The trend may presage a throughput of 180 requests per second tomorrow.",
            "Such early signals presage slower responses after the next deployment.",
            "The current indicators presage fewer timeouts during tomorrow's run.",
        ),
    ),
    "dormant": (
        "RUN_STATUS",
        (
            "The archive task is now dormant while it awaits its next trigger.",
            "After completing its cycle, the scheduled worker became dormant.",
            "The batch processor remains dormant pending a new input file.",
            "The nightly job has been dormant since its last successful completion.",
        ),
    ),
    "displaces": (
        "SUPERSEDES",
        (
            "Edition copper displaces edition tin in the maintained release sequence.",
            "As the authoritative successor, the new charter displaces the old charter.",
            "The revised build displaces its predecessor from the current release slot.",
            "Today's signed revision displaces yesterday's revision for future use.",
        ),
    ),
    "norm": (
        "MAGNITUDE",
        (
            "Compute the scalar norm for the signed quantity -43.",
            "What norm corresponds to a one-dimensional value of negative twenty?",
            "For the scalar -72, state its norm without retaining the sign.",
            "Give the norm of the signed scalar whose value is minus thirty-one.",
        ),
    ),
    "antipode": (
        "NEGATE",
        (
            "Find the additive antipode of minus twenty-two on the number line.",
            "Map positive 46 to its additive antipode.",
            "Which number is the additive antipode of 35?",
            "On the signed real line, provide the additive antipode of -64.",
        ),
    ),
    "citation": (
        "REFERENCE",
        (
            "Attach the maintenance notebook entry as a citation for this assertion.",
            "A citation to inspection record elm should accompany the claim.",
            "Point readers to the source log with an explicit citation.",
            "Use a citation to the signed test report to document the source.",
        ),
    ),
    "concordant": (
        "EVIDENCE_CONSISTENCY",
        (
            "The laboratory measurements are concordant with the independent field log.",
            "Both witness records are concordant about the order of events.",
            "The archived report and the new sample provide concordant evidence.",
            "The instrument traces remain concordant across the two probes.",
        ),
    ),
}


def oov_items() -> list[dict]:
    return [
        item("oov", i, text, [kind], "held-out-vocabulary:" + term)
        for i, (term, kind, text) in enumerate(
            (term, kind, text)
            for term, (kind, texts) in OOV_CONTROLS.items()
            for text in texts
        )
    ]


def composition_items() -> list[dict]:
    rows: list[dict] = []

    def add(text: str, expected: list, audit: str, pair: str | None = None) -> None:
        rows.append(item("composition", len(rows), text, expected, audit, pair=pair))

    # Ten heterogeneous frame sets; temporal and politeness context carries no
    # independent fact label, and reports remain reported rather than verified.
    for i, subject in enumerate(
        ("transport delay", "decode time", "search delay", "merge time", "flush delay")
    ):
        s = subject.replace(" ", "_")
        add(
            f"For the benchmark, keep {subject} under {31 + i} ms, and measured {subject} was {74 + i} ms.",
            [goal(s, 31 + i), observed(s, 74 + i)],
            "multi-frame",
        )
        add(
            f"Measured {subject} was {91 + i} ms; never modify snapshot-{81 + i}; {subject} will remain below {68 + i} ms.",
            [
                observed(s, 91 + i),
                policy("modify", f"snapshot_{81 + i}"),
                predicted(s, 68 + i),
            ],
            "multi-frame",
        )

    repeated = (
        (
            "Keep cache delay under 18 ms and keep network delay under 36 ms during this drill.",
            [goal("cache_delay", 18), goal("network_delay", 36)],
        ),
        (
            "For the dry run, measured cache delay was 22 ms and measured network delay was 41 ms.",
            [observed("cache_delay", 22), observed("network_delay", 41)],
        ),
        (
            "Cache delay will remain below 25 ms and network delay will remain below 39 ms after this drill.",
            [predicted("cache_delay", 25), predicted("network_delay", 39)],
        ),
        (
            "During this drill, do not modify archive-east and do not delete archive-west.",
            [policy("modify", "archive_east"), policy("delete", "archive_west")],
        ),
        (
            "Keep parse time under 21 ms; keep format time under 29 ms; measured parse time was 44 ms.",
            [
                goal("parse_time", 21),
                goal("format_time", 29),
                observed("parse_time", 44),
            ],
        ),
        (
            "In this rehearsal maintain write time under 33 ms and maintain read time under 17 ms.",
            [goal("write_time", 33), goal("read_time", 17)],
        ),
        (
            "Recorded write time was 46 ms; recorded read time was 28 ms in the rehearsal.",
            [observed("write_time", 46), observed("read_time", 28)],
        ),
        (
            "After the rehearsal, write time will stay below 38 ms; read time will stay below 24 ms.",
            [predicted("write_time", 38), predicted("read_time", 24)],
        ),
        (
            "Never update release-east; never modify release-west while rehearsing.",
            [policy("update", "release_east"), policy("modify", "release_west")],
        ),
        (
            "Measured index time was 65 ms; measured scan time was 71 ms; keep index time under 42 ms.",
            [
                observed("index_time", 65),
                observed("scan_time", 71),
                goal("index_time", 42),
            ],
        ),
    )
    for text, targets in repeated:
        add(text, targets, "repeated-kind")

    # Buried prohibitions appear between two other claims, in varying order.
    for i, subject in enumerate(
        ("commit time", "query delay", "update time", "packet delay", "reload time")
    ):
        s = subject.replace(" ", "_")
        add(
            f"Measured {subject} was {83 + i} ms and, if you would, please do not update vault-{91 + i}; keep {subject} under {49 + i} ms.",
            [observed(s, 83 + i), policy("update", f"vault_{91 + i}"), goal(s, 49 + i)],
            "polite-buried-constraint",
        )
        add(
            f"Keep {subject} at most {54 + i} ms; as a courtesy, please never modify ledger-{91 + i}; {subject} will remain below {65 + i} ms.",
            [
                goal(s, 54 + i, "LTE"),
                policy("modify", f"ledger_{91 + i}"),
                predicted(s, 65 + i),
            ],
            "polite-buried-constraint",
        )

    for i, subject in enumerate(
        ("handover delay", "encode time", "raster time", "route delay", "buffer time")
    ):
        s = subject.replace(" ", "_")
        targets = [goal(s, 42 + i), observed(s, 79 + i)]
        add(
            f"For tomorrow's rehearsal, keep {subject} under {42 + i} ms; measured {subject} was {79 + i} ms.",
            targets,
            "paraphrase-equivalence",
            f"v7-paraphrase-{i}",
        )
        add(
            f"During the last sample we recorded {subject} at {79 + i} ms. For tomorrow's rehearsal, maintain {subject} below {42 + i} ms.",
            targets,
            "paraphrase-equivalence",
            f"v7-paraphrase-{i}",
        )

    for i, target in enumerate(
        ("shared-config", "test-ledger", "draft-index", "pilot-store", "trial-record")
    ):
        t = target.replace("-", "_")
        add(
            f"For this exercise, do not update {target}; you must update {target}.",
            [policy("update", t), policy("update", t, positive=True)],
            "conflicting-constraints",
        )
        add(
            f"You must delete {target}; nevertheless, do not delete {target} during the exercise.",
            [policy("delete", t, positive=True), policy("delete", t)],
            "conflicting-constraints",
        )

    for i, subject in enumerate(
        ("lease time", "retry delay", "polling delay", "lock time", "socket delay")
    ):
        s = subject.replace(" ", "_")
        add(
            f"For this test, do not lower {subject} below {27 + i} ms; measured {subject} was {53 + i} ms.",
            [policy("lower", f"{s}_below_{27 + i}_ms"), observed(s, 53 + i)],
            "negation-composition",
            f"v7-negation-{i}",
        )
        add(
            f"For this test, lower {subject} below {27 + i} ms; measured {subject} was {53 + i} ms.",
            [goal(s, 27 + i), observed(s, 53 + i)],
            "negation-composition",
            f"v7-negation-{i}",
        )

    for i, subject in enumerate(
        ("egress delay", "ingress delay", "lookup time", "shutdown time", "settle time")
    ):
        s = subject.replace(" ", "_")
        targets = [
            goal(s, 38 + i),
            observed(s, 69 + i),
            policy("modify", f"reserved_{i + 61}"),
        ]
        clauses = [
            f"keep {subject} under {38 + i} ms",
            f"measured {subject} was {69 + i} ms",
            f"do not modify reserved-{i + 61}",
        ]
        add(
            "For this stage, " + "; ".join(clauses) + ".",
            targets,
            "order-swap-composition",
            f"v7-order-{i}",
        )
        add(
            "For this stage, " + "; ".join(reversed(clauses)) + ".",
            targets,
            "order-swap-composition",
            f"v7-order-{i}",
        )

    for i, subject in enumerate(
        (
            "resume time",
            "replay delay",
            "resolve time",
            "forward delay",
            "schedule time",
        )
    ):
        s = subject.replace(" ", "_")
        for variant in range(2):
            n, v = 29 + i + variant * 10, 58 + i + variant * 10
            add(
                f"In this scenario, keep {subject} below {n} ms and measured {subject} was {v} ms.",
                [goal(s, n), observed(s, v)],
                "number-change-composition",
                f"v7-number-{i}",
            )
    return rows


CALIBRATION_CONTROLS = (
    ("GOAL", "The desired result of the next exercise is a shorter checkout delay."),
    (
        "CONSTRAINT",
        "All customer account rows must remain unchanged through the rehearsal.",
    ),
    (
        "OBSERVATION",
        "The recorded checkout delay in the morning sample was 57 milliseconds.",
    ),
    (
        "PREDICTION",
        "The checkout delay is likely to drop to 44 milliseconds by evening.",
    ),
    ("GOAL", "Aim to bring the archive loading time down before the demonstration."),
    (
        "CONSTRAINT",
        "No edits to the release manifest are allowed during the demonstration.",
    ),
    ("OBSERVATION", "Our probe registered an archive loading time of 76 milliseconds."),
    (
        "PREDICTION",
        "The archive loading time should fall below 62 milliseconds tomorrow.",
    ),
    ("GOAL", "The objective for the pilot is a smaller backlog of incoming requests."),
    ("CONSTRAINT", "The saved customer snapshot must not be replaced in this pilot."),
    (
        "OBSERVATION",
        "The latest reading from the monitor showed a backlog of 34 requests.",
    ),
    (
        "PREDICTION",
        "We anticipate the backlog reaching 21 requests in the next interval.",
    ),
    ("GOAL", "Work toward a quicker response from the inventory endpoint."),
    ("CONSTRAINT", "The accounting database is outside the permitted write scope."),
    (
        "OBSERVATION",
        "The test log reports the inventory response time as 68 milliseconds.",
    ),
    (
        "PREDICTION",
        "The inventory response time is forecast to settle near 51 milliseconds.",
    ),
    ("GOAL", "Achieve a lower queueing delay during the upcoming trial."),
    ("CONSTRAINT", "Writes to the protected settings file are forbidden in the trial."),
    (
        "OBSERVATION",
        "The operator measured a queueing delay of 73 milliseconds in the trial.",
    ),
    (
        "PREDICTION",
        "Next trial's queueing delay will probably stay under 59 milliseconds.",
    ),
)


def verify_inputs() -> None:
    for path, digest in (
        (PREVIOUS_SUITE, PREVIOUS_SHA256),
        (PREVIOUS_MANIFEST, PREVIOUS_MANIFEST_SHA256),
        (HELD_OUT_PATH, HELD_OUT_SHA256),
        (CALIBRATION_PATH, CALIBRATION_SHA256),
    ):
        if file_sha256(path) != digest:
            raise ValueError(f"preserved input changed: {path}")


def historical_surfaces() -> tuple[set[str], set[str]]:
    texts = []
    for version in range(1, 7):
        suite = json.loads(
            (FROZEN / f"public-audit-v{version}-260.json").read_text(encoding="utf-8")
        )
        texts.extend(row["input"] for rows in suite["splits"].values() for row in rows)
    for path in sorted(FROZEN.glob("calibration-fit-v[12].jsonl")):
        texts.extend(
            json.loads(line)["text"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    failures = json.loads(
        (FROZEN / "frame-parser-known-failures-v1.json").read_text(encoding="utf-8")
    )

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            texts.extend(
                v
                for k, v in value.items()
                if k in ("input", "text") and isinstance(v, str)
            )
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(failures)
    return {normalized(x) for x in texts}, {template(x) for x in texts}


def build_suite() -> dict:
    verify_inputs()
    previous = json.loads(PREVIOUS_SUITE.read_text(encoding="utf-8"))
    retention = copy.deepcopy(previous["splits"]["retention"])
    for i, row in enumerate(retention):
        row["provenance"] = {
            "source_suite_sha256": PREVIOUS_SHA256,
            "source_item_id": row["id"],
            "status": "PRESERVED_RETENTION_NOT_FRESH",
            "training_use": "FORBIDDEN",
        }
        row["id"] = f"v7-retention-{i:03d}"
        row["surface_policy"] = "PRESERVED_RETENTION_NOT_FRESH"
    return {
        "schema": "kev.eval-suite.v1",
        "id": "kev-public-audit-v7-260",
        "frozen": True,
        "provenance": {
            "status": "FROZEN; create v8 rather than editing these bytes",
            "authored_without_training_corpus_access": True,
            "authored_without_v7_runtime_or_model_results": True,
            "review": "SAME_PARTY_SYNTHETIC_SEMANTIC_REVIEW_NOT_EXTERNAL_INDEPENDENT_EVALUATION",
            "independence_boundary": "isolated evaluation author; no training corpus, runtime changes, or model inference consulted",
            "target_source": "manually authored semantic frame judgments, never parser outputs",
            "retention_reuse_count": 40,
            "new_surface_count": 220,
            "prior_current_suite_sha256": PREVIOUS_SHA256,
            "held_out_vocabulary_sha256": HELD_OUT_SHA256,
            "calibration_fit_sha256": CALIBRATION_SHA256,
            "training_exclusion": "every promotion/retention/calibration surface and held-out term remains forbidden in weight training",
            "preserved_generations": list(range(1, 7)),
        },
        "splits": {
            "fresh": fresh_items(),
            "retention": retention,
            "oov": oov_items(),
            "composition": composition_items(),
            "calibration": [
                item("calibration", i, text, [kind], "calibration-semantic")
                for i, (kind, text) in enumerate(CALIBRATION_CONTROLS)
            ],
        },
    }


def validate_suite(suite: dict) -> dict:
    verify_inputs()
    if {name: len(rows) for name, rows in suite["splits"].items()} != SPLIT_COUNTS:
        raise ValueError("v7 split counts changed")
    rows = [row for group in suite["splits"].values() for row in group]
    if (
        len({row["id"] for row in rows}) != 260
        or len({normalized(row["input"]) for row in rows}) != 260
    ):
        raise ValueError("duplicate v7 identifiers or surfaces")
    exact, templates = historical_surfaces()
    for split, group in suite["splits"].items():
        for row in group:
            if split != "retention" and (
                normalized(row["input"]) in exact or template(row["input"]) in templates
            ):
                raise ValueError(
                    f"historical exact or number-only template overlap: {row['id']}"
                )
            kinds = [v if isinstance(v, str) else v["kind"] for v in row["expected"]]
            words = re.findall(r"[a-z0-9]+", row["input"].casefold())
            for kind in kinds:
                label = kind.casefold().split("_")
                if any(words[i : i + len(label)] == label for i in range(len(words))):
                    raise ValueError(f"expected label leaked in {row['id']}")
            for term in OOV_CONTROLS:
                if split != "oov" and re.search(rf"\b{term}\b", row["input"], re.I):
                    raise ValueError(f"held-out vocabulary outside OOV: {row['id']}")
    previous = json.loads(PREVIOUS_SUITE.read_text(encoding="utf-8"))["splits"][
        "retention"
    ]
    for old, new in zip(previous, suite["splits"]["retention"], strict=True):
        if any(
            old[key] != new[key]
            for key in ("input", "expected", "comparison", "projection", "audit")
        ):
            raise ValueError("retention content or target changed")
    for term in OOV_CONTROLS:
        if (
            sum(
                bool(re.search(rf"\b{term}\b", row["input"], re.I))
                for row in suite["splits"]["oov"]
            )
            != 4
        ):
            raise ValueError("OOV term must appear in exactly four cases")
    composition = suite["splits"]["composition"]
    audits = Counter(row["audit"] for row in composition)
    if audits != Counter({name: 10 for name in COMPOSITION_AUDITS}):
        raise ValueError("composition audit balance changed")
    if any(
        row["projection"] != "semantic_frames" or len(row["expected"]) < 2
        for row in composition
    ):
        raise ValueError("composition requires exact multiframe semantic targets")
    pairs: dict[str, list] = defaultdict(list)
    for row in rows:
        if row.get("pair_id"):
            pairs[row["pair_id"]].append(row)
    for name, pair in pairs.items():
        if len(pair) != 2:
            raise ValueError("paired audit does not have two cases")
        same = canonical_json_sha256(pair[0]["expected"]) == canonical_json_sha256(
            pair[1]["expected"]
        )
        if same != (name.startswith("v7-order") or name.startswith("v7-paraphrase")):
            raise ValueError("paired audit has incorrect semantic equality")
    repeated = [row for row in composition if row["audit"] == "repeated-kind"]
    if any(
        len({f["kind"] for f in row["expected"]}) == len(row["expected"])
        for row in repeated
    ):
        raise ValueError("repeated-kind audit lost frame multiplicity")
    return {
        "item_count": 260,
        "new_surfaces": 220,
        "preserved_retention": 40,
        "new_exact_historical_overlaps": 0,
        "new_digit_template_historical_overlaps": 0,
        "expected_label_leaks": 0,
        "held_out_terms_outside_oov": 0,
        "all_composition_targets_exact_semantic_frames": True,
        "composition_audit_counts": dict(sorted(audits.items())),
        "pair_count": len(pairs),
        "review_basis": "AUTHOR_SEMANTIC_REVIEW_AND_STATIC_INVARIANTS_NO_MODEL_INFERENCE",
    }


def build_manifest(suite: dict | None = None) -> dict:
    if suite is None:
        suite = build_suite()
    audit = validate_suite(suite)
    raw = json_bytes(suite)
    prior = json.loads(PREVIOUS_MANIFEST.read_text(encoding="utf-8"))
    artifacts = copy.deepcopy(prior["artifacts"])
    artifacts["preserved_promotion_suite_v6"] = artifacts.pop("promotion_suite")
    artifacts["preserved_promotion_suite_v6"]["status"] = (
        "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE"
    )
    artifacts["preserved_manifest_v6"] = {
        "path": PREVIOUS_MANIFEST.relative_to(ROOT).as_posix(),
        "sha256": PREVIOUS_MANIFEST_SHA256,
        "size_bytes": PREVIOUS_MANIFEST.stat().st_size,
    }
    artifacts["promotion_suite"] = {
        "path": SUITE_PATH.relative_to(ROOT).as_posix(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "size_bytes": len(raw),
        "canonical_sha256": canonical_json_sha256(suite),
    }
    return {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 7,
        "preserved_generations": list(range(1, 7)),
        "artifacts": artifacts,
        "mutation_policy": "create a new version; never refresh v1 through v7 in place",
        "training_boundary": copy.deepcopy(prior["training_boundary"]),
        "freshness_boundary": audit,
        "review_boundary": "SAME_PARTY_SYNTHETIC_NOT_INDEPENDENT_EXTERNAL_EVALUATION",
    }


def main() -> None:
    if any(path.exists() for path in (SUITE_PATH, MANIFEST_PATH)):
        raise SystemExit("v7 artifacts already exist; refusing to refresh them")
    suite = build_suite()
    manifest = build_manifest(suite)
    for path, value in ((SUITE_PATH, suite), (MANIFEST_PATH, manifest)):
        with path.open("xb") as handle:
            handle.write(json_bytes(value))
    print(
        json.dumps(
            {
                "suite_sha256": file_sha256(SUITE_PATH),
                "suite_canonical_sha256": canonical_json_sha256(suite),
                "manifest_sha256": file_sha256(MANIFEST_PATH),
                "audit": manifest["freshness_boundary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

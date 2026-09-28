"""Build the permission-bearing synthetic corpus for KEV's first v5 experiment.

This builder is intentionally blind to the v5 promotion suite.  Evaluation and
training are authored on separate branches of the experiment: this file reads
only the already development-influenced v1-v4 surfaces plus the vocabulary-only
v2 holdout supplied by the evaluation owner.  The normal trainer performs the
final all-suite exclusion check when a challenger is eventually trained.

The corpus is repository-authored synthetic material.  It is reviewed by the
same Codex agent that authored it under the user's explicit authorization for
this experiment.  It is *not* represented as independent human review, and no
ordinary chat transcript is copied into a lesson.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from kev.artifacts import canonical_json_sha256, file_sha256, validate_sha256
from kev.frames import FRAME_KINDS
from kev.uc51a2.semantic_breadth import _reviewed_rows_from_text


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "training" / "reviewed"
CORPUS_NAME = "semantic-frame-paraphrases-v5-reviewed.jsonl"
MANIFEST_NAME = "semantic-frame-paraphrases-v5-manifest.json"
REVIEWED_AT = "2026-09-28T11:06:09Z"
REVIEWER = "openai-codex-agent:reviewed-corpus-task"
PERMISSION = (
    "repository-authored synthetic text; user-authorized for the KEV v5 local "
    "training experiment; no third-party text; not independent human review"
)
PINNED_V1_VOCABULARY = ROOT / "evals" / "frozen" / "held-out-vocabulary-v1.txt"
PINNED_V1_VOCABULARY_SHA256 = (
    "e7b7c7ec703e378a9d3ff419b00f9b67e370c07ff2c773d8f6262ec83f2dba3a"
)
PINNED_V2_VOCABULARY = ROOT / "evals" / "frozen" / "held-out-vocabulary-v2.txt"
PINNED_V2_VOCABULARY_SHA256 = (
    "087da62e8129dd259c1b2662f0439eb7d6387714528af14444a4575c66e76798"
)
V5_SUITE_FILE_SHA256_PROVENANCE_ONLY = (
    "e913f483bdce2e3ea1a435f0c139a846874a73660665c69b8723a9670348451c"
)
V5_SUITE_CANONICAL_SHA256_PROVENANCE_ONLY = (
    "1cb5431b83e93f284bb1277ceea4c6e27ad1249f6a5c6fb24df4e22371e42698"
)

# These are deliberately explicit rather than discovered with a glob.  Adding a
# v5 suite must not make this independently authored builder read that suite.
V1_V4_SURFACE_ARTIFACTS: tuple[tuple[str, str, str], ...] = (
    (
        "evals/frozen/public-audit-v1-260.json",
        "1c5889428acb0920786f813331f5188b63393d63ec64986c123350218c8b58f0",
        "EVALUATION_SUITE",
    ),
    (
        "evals/frozen/public-audit-v2-260.json",
        "ef9ff61967f8716e541ccdef3adea887da99dc988799098a9faf47e2ea3dd3c4",
        "EVALUATION_SUITE",
    ),
    (
        "evals/frozen/public-audit-v3-260.json",
        "1bce1b6c1123336e30531b3cf54c1d01ff4aa420cfec0cb078eaca7f40f3b2d8",
        "EVALUATION_SUITE",
    ),
    (
        "evals/frozen/public-audit-v4-260.json",
        "6e3ef02d4abc1101ee95d069e80e51963baa911b0ebd4305e9ec536ba0c57322",
        "EVALUATION_SUITE",
    ),
    (
        "evals/frozen/calibration-fit-v1.jsonl",
        "5734adb2045fa9d192888a2932c2821842833f6a17d9bc27db8bd5defe37e94c",
        "CALIBRATION_FIT",
    ),
    (
        "evals/frozen/frame-parser-known-failures-v1.json",
        "1aa0d4b519bfdd96e1874f25a0f814149a3b48f51dc8e38274f3847a46a6a901",
        "DEVELOPMENT_INFLUENCED_FAILURES",
    ),
)


def _slot(value: Any, slot_type: str) -> dict[str, Any]:
    return {"type": slot_type, "value": value}


def _threshold(
    kind: str,
    subject: str,
    operator: str,
    value: int | float,
    unit: str,
) -> dict[str, Any]:
    relation = "FORECAST" if kind == "PREDICTION" else "THRESHOLD"
    return {
        "kind": kind,
        "relation": relation,
        "slots": {
            "operator": _slot(operator, "operator"),
            "subject": _slot(subject, "entity"),
            "unit": _slot(unit, "unit"),
            "value": _slot(value, "number"),
        },
    }


def _directive(action: str, object_: str) -> dict[str, Any]:
    return {
        "kind": "GOAL",
        "relation": "DIRECTIVE",
        "slots": {
            "action": _slot(action, "action"),
            "object": _slot(object_, "entity"),
        },
    }


def _policy(action: str, object_: str, polarity: bool) -> dict[str, Any]:
    return {
        "kind": "CONSTRAINT",
        "relation": "ACTION_POLICY",
        "slots": {
            "action": _slot(action, "action"),
            "modality": _slot("REQUIRED" if polarity else "FORBIDDEN", "modality"),
            "object": _slot(object_, "entity"),
            "polarity": _slot(polarity, "boolean"),
        },
    }


def _measurement(subject: str, value: int | float, unit: str) -> dict[str, Any]:
    return {
        "kind": "OBSERVATION",
        "relation": "MEASUREMENT",
        "slots": {
            "subject": _slot(subject, "entity"),
            "unit": _slot(unit, "unit"),
            "value": _slot(value, "number"),
        },
    }


def _forecast(
    subject: str,
    operator: str,
    value: int | float,
    unit: str,
) -> dict[str, Any]:
    return _threshold("PREDICTION", subject, operator, value, unit)


def _base(kind: str, **slots: tuple[Any, str]) -> dict[str, Any]:
    return {
        "kind": kind,
        "relation": kind,
        "slots": {
            name: _slot(value, slot_type)
            for name, (value, slot_type) in sorted(slots.items())
        },
    }


def _normal_surface(text: str) -> str:
    return text.strip().casefold()


def _vocabulary(path: Path, expected_sha256: str) -> tuple[set[str], dict[str, Any]]:
    expected = validate_sha256(expected_sha256, field=f"expected SHA-256 for {path}")
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(
            f"held-out vocabulary SHA-256 mismatch for {path}: expected {expected}, got {actual}"
        )
    terms = {
        line.strip().casefold()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not terms:
        raise ValueError(f"held-out vocabulary is empty: {path}")
    invalid = sorted(
        term for term in terms if not re.fullmatch(r"[a-z][a-z0-9_-]*", term)
    )
    if invalid:
        raise ValueError(
            f"held-out vocabulary must contain one token per line: {invalid}"
        )
    return terms, {
        "schema": "kev.artifact-ref.v1",
        "path": path.name,
        "sha256": actual,
        "size_bytes": path.stat().st_size,
        "term_count": len(terms),
        "terms_canonical_sha256": canonical_json_sha256(sorted(terms)),
    }


def _suite_surfaces(value: Mapping[str, Any]) -> set[str]:
    surfaces: set[str] = set()
    splits = value.get("splits")
    if not isinstance(splits, Mapping):
        raise ValueError("frozen suite is missing splits")
    for rows in splits.values():
        if not isinstance(rows, list):
            raise ValueError("frozen suite split is not a list")
        for row in rows:
            if isinstance(row, Mapping):
                text = str(row.get("input", row.get("text", ""))).strip()
                if text:
                    surfaces.add(_normal_surface(text))
    return surfaces


def _known_failure_surfaces(value: Mapping[str, Any]) -> set[str]:
    surfaces: set[str] = set()
    cases = value.get("cases")
    if not isinstance(cases, list):
        raise ValueError("known-failure artifact is missing cases")
    for case in cases:
        if not isinstance(case, Mapping):
            continue
        primary = str(case.get("input", "")).strip()
        if primary:
            surfaces.add(_normal_surface(primary))
        reverse = case.get("reverse_control")
        if isinstance(reverse, Mapping):
            control = str(reverse.get("input", "")).strip()
            if control:
                surfaces.add(_normal_surface(control))
    return surfaces


def _frozen_v1_v4_surfaces() -> tuple[set[str], list[dict[str, Any]]]:
    surfaces: set[str] = set()
    references: list[dict[str, Any]] = []
    for relative, expected, role in V1_V4_SURFACE_ARTIFACTS:
        path = ROOT / relative
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"preserved training exclusion changed at {relative}: "
                f"expected {expected}, got {actual}"
            )
        raw = path.read_bytes()
        if role == "CALIBRATION_FIT":
            for line in raw.decode("utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                text = str(row.get("text", row.get("input", ""))).strip()
                if text:
                    surfaces.add(_normal_surface(text))
        else:
            value = json.loads(raw.decode("utf-8"))
            surfaces.update(
                _known_failure_surfaces(value)
                if role == "DEVELOPMENT_INFLUENCED_FAILURES"
                else _suite_surfaces(value)
            )
        references.append(
            {
                "schema": "kev.artifact-ref.v1",
                "path": relative,
                "role": role,
                "sha256": actual,
                "size_bytes": len(raw),
            }
        )
    return surfaces, references


class _Corpus:
    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []

    def add(
        self,
        group: str,
        texts: Sequence[str],
        frames: Sequence[Mapping[str, Any]],
        *phenomena: str,
    ) -> None:
        if len(texts) < 2:
            raise ValueError(f"paraphrase group {group!r} needs at least two rows")
        frame_list = [deepcopy(dict(frame)) for frame in frames]
        kinds = list(dict.fromkeys(str(frame["kind"]) for frame in frame_list))
        for index, text in enumerate(texts, 1):
            self.rows.append(
                {
                    "schema": "kev.reviewed-lesson.v1",
                    "id": f"v5-{group}-{index:02d}",
                    "status": "REVIEWED",
                    "reviewed_by": REVIEWER,
                    "reviewed_at": REVIEWED_AT,
                    "permission": PERMISSION,
                    "text": text,
                    "frame_kinds": kinds,
                    "frame_cardinality": len(frame_list),
                    "frames": deepcopy(frame_list),
                    "paraphrase_group": group,
                    "training_tags": sorted(set(phenomena)),
                    "authorship": {
                        "source_type": "REPOSITORY_AUTHORED_SYNTHETIC",
                        "authored_by": "openai-codex-agent:reviewed-corpus-task",
                        "authorization": "user-authorized KEV v5 experiment",
                        "ordinary_chat_source": False,
                        "third_party_source": False,
                    },
                    "review": {
                        "review_type": "SAME_PARTY_AGENT_REVIEW",
                        "independent_human_review": False,
                        "scope": "surface text and exact typed frame-set target",
                    },
                }
            )


def _add_single_frame_groups(corpus: _Corpus) -> None:
    single_groups: tuple[
        tuple[str, tuple[str, ...], tuple[dict[str, Any], ...], tuple[str, ...]], ...
    ] = (
        (
            "goal-queue-42",
            (
                "Keep queue delay under 42 ms.",
                "Our aim is to hold queue delay below 42 milliseconds.",
                "We want queue delay to stay less than 42 ms.",
            ),
            (_threshold("GOAL", "queue_delay", "LT", 42, "ms"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "goal-export-88",
            (
                "Maintain export rate at no less than 88 rps.",
                "Set export rate to at least 88 requests per second.",
                "Our target is to keep export rate above or equal to 88 rps.",
            ),
            (_threshold("GOAL", "export_rate", "GTE", 88, "rps"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "goal-sandbox-throughput",
            (
                "Please improve sandbox throughput.",
                "Improve the sandbox throughput, please.",
                "Could you please improve sandbox throughput?",
            ),
            (_directive("improve", "sandbox_throughput"),),
            ("single-frame", "paraphrase", "polite"),
        ),
        (
            "constraint-staging-registry",
            (
                "Do not modify the staging registry.",
                "The staging registry must not be modified.",
                "Please keep the staging registry unchanged.",
            ),
            (_policy("modify", "staging_registry", False),),
            ("single-frame", "negation", "polite"),
        ),
        (
            "constraint-test-catalog",
            (
                "The test catalog must be updated.",
                "You are required to update the test catalog.",
                "Please ensure that updating the test catalog is required.",
            ),
            (_policy("update", "test_catalog", True),),
            ("single-frame", "positive-policy", "polite"),
        ),
        (
            "constraint-memory-640",
            (
                "Memory use must be at most 640 MB.",
                "The memory use shall not exceed 640 MB.",
                "Please keep memory use no more than 640 MB.",
            ),
            (_threshold("CONSTRAINT", "memory_use", "LTE", 640, "mb"),),
            ("single-frame", "number-control", "polite"),
        ),
        (
            "observation-queue-67",
            (
                "The measured queue delay was 67 ms.",
                "Queue delay was recorded at 67 milliseconds.",
                "We observed queue delay at 67 ms.",
            ),
            (_measurement("queue_delay", 67, "ms"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "observation-export-91",
            (
                "The observed export rate was 91 rps.",
                "Export rate has been measured at 91 requests per second.",
                "We recorded export rate at 91 rps.",
            ),
            (_measurement("export_rate", 91, "rps"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "observation-disk-58",
            (
                "Actual disk use was 58 percent.",
                "Disk use was observed at 58%.",
                "The recorded disk use is 58 percent.",
            ),
            (_measurement("disk_use", 58, "%"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "prediction-queue-49",
            (
                "We expect queue delay to be 49 ms.",
                "Queue delay will reach 49 milliseconds.",
                "The forecast has queue delay at 49 ms.",
            ),
            (_forecast("queue_delay", "EQ", 49, "ms"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "prediction-export-96",
            (
                "We predict export rate will be at least 96 rps.",
                "Export rate is expected to remain no less than 96 requests per second.",
                "The forecast anticipates export rate above or equal to 96 rps.",
            ),
            (_forecast("export_rate", "GTE", 96, "rps"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "prediction-cache-320",
            (
                "We anticipate cache size will stay at most 320 MB.",
                "Cache size is likely to remain no more than 320 MB.",
                "The forecast puts cache size below or equal to 320 MB.",
            ),
            (_forecast("cache_size", "LTE", 320, "mb"),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "copy-ember-cedar",
            (
                "Copy the ember payload into the cedar buffer.",
                "Put a copy of the ember payload in the cedar buffer.",
                "The cedar buffer should receive a copy of the ember payload.",
            ),
            (
                _base(
                    "COPY_VALUE",
                    source=("ember_payload", "value"),
                    destination=("cedar_buffer", "reference"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "copy-north-south",
            (
                "Copy the north value to the south record.",
                "Place a copy of the north value into the south record.",
                "The south record needs a copy of the north value.",
            ),
            (
                _base(
                    "COPY_VALUE",
                    source=("north_value", "value"),
                    destination=("south_record", "reference"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "copy-trial-audit",
            (
                "Copy the trial metric into the audit slot.",
                "Store a copy of the trial metric in the audit slot.",
                "The audit slot should get a copy of the trial metric.",
            ),
            (
                _base(
                    "COPY_VALUE",
                    source=("trial_metric", "value"),
                    destination=("audit_slot", "reference"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "supersedes-sable-azure",
            (
                "Release azure supersedes release sable.",
                "Replace release sable with release azure.",
                "Release sable is retired; release azure is now current.",
            ),
            (
                _base(
                    "SUPERSEDES",
                    old_value=("release_sable", "value"),
                    current_value=("release_azure", "value"),
                ),
            ),
            ("single-frame", "paraphrase", "revision"),
        ),
        (
            "supersedes-policy-r3",
            (
                "Policy r3 replaces policy r2.",
                "Replace policy r2 with policy r3.",
                "Policy r2 is obsolete; policy r3 applies now.",
            ),
            (
                _base(
                    "SUPERSEDES",
                    old_value=("policy_r2", "value"),
                    current_value=("policy_r3", "value"),
                ),
            ),
            ("single-frame", "paraphrase", "revision"),
        ),
        (
            "supersedes-seed-b",
            (
                "Model seed b supersedes model seed a.",
                "Replace model seed a with model seed b.",
                "Model seed a is former; model seed b is now current.",
            ),
            (
                _base(
                    "SUPERSEDES",
                    old_value=("model_seed_a", "value"),
                    current_value=("model_seed_b", "value"),
                ),
            ),
            ("single-frame", "paraphrase", "revision"),
        ),
        (
            "magnitude-minus-17",
            (
                "Find the magnitude of -17.",
                "What is the absolute value of -17?",
                "Give the distance from zero for -17.",
            ),
            (_base("MAGNITUDE", input=(-17, "number")),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "magnitude-minus-29",
            (
                "Find the magnitude of -29.",
                "What is the absolute value of -29?",
                "Give the distance from zero for -29.",
            ),
            (_base("MAGNITUDE", input=(-29, "number")),),
            ("single-frame", "paraphrase", "number-change"),
        ),
        (
            "magnitude-plus-34",
            (
                "Find the magnitude of 34.",
                "What is the absolute value of 34?",
                "Give the unsigned value of 34.",
            ),
            (_base("MAGNITUDE", input=(34, "number")),),
            ("single-frame", "paraphrase", "number-change"),
        ),
        (
            "negate-13",
            (
                "Negate 13.",
                "Return the opposite of 13.",
                "Reverse the sign of 13.",
            ),
            (_base("NEGATE", input=(13, "number")),),
            ("single-frame", "paraphrase", "number-control"),
        ),
        (
            "negate-minus-21",
            (
                "Negate -21.",
                "Return the opposite of -21.",
                "Reverse the sign of -21.",
            ),
            (_base("NEGATE", input=(-21, "number")),),
            ("single-frame", "paraphrase", "number-change"),
        ),
        (
            "negate-8",
            (
                "Negate 8.",
                "Return the opposite of 8.",
                "Give the sign-reversed form of 8.",
            ),
            (_base("NEGATE", input=(8, "number")),),
            ("single-frame", "paraphrase", "number-change"),
        ),
        (
            "reference-issue-184",
            (
                "Refer to issue-184.",
                "Use issue-184 as the reference.",
                "The cited record should be issue-184.",
            ),
            (_base("REFERENCE", target=("issue-184", "reference")),),
            ("single-frame", "paraphrase"),
        ),
        (
            "reference-delta-7",
            (
                "Refer to record-delta-7.",
                "Use record-delta-7 as the reference.",
                "The cited record should be record-delta-7.",
            ),
            (_base("REFERENCE", target=("record-delta-7", "reference")),),
            ("single-frame", "paraphrase"),
        ),
        (
            "reference-note-52",
            (
                "Refer to note-52.",
                "Use note-52 as the reference.",
                "The cited record should be note-52.",
            ),
            (_base("REFERENCE", target=("note-52", "reference")),),
            ("single-frame", "paraphrase"),
        ),
        (
            "active-coral-3",
            (
                "Candidate coral-3 is active.",
                "Candidate coral-3 is serving.",
                "The active selection is candidate coral-3.",
            ),
            (
                _base(
                    "ACTIVE_SELECTION",
                    selection=("candidate_coral-3", "identifier"),
                    active=(True, "boolean"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "active-maple-8",
            (
                "Release maple-8 is active.",
                "Release maple-8 is serving.",
                "The selected release is maple-8.",
            ),
            (
                _base(
                    "ACTIVE_SELECTION",
                    selection=("release_maple-8", "identifier"),
                    active=(True, "boolean"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "active-silver-2",
            (
                "Profile silver-2 is active.",
                "Profile silver-2 is live.",
                "The selected profile is silver-2.",
            ),
            (
                _base(
                    "ACTIVE_SELECTION",
                    selection=("profile_silver-2", "identifier"),
                    active=(True, "boolean"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "run-trial-17",
            (
                "Run trial-17 completed.",
                "The status of run trial-17 is completed.",
                "Run trial-17 has a completed status.",
            ),
            (
                _base(
                    "RUN_STATUS",
                    run_id=("trial-17", "identifier"),
                    status=("COMPLETED", "status"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "run-probe-22",
            (
                "Run probe-22 failed.",
                "The status of run probe-22 is failed.",
                "Run probe-22 has a failed status.",
            ),
            (
                _base(
                    "RUN_STATUS",
                    run_id=("probe-22", "identifier"),
                    status=("FAILED", "status"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "run-sample-31",
            (
                "Run sample-31 is running.",
                "The status of run sample-31 is running.",
                "Run sample-31 has a running status.",
            ),
            (
                _base(
                    "RUN_STATUS",
                    run_id=("sample-31", "identifier"),
                    status=("RUNNING", "status"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "receipt-rec-31",
            (
                "Receipt rec-31 returned 204.",
                "Receipt rec-31 records value 204.",
                "The result on receipt rec-31 is 204.",
            ),
            (
                _base(
                    "RECEIPT_VALUE",
                    receipt_id=("rec-31", "identifier"),
                    value=(204, "number"),
                ),
            ),
            ("single-frame", "paraphrase"),
        ),
        (
            "receipt-rec-44",
            (
                "Receipt rec-44 returned 12.",
                "Receipt rec-44 records value 12.",
                "The result on receipt rec-44 is 12.",
            ),
            (
                _base(
                    "RECEIPT_VALUE",
                    receipt_id=("rec-44", "identifier"),
                    value=(12, "number"),
                ),
            ),
            ("single-frame", "paraphrase", "number-change"),
        ),
        (
            "receipt-rec-58",
            (
                "Receipt rec-58 returned 37.",
                "Receipt rec-58 records value 37.",
                "The result on receipt rec-58 is 37.",
            ),
            (
                _base(
                    "RECEIPT_VALUE",
                    receipt_id=("rec-58", "identifier"),
                    value=(37, "number"),
                ),
            ),
            ("single-frame", "paraphrase", "number-change"),
        ),
        (
            "evidence-consistent-alpha",
            (
                "The evidence for sample alpha is consistent.",
                "The sample alpha evidence items are in agreement.",
                "For sample alpha, the evidence shows consistent findings.",
            ),
            (_base("EVIDENCE_CONSISTENCY", status=("CONSISTENT", "status")),),
            ("single-frame", "paraphrase"),
        ),
        (
            "evidence-conflict-beta",
            (
                "The evidence for sample beta is conflicting.",
                "The sample beta evidence items are in conflict.",
                "For sample beta, the evidence indicates disagreement.",
            ),
            (_base("EVIDENCE_CONSISTENCY", status=("CONFLICT", "status")),),
            ("single-frame", "paraphrase", "conflict"),
        ),
        (
            "evidence-consistent-gamma",
            (
                "The evidence for sample gamma is consistent.",
                "The sample gamma evidence items are in agreement.",
                "For sample gamma, the evidence indicates consistent results.",
            ),
            (_base("EVIDENCE_CONSISTENCY", status=("CONSISTENT", "status")),),
            ("single-frame", "paraphrase"),
        ),
    )
    for group, texts, frames, tags in single_groups:
        corpus.add(group, texts, frames, *tags)


def _phrase(symbol: str) -> str:
    return symbol.replace("_", " ")


def _add_goal_constraint(
    corpus: _Corpus,
    index: int,
    subject: str,
    target: int,
    object_: str,
) -> None:
    subject_text, object_text = _phrase(subject), _phrase(object_)
    frames = (
        _threshold("GOAL", subject, "LT", target, "ms"),
        _policy("modify", object_, False),
    )
    corpus.add(
        f"composition-goal-constraint-{index:02d}",
        (
            f"Keep {subject_text} below {target} ms, and do not modify {object_text}.",
            f"Do not modify {object_text}; keep {subject_text} under {target} milliseconds.",
            f"Could you please keep {subject_text} less than {target} ms without ever modifying {object_text}?",
            f"While {object_text} remains unchanged, our aim is {subject_text} below {target} ms.",
        ),
        frames,
        "multi-frame",
        "order-swap",
        "negation",
        "polite-buried-constraint",
        "number-control",
    )


def _add_goal_observation(
    corpus: _Corpus,
    index: int,
    subject: str,
    target: int,
    observed: int,
) -> None:
    surface = _phrase(subject)
    frames = (
        _threshold("GOAL", subject, "LT", target, "ms"),
        _measurement(subject, observed, "ms"),
    )
    corpus.add(
        f"composition-goal-observation-{index:02d}",
        (
            f"Keep {surface} below {target} ms; the measured {surface} was {observed} ms.",
            f"The measured {surface} was {observed} milliseconds, while our aim is below {target} ms.",
            f"Please target {surface} under {target} ms, given that we observed {surface} at {observed} ms.",
            f"We recorded {surface} at {observed} ms and want {surface} less than {target} ms.",
        ),
        frames,
        "multi-frame",
        "order-swap",
        "number-control",
        "polite",
    )


def _add_observation_prediction(
    corpus: _Corpus,
    index: int,
    subject: str,
    observed: int,
    predicted: int,
) -> None:
    surface = _phrase(subject)
    frames = (
        _measurement(subject, observed, "ms"),
        _forecast(subject, "EQ", predicted, "ms"),
    )
    corpus.add(
        f"composition-observation-prediction-{index:02d}",
        (
            f"The measured {surface} was {observed} ms, and we expect {surface} to reach {predicted} ms.",
            f"We predict {surface} will be {predicted} milliseconds; it was observed at {observed} ms.",
            f"After recording {surface} at {observed} ms, the forecast puts {surface} at {predicted} ms.",
            f"{surface.capitalize()} will reach {predicted} ms, while the actual reading was {observed} ms.",
        ),
        frames,
        "multi-frame",
        "order-swap",
        "number-control",
    )


def _add_triple(
    corpus: _Corpus,
    index: int,
    subject: str,
    target: int,
    observed: int,
    object_: str,
) -> None:
    surface, object_text = _phrase(subject), _phrase(object_)
    frames = (
        _threshold("GOAL", subject, "LT", target, "ms"),
        _policy("modify", object_, False),
        _measurement(subject, observed, "ms"),
    )
    corpus.add(
        f"composition-triple-{index:02d}",
        (
            f"Keep {surface} below {target} ms; do not modify {object_text}; measured {surface} was {observed} ms.",
            f"Measured {surface} was {observed} ms, {object_text} must remain unchanged, and our aim is below {target} ms.",
            f"Could you please target {surface} under {target} ms without ever modifying {object_text}, given the observed {observed} ms reading?",
            f"Do not modify {object_text}; we recorded {surface} at {observed} ms; keep it less than {target} ms.",
        ),
        frames,
        "multi-frame",
        "order-swap",
        "polite-buried-constraint",
        "negation",
        "number-control",
    )


def _add_quad(
    corpus: _Corpus,
    index: int,
    subject: str,
    target: int,
    observed: int,
    predicted: int,
    object_: str,
) -> None:
    surface, object_text = _phrase(subject), _phrase(object_)
    frames = (
        _threshold("GOAL", subject, "LT", target, "ms"),
        _policy("modify", object_, False),
        _measurement(subject, observed, "ms"),
        _forecast(subject, "EQ", predicted, "ms"),
    )
    corpus.add(
        f"composition-quad-{index:02d}",
        (
            f"Keep {surface} below {target} ms; never modify {object_text}; measured {surface} was {observed} ms; we expect {predicted} ms next.",
            f"We expect {surface} at {predicted} ms after observing {observed} ms, while {object_text} stays unchanged and the target remains below {target} ms.",
            f"Could you please aim for {surface} under {target} ms without modifying {object_text}? The observed reading was {observed} ms and the forecast is {predicted} ms.",
            f"The recorded {surface} is {observed} ms; predicted {surface} is {predicted} ms; do not modify {object_text}; keep the target below {target} ms.",
        ),
        frames,
        "multi-frame",
        "order-swap",
        "polite-buried-constraint",
        "negation",
        "number-control",
    )


def _add_conflicting_constraints(
    corpus: _Corpus,
    index: int,
    object_: str,
) -> None:
    object_text = _phrase(object_)
    frames = (
        _policy("update", object_, True),
        _policy("update", object_, False),
    )
    corpus.add(
        f"composition-conflicting-constraints-{index:02d}",
        (
            f"The rehearsal says to update {object_text}, but the safeguard says never update {object_text}.",
            f"Do not update {object_text}, although the rehearsal requires updating {object_text}.",
            f"Please preserve both rules: update {object_text}, and do not update {object_text}.",
            f"Updating {object_text} is required by one rule and forbidden by another.",
        ),
        frames,
        "multi-frame",
        "repeated-kind",
        "conflicting-constraints",
        "order-swap",
        "negation",
    )


def _add_two_goals(
    corpus: _Corpus,
    index: int,
    first_subject: str,
    first_value: int,
    second_subject: str,
    second_value: int,
) -> None:
    first, second = _phrase(first_subject), _phrase(second_subject)
    frames = (
        _threshold("GOAL", first_subject, "LT", first_value, "ms"),
        _threshold("GOAL", second_subject, "LT", second_value, "ms"),
    )
    corpus.add(
        f"composition-two-goals-{index:02d}",
        (
            f"Keep {first} below {first_value} ms and {second} below {second_value} ms.",
            f"Our aims are {second} under {second_value} ms plus {first} under {first_value} ms.",
            f"Please target less than {first_value} ms for {first}, while holding {second} below {second_value} ms.",
            f"Set {second} below {second_value} milliseconds; also set {first} below {first_value} ms.",
        ),
        frames,
        "multi-frame",
        "repeated-kind",
        "order-swap",
        "number-change",
        "polite",
    )


def _add_composition_groups(corpus: _Corpus) -> None:
    for index, values in enumerate(
        (
            ("render_delay", 31, "staging_index"),
            ("commit_delay", 37, "test_archive"),
            ("search_delay", 43, "preview_catalog"),
            ("sync_delay", 47, "sandbox_ledger"),
        ),
        1,
    ):
        _add_goal_constraint(corpus, index, *values)
    for index, values in enumerate(
        (
            ("render_delay", 35, 74),
            ("commit_delay", 41, 79),
            ("search_delay", 46, 83),
            ("sync_delay", 52, 89),
        ),
        1,
    ):
        _add_goal_observation(corpus, index, *values)
    for index, values in enumerate(
        (
            ("render_delay", 72, 55),
            ("commit_delay", 77, 59),
            ("search_delay", 81, 63),
        ),
        1,
    ):
        _add_observation_prediction(corpus, index, *values)
    for index, values in enumerate(
        (
            ("render_delay", 34, 71, "staging_index"),
            ("commit_delay", 40, 76, "test_archive"),
            ("search_delay", 45, 82, "preview_catalog"),
        ),
        1,
    ):
        _add_triple(corpus, index, *values)
    for index, values in enumerate(
        (
            ("render_delay", 32, 70, 54, "staging_index"),
            ("commit_delay", 38, 75, 58, "test_archive"),
        ),
        1,
    ):
        _add_quad(corpus, index, *values)
    _add_conflicting_constraints(corpus, 1, "staging_catalog")
    _add_conflicting_constraints(corpus, 2, "preview_registry")
    _add_two_goals(corpus, 1, "render_delay", 29, "commit_delay", 36)
    _add_two_goals(corpus, 2, "search_delay", 44, "sync_delay", 50)


def build_rows() -> list[dict[str, Any]]:
    corpus = _Corpus()
    _add_single_frame_groups(corpus)
    _add_composition_groups(corpus)
    return corpus.rows


def corpus_bytes(rows: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(
        json.dumps(
            row,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
        for row in rows
    )


def _words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.casefold()))


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out_terms: Iterable[str],
    forbidden_surfaces: Iterable[str],
) -> None:
    if not rows:
        raise ValueError("corpus is empty")
    identifiers = [str(row.get("id", "")) for row in rows]
    if any(not identifier for identifier in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("lesson identifiers must be non-empty and unique")
    texts = [str(row.get("text", "")) for row in rows]
    normalized = [_normal_surface(text) for text in texts]
    if len(normalized) != len(set(normalized)):
        raise ValueError("lesson surfaces must be unique")

    forbidden = {_normal_surface(text) for text in forbidden_surfaces if text.strip()}
    collision = sorted(set(normalized) & forbidden)
    if collision:
        raise ValueError(f"corpus overlaps preserved frozen surfaces: {collision}")
    held_out = {term.strip().casefold() for term in held_out_terms if term.strip()}
    leaked = sorted({term for text in texts for term in (_words(text) & held_out)})
    if leaked:
        raise ValueError(f"corpus contains held-out vocabulary: {leaked}")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    seen_kinds: set[str] = set()
    cardinalities: set[int] = set()
    tags: set[str] = set()
    for row in rows:
        if row.get("status") != "REVIEWED":
            raise ValueError("every corpus row must be explicitly REVIEWED")
        if row.get("reviewed_by") != REVIEWER or row.get("reviewed_at") != REVIEWED_AT:
            raise ValueError("review provenance changed")
        if row.get("permission") != PERMISSION:
            raise ValueError("permission provenance changed")
        authorship = row.get("authorship")
        review = row.get("review")
        if not isinstance(authorship, Mapping) or any(
            authorship.get(field) is not False
            for field in ("ordinary_chat_source", "third_party_source")
        ):
            raise ValueError("corpus provenance must exclude chat and third-party text")
        if (
            not isinstance(review, Mapping)
            or review.get("independent_human_review") is not False
        ):
            raise ValueError("corpus must not claim independent human review")
        frames = row.get("frames")
        if not isinstance(frames, list) or not frames:
            raise ValueError("every corpus row needs exact structured frame targets")
        if row.get("frame_cardinality") != len(frames):
            raise ValueError("declared frame cardinality does not match frames")
        if len(frames) > 6:
            raise ValueError("frame cardinality exceeds model support")
        kinds = [
            str(frame.get("kind")) for frame in frames if isinstance(frame, Mapping)
        ]
        if len(kinds) != len(frames) or set(row.get("frame_kinds", ())) != set(kinds):
            raise ValueError("frame kind targets disagree with structured frames")
        seen_kinds.update(kinds)
        cardinalities.add(len(frames))
        tags.update(str(tag) for tag in row.get("training_tags", ()))
        group = str(row.get("paraphrase_group", ""))
        if not group:
            raise ValueError("every row must belong to an explicit paraphrase group")
        groups.setdefault(group, []).append(row)

    if seen_kinds != set(FRAME_KINDS):
        raise ValueError(
            f"corpus kind coverage differs from FRAME_KINDS: {sorted(seen_kinds)}"
        )
    if not {1, 2, 3, 4}.issubset(cardinalities):
        raise ValueError("corpus must cover cardinalities one through four")
    required_tags = {
        "single-frame",
        "multi-frame",
        "negation",
        "order-swap",
        "number-change",
        "polite-buried-constraint",
        "conflicting-constraints",
        "repeated-kind",
    }
    if missing := sorted(required_tags - tags):
        raise ValueError(f"required training phenomena are missing: {missing}")
    for group, members in groups.items():
        if len(members) < 2:
            raise ValueError(f"paraphrase group {group!r} is a singleton")
        target_hashes = {canonical_json_sha256(member["frames"]) for member in members}
        if len(target_hashes) != 1:
            raise ValueError(f"paraphrase group {group!r} has multiple frame sets")

    # Reuse the actual trainer's schema and exact-target checks, but provide the
    # independently collected exclusions so this call remains blind to v5.
    _reviewed_rows_from_text(
        corpus_bytes(rows).decode("utf-8"),
        held_out,
        forbidden,
    )


def _metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    kind_occurrences: Counter[str] = Counter()
    row_kind_presence: Counter[str] = Counter()
    cardinalities: Counter[str] = Counter()
    tags: Counter[str] = Counter()
    groups: Counter[str] = Counter()
    for row in rows:
        frames = list(row["frames"])
        kinds = [str(frame["kind"]) for frame in frames]
        kind_occurrences.update(kinds)
        row_kind_presence.update(set(kinds))
        cardinalities[str(len(frames))] += 1
        tags.update(str(tag) for tag in row["training_tags"])
        groups[str(row["paraphrase_group"])] += 1
    return {
        "row_count": len(rows),
        "paraphrase_group_count": len(groups),
        "contrastive_positive_pair_count": sum(
            count * (count - 1) // 2 for count in groups.values()
        ),
        "kind_occurrences": dict(sorted(kind_occurrences.items())),
        "row_kind_presence": dict(sorted(row_kind_presence.items())),
        "cardinality_rows": dict(sorted(cardinalities.items())),
        "training_tag_rows": dict(sorted(tags.items())),
    }


def build_artifacts(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
) -> tuple[bytes, bytes, dict[str, Any]]:
    v1_terms, v1_reference = _vocabulary(
        PINNED_V1_VOCABULARY, PINNED_V1_VOCABULARY_SHA256
    )
    v2_terms, v2_reference = _vocabulary(held_out_v2_path.resolve(), held_out_v2_sha256)
    forbidden, exclusions = _frozen_v1_v4_surfaces()
    rows = build_rows()
    validate_rows(
        rows,
        held_out_terms=v1_terms | v2_terms,
        forbidden_surfaces=forbidden,
    )
    encoded_corpus = corpus_bytes(rows)
    corpus_sha256 = hashlib.sha256(encoded_corpus).hexdigest()
    semantic_targets = [{"id": row["id"], "frames": row["frames"]} for row in rows]
    builder_reference = {
        "schema": "kev.artifact-ref.v1",
        "path": "scripts/build_reviewed_training_corpus_v5.py",
        "sha256": file_sha256(Path(__file__)),
        "size_bytes": Path(__file__).stat().st_size,
    }
    v1_reference["path"] = "evals/frozen/held-out-vocabulary-v1.txt"
    v2_reference["path"] = "evals/frozen/held-out-vocabulary-v2.txt"
    manifest = {
        "schema": "kev.reviewed-training-corpus-manifest.v1",
        "id": "kev-v5-reviewed-semantic-frame-corpus",
        "frozen": True,
        "created_at": REVIEWED_AT,
        "purpose": (
            "first bounded v5 experiment for frame-kind and cardinality proposal heads"
        ),
        "corpus": {
            "schema": "kev.artifact-ref.v1",
            "path": f"training/reviewed/{CORPUS_NAME}",
            "sha256": corpus_sha256,
            "size_bytes": len(encoded_corpus),
            "canonical_rows_sha256": canonical_json_sha256(rows),
            "semantic_targets_sha256": canonical_json_sha256(semantic_targets),
        },
        "builder": builder_reference,
        "metrics": _metrics(rows),
        "provenance": {
            "authorship": "REPOSITORY_AUTHORED_SYNTHETIC",
            "authored_by": "openai-codex-agent:reviewed-corpus-task",
            "reviewed_by": REVIEWER,
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "authorization": "user-authorized KEV v5 repository experiment",
            "permission": PERMISSION,
            "ordinary_chat_training_data": False,
            "third_party_text": False,
        },
        "separation": {
            "v5_promotion_suite_read_by_builder": False,
            "v5_promotion_suite_surface_used": False,
            "v5_promotion_suite_provenance_only": {
                "file_sha256": V5_SUITE_FILE_SHA256_PROVENANCE_ONLY,
                "canonical_sha256": V5_SUITE_CANONICAL_SHA256_PROVENANCE_ONLY,
                "content_read": False,
            },
            "v5_vocabulary_holdout_read_only": True,
            "v1_v4_exact_surface_exclusions": exclusions,
            "held_out_vocabularies": [v1_reference, v2_reference],
            "final_training_boundary": (
                "train() must still verify this corpus against every frozen suite at execution time"
            ),
        },
        "coverage": {
            "frame_kinds": list(FRAME_KINDS),
            "cardinalities": [1, 2, 3, 4],
            "required_phenomena": [
                "single-frame",
                "multi-frame",
                "negation",
                "order-swap",
                "number-change",
                "polite-buried-constraint",
                "conflicting-constraints",
                "repeated-kind",
            ],
        },
        "limitations": [
            "synthetic corpus; it does not establish natural-language breadth",
            "same-party agent review; it is not independent human or external review",
            "exact surface and vocabulary separation do not prove semantic independence",
            "training completion and lower loss cannot qualify a model",
        ],
        "mutation_policy": (
            "immutable after first write; create a new corpus and manifest version for any change"
        ),
    }
    encoded_manifest = (
        json.dumps(
            manifest,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    return encoded_corpus, encoded_manifest, manifest


def write_artifacts(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    corpus, manifest_bytes, manifest = build_artifacts(
        held_out_v2_path, held_out_v2_sha256
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    corpus_path = output_directory / CORPUS_NAME
    manifest_path = output_directory / MANIFEST_NAME
    if corpus_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "v5 reviewed corpus artifacts already exist; refusing to refresh them"
        )
    with corpus_path.open("xb") as handle:
        handle.write(corpus)
    with manifest_path.open("xb") as handle:
        handle.write(manifest_bytes)
    return manifest


def check_artifacts(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    expected_corpus, expected_manifest, manifest = build_artifacts(
        held_out_v2_path, held_out_v2_sha256
    )
    corpus_path = output_directory / CORPUS_NAME
    manifest_path = output_directory / MANIFEST_NAME
    if corpus_path.read_bytes() != expected_corpus:
        raise ValueError("frozen v5 reviewed corpus does not replay byte-for-byte")
    if manifest_path.read_bytes() != expected_manifest:
        raise ValueError("frozen v5 corpus manifest does not replay byte-for-byte")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the immutable reviewed v5 training corpus."
    )
    parser.add_argument("--held-out-v2", type=Path, default=PINNED_V2_VOCABULARY)
    parser.add_argument("--held-out-v2-sha256", default=PINNED_V2_VOCABULARY_SHA256)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    operation = check_artifacts if args.check else write_artifacts
    manifest = operation(
        args.held_out_v2,
        args.held_out_v2_sha256,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

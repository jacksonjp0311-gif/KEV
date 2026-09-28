"""Retain and replay development-only frame-binding probes from trusted sources.

Creation reads the original source from a pinned Git commit once. Replay uses
the retained, hash-checked source snapshots, so a clean clone needs no old Git
history or model download. Every head proposal is supplied by the developer;
these probes do not measure learned model capability or qualify a checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import types
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "experiments" / "composition-v7-development"
BASELINE_COMMIT = "1f5d9f322876d3e9ad7500b543521a64a3de5be1"
BASELINE_SHA256 = "3e8f583611effaab4d435dd70edb3f91d48384af682061370e98cbf57152884d"
AFTER_SHA256 = "8bfb6afcccd15e2ba668ab56999ce381932ca507eb1a4185f9caabf0bb9b2497"
TIMESTAMP = "2026-09-28T00:00:00Z"


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)


def _source_module(data: bytes, expected_sha256: str) -> types.ModuleType:
    if _sha256(data) != expected_sha256:
        raise ValueError(f"source snapshot hash mismatch: {expected_sha256}")
    name = f"kev_development_frames_{expected_sha256}"
    module = types.ModuleType(name)
    # Dataclass processing consults the registered module during execution.
    sys.modules[name] = module
    exec(compile(data, f"sha256:{expected_sha256}", "exec"), module.__dict__)
    return module


def _threshold(
    kind: str, subject: str, operator: str, value: int, unit: str
) -> dict[str, Any]:
    return {
        "kind": kind,
        "relation": "FORECAST" if kind == "PREDICTION" else "THRESHOLD",
        "slots": {
            "subject": subject,
            "operator": operator,
            "value": value,
            "unit": unit,
        },
    }


def _probes() -> list[dict[str, Any]]:
    goal = _threshold("GOAL", "buffer_delay", "LT", 17, "ms")
    constraint = {
        "kind": "CONSTRAINT",
        "relation": "ACTION_POLICY",
        "slots": {
            "action": "modify",
            "object": "archive",
            "polarity": False,
            "modality": "FORBIDDEN",
        },
    }
    probes: list[dict[str, Any]] = [
        {
            "id": "repeated-kind",
            "input": "buffer delay below 17 ms; buffer delay above 29 ms",
            "proposal": {"kinds": ["GOAL"], "cardinality": 2},
            "expected": [goal, _threshold("GOAL", "buffer_delay", "GT", 29, "ms")],
        },
        {
            "id": "kind-clause-order",
            "input": "cache pressure at 29 percent; buffer delay below 17 ms",
            "proposal": {"kinds": ["GOAL", "OBSERVATION"], "cardinality": 2},
            "expected": [
                {
                    "kind": "OBSERVATION",
                    "relation": "MEASUREMENT",
                    "slots": {"subject": "cache_pressure", "value": 29, "unit": "%"},
                },
                goal,
            ],
        },
        {
            "id": "ambiguity",
            "input": "buffer delay below 17 ms",
            "proposal": {
                "kinds": ["GOAL", "CONSTRAINT"],
                "cardinality": 2,
                "scores": {"GOAL": 0.999, "CONSTRAINT": 0.001},
            },
            "expected": [],
        },
        {
            "id": "subject-cue",
            "input": "We would like response time to be below 17 milliseconds; please ensure cache pressure is below 29 percent",
            "proposal": {"kinds": ["CONSTRAINT", "GOAL"], "cardinality": 2},
            "expected": [
                _threshold("GOAL", "latency", "LT", 17, "ms"),
                _threshold("CONSTRAINT", "cache_pressure", "LT", 29, "%"),
            ],
        },
        {
            "id": "narrow-kind-cue",
            "input": "We estimate buffer delay below 17 ms",
            "proposal": {"kinds": ["GOAL"], "cardinality": 1},
            "expected": [],
        },
        {
            "id": "narrow-modal",
            "input": "The policy should require buffer delay below 17 ms",
            "proposal": {"kinds": ["GOAL"], "cardinality": 1},
            "expected": [],
        },
        {
            "id": "compound-action-object",
            "input": "modify archive then delete backups",
            "proposal": {"kinds": ["GOAL"], "cardinality": 1},
            "expected": [],
        },
        {
            "id": "parser-preserved",
            "input": "Reduce buffer delay below 17 ms; do not modify archive",
            "proposal": {"kinds": ["PREDICTION", "OBSERVATION"], "cardinality": 6},
            "expected": [goal, constraint],
        },
    ]
    for probe in probes:
        if probe["id"] in {"narrow-kind-cue", "narrow-modal"}:
            text = probe["input"]
            probe["proposal"]["spans"] = [
                {"kind": "GOAL", "start": text.index("buffer"), "end": len(text)}
            ]
    return probes


def _measure(before: bytes, after: bytes) -> dict[str, Any]:
    modules = {
        "before": _source_module(before, BASELINE_SHA256),
        "after": _source_module(after, AFTER_SHA256),
    }
    results: list[dict[str, Any]] = []
    for probe in _probes():
        result = dict(probe)
        for label, module in modules.items():
            frames = module.extract_frames(
                probe["input"],
                source_type="development_probe",
                source_id=f"composition-v7-development/{probe['id']}",
                actor="developer-supplied-fixture",
                timestamp=TIMESTAMP,
                proposal_result=probe["proposal"],
            )
            semantics = [
                {
                    "kind": frame.kind,
                    "relation": frame.relation,
                    "slots": {name: slot.value for name, slot in frame.slots.items()},
                }
                for frame in frames
            ]
            result[label] = {
                "raw_frames": [frame.to_dict() for frame in frames],
                "semantics": semantics,
                "expectation_passed": semantics == probe["expected"],
            }
        results.append(result)
    return {
        "cases": results,
        "summary": {
            "total": len(results),
            "before_expectations_passed": sum(
                row["before"]["expectation_passed"] for row in results
            ),
            "after_expectations_passed": sum(
                row["after"]["expectation_passed"] for row in results
            ),
            "parser_control_raw_output_unchanged": results[-1]["before"]
            == results[-1]["after"],
        },
    }


def _report(before: bytes, after: bytes) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema": "kev.composition-development-probes.v1",
        "promotion_eligible": False,
        "claim_boundary": "Development-influenced mechanism probes with developer-supplied head proposals; not learned-model capability, fresh evaluation, or checkpoint qualification.",
        "training": "NONE",
        "checkpoint": None,
        "fixed_timestamp": TIMESTAMP,
        "sources": {
            "before": {
                "git_commit": BASELINE_COMMIT,
                "repository_path": "kev/frames.py",
                "snapshot": "before-frames.py.txt",
                "sha256": BASELINE_SHA256,
            },
            "after": {
                "repository_path": "kev/frames.py",
                "snapshot": "after-frames.py.txt",
                "sha256": AFTER_SHA256,
            },
            "replay_script": {
                "path": "scripts/measure_composition_binding_v7.py",
                "sha256": _sha256(Path(__file__).read_bytes()),
            },
        },
        **_measure(before, after),
    }
    report["integrity_sha256"] = _sha256(_canonical(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Replay retained source snapshots without Git history or writes.",
    )
    args = parser.parse_args()
    if args.check:
        before = (OUTPUT / "before-frames.py.txt").read_bytes()
        after = (OUTPUT / "after-frames.py.txt").read_bytes()
        expected_bytes = _json_bytes(_report(before, after))
        retained_bytes = (OUTPUT / "probes.json").read_bytes()
        if retained_bytes != expected_bytes:
            raise ValueError(
                "retained development evidence does not match exact replay"
            )
        report = json.loads(retained_bytes)
        mode = "REPLAY_VERIFIED"
    else:
        if OUTPUT.exists():
            raise FileExistsError(f"refusing to overwrite retained evidence: {OUTPUT}")
        before = subprocess.check_output(
            [
                "git",
                "-c",
                f"safe.directory={ROOT.as_posix()}",
                "-C",
                str(ROOT),
                "show",
                f"{BASELINE_COMMIT}:kev/frames.py",
            ]
        )
        after = (ROOT / "kev" / "frames.py").read_bytes()
        report = _report(before, after)
        if report["summary"]["after_expectations_passed"] != report["summary"]["total"]:
            raise ValueError(
                "changed runtime did not satisfy the declared development expectations"
            )
        OUTPUT.mkdir(parents=False, exist_ok=False)
        _write_exclusive(OUTPUT / "before-frames.py.txt", before)
        _write_exclusive(OUTPUT / "after-frames.py.txt", after)
        _write_exclusive(OUTPUT / "probes.json", _json_bytes(report))
        mode = "EVIDENCE_WRITTEN"
    print(
        json.dumps(
            {
                "status": mode,
                "promotion_eligible": False,
                "evidence": str((OUTPUT / "probes.json").relative_to(ROOT)),
                "file_sha256": _sha256((OUTPUT / "probes.json").read_bytes()),
                "integrity_sha256": report["integrity_sha256"],
                "summary": report["summary"],
                "execution_python_version": platform.python_version(),
            },
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

"""Freeze, run, and replay a NON-ACTIVATING clause-local research experiment.

Run from the repository root with ``python -m scripts.clause_local_experiment``.
No production registry is written. All outputs use exclusive creation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from kev.clause_proposals import ClauseLocalPredictor
from kev.evaluation import compare_evaluations, evaluate_suite, validate_frozen_suite
from kev.frames import extract_frames
from kev.model_runtime import CheckpointPredictor, predict_proposal, semantic_frame
from kev.sentence_training import train_frozen_encoder_challenger
from kev.uc51a3.alive import AliveStore

ROOT = Path(__file__).resolve().parents[1]
BOUNDARY = ROOT / "experiments/clause-local-v1-boundary"
OUTPUT = ROOT / "experiments/clause-local-v1-evidence"
PARENT = ROOT / "models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt"
ENCODER = ROOT / "models/local/encoders/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/encoder-manifest.json"
RESERVED = ["orchid", "cedar", "zircon", "quartz", "feldspar", "mica"]
KINDS = ["GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
        stream.write("\n")


def target(kind: str, subject: str, number: int) -> dict[str, Any]:
    slots: dict[str, Any] = {"subject": subject.replace(" ", "_"), "value": number, "unit": "ms"}
    if kind != "OBSERVATION":
        slots["operator"] = "EQ" if kind == "PREDICTION" else "LT"
    return {"kind": kind, "relation": {"GOAL": "THRESHOLD", "CONSTRAINT": "THRESHOLD", "OBSERVATION": "MEASUREMENT", "PREDICTION": "FORECAST"}[kind], "slots": slots}


def surface(kind: str, subject: str, number: int, variant: int = 0) -> str:
    templates = {
        "GOAL": ["We wish {s} below {n} ms", "We seek {s} below {n} milliseconds"],
        "CONSTRAINT": ["Ensure that {s} is below {n} ms", "Require that {s} is below {n} milliseconds"],
        "OBSERVATION": ["We see {s} is {n} ms", "We find that {s} is {n} milliseconds"],
        "PREDICTION": ["We estimate {s} is {n} ms", "We estimate that {s} is {n} milliseconds"],
    }
    return templates[kind][variant].format(s=subject, n=number)


def freeze() -> None:
    if BOUNDARY.exists():
        raise FileExistsError("boundary already exists; never refresh an exposed suite")
    now = datetime.now(timezone.utc).isoformat()
    lessons = []
    for kind in KINDS:
        for subject in ["processing delay", "render delay", "transfer delay", "decode delay"]:
            for number in [17, 29, 43, 61]:
                frame = target(kind, subject, number)
                typed = {**frame, "slots": {name: {"value": value, "type": {"subject": "entity", "value": "number", "operator": "operator", "unit": "unit"}[name]} for name, value in frame["slots"].items()}}
                group = f"local-{kind}-{subject.replace(' ', '-')}-{number}"
                for variant in range(2):
                    lessons.append({
                        "schema": "kev.reviewed-lesson.v1", "id": f"{group}-{variant}",
                        "text": surface(kind, subject, number, variant),
                        "frames": [typed], "frame_kinds": [kind], "frame_cardinality": 1,
                        "paraphrase_group": group, "status": "REVIEWED",
                        "reviewed_by": "openai-codex-agent:clause-local-v1",
                        "reviewed_at": now,
                        "permission": "Repository-authored synthetic text; user-authorized KEV research; no ordinary chat or third-party text. Same-party agent semantic review, not independent human review.",
                    })
    splits: dict[str, list[Any]] = {"fresh": [], "oov": [], "composition": []}
    localization: dict[str, list[list[str]]] = {}

    def add(split: str, clauses: list[str], frames: list[Any], labels: list[list[str]], audit: str):
        item_id = f"local-v1-{split}-{len(splits[split]):03d}"
        splits[split].append({"id": item_id, "input": "; ".join(clauses) + ".", "expected": frames, "comparison": "set_exact", "audit": audit})
        localization[item_id] = labels

    for subject in ["orchid delay", "cedar delay"]:
        for kind in KINDS:
            for variant in range(2):
                add("fresh", [surface(kind, subject, 73, variant)], [target(kind, subject, 73)], [[kind]], "single-frame")
    for noun in RESERVED[2:]:
        for kind in KINDS:
            add("oov", [surface(kind, noun + " delay", 89)], [target(kind, noun + " delay", 89)], [[kind]], "reserved-vocabulary")
    for index in range(8):
        kind = KINDS[index % 4]
        other = KINDS[(index + 1) % 4]
        subject, second = "orchid delay", "cedar delay"
        n = 97 + index
        a, b = surface(kind, subject, n), surface(other, second, n + 11, 1)
        fa, fb = target(kind, subject, n), target(other, second, n + 11)
        add("composition", [a, b], [fa, fb], [[kind], [other]], "multi-frame")
        add("composition", [b, a], [fa, fb], [[other], [kind]], "order-swap")
        add("composition", [a, "please " + surface("CONSTRAINT", second, n + 7)], [fa, target("CONSTRAINT", second, n + 7)], [[kind], ["CONSTRAINT"]], "polite-buried-constraint")
        add("composition", [surface("CONSTRAINT", subject, n), surface("CONSTRAINT", subject, n + 1)], [target("CONSTRAINT", subject, n), target("CONSTRAINT", subject, n + 1)], [["CONSTRAINT"], ["CONSTRAINT"]], "repeated-kind-number-change")
        add("composition", ["It is not true that " + a.lower(), b], [fb], [[], [other]], "negation")
        add("composition", [surface(kind, subject, n, 1), surface(other, second, n + 11)], [fa, fb], [[kind], [other]], "paraphrase-same-frames")
        conflicting = target("CONSTRAINT", subject, n + 1)
        conflicting["slots"]["operator"] = "GT"
        add("composition", [surface("CONSTRAINT", subject, n), f"Require that {subject} is above {n + 1} ms"], [target("CONSTRAINT", subject, n), conflicting], [["CONSTRAINT"], ["CONSTRAINT"]], "conflicting-constraints")
    old = ROOT / "evals/frozen/public-audit-v7-260.json"
    splits["retention"] = json.loads(old.read_text(encoding="utf-8"))["splits"]["retention"]
    suite = validate_frozen_suite({"schema": "kev.eval-suite.v1", "id": "clause-local-v1-research", "frozen": True, "splits": splits, "provenance": {
        "review": "SAME_PARTY_SYNTHETIC_NOT_INDEPENDENT", "created_at": now,
        "targets": "authored semantic judgments, not runtime outputs",
        "limitations": "Templates shared with training; subject tokens and numbers held out. Composition variants are correlated, not independent samples. Negation is a negative-control audit, not a trained zero-frame class.",
        "training_use": "FORBIDDEN", "activation": "NONE_RESEARCH_ONLY",
        "retention_source_sha256": digest(old),
    }})
    BOUNDARY.mkdir(parents=True)
    with (BOUNDARY / "reviewed-lessons.jsonl").open("x", encoding="utf-8", newline="\n") as stream:
        for row in lessons:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    write(BOUNDARY / "suite.json", suite)
    write(BOUNDARY / "localization-targets.json", localization)
    paths = [*sorted((ROOT / "kev").rglob("*.py")), Path(__file__), PARENT, ENCODER, old, ROOT / "models/registry.json", ROOT / "constraints-research.txt", *sorted(BOUNDARY.glob("*"))]
    write(BOUNDARY / "protocol.json", {
        "schema": "kev.clause-local-protocol.v1", "created_at": now,
        "activation": "NONE_RESEARCH_ONLY", "seed": 52103, "steps": 400,
        "reserved_vocabulary": RESERVED, "retention_epsilon": 0,
        "selection": "one preregistered seed, no retries or best-seed selection",
        "hypothesis": "clause-local learned kind proposals improve multi-frame extraction beyond same-weight global proposals and parser-only extraction",
        "support_requires": "strict fresh and composition gains over both parser-only and same-checkpoint global; no retention or OOV regression versus global; local kind/cardinality exact accuracy reported separately",
        "promotion": "Never activates. A supported hypothesis still requires versioned production-runtime integration, held-out calibration and the full promotion gate.",
        "hashes": {str(path.relative_to(ROOT)).replace('\\', '/'): digest(path) for path in paths},
    })
    print(json.dumps({"boundary_sha256": digest(BOUNDARY / "protocol.json"), "lessons": len(lessons), "splits": {key: len(value) for key, value in splits.items()}}))


def verify_protocol() -> dict[str, Any]:
    protocol = json.loads((BOUNDARY / "protocol.json").read_text(encoding="utf-8"))
    for name, expected in protocol["hashes"].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f"frozen input changed: {name}")
    return protocol


def parser_only(item):
    return {"predicted": [semantic_frame(frame) for frame in extract_frames(item["input"], timestamp="1970-01-01T00:00:00Z")], "score_status": "UNCALIBRATED"}


def reports(checkpoint: Path) -> dict[str, Any]:
    predictor = CheckpointPredictor(checkpoint, encoder_manifest=ENCODER)
    local = ClauseLocalPredictor(lambda text: predict_proposal(predictor.model, text), predictor.sha256)
    suite = BOUNDARY / "suite.json"
    result = {
        "parser": evaluate_suite(suite, predictor=parser_only),
        "global": evaluate_suite(suite, predictor=predictor, checkpoint_path=checkpoint),
        "local": evaluate_suite(suite, predictor=local, checkpoint_path=checkpoint),
    }
    targets = json.loads((BOUNDARY / "localization-targets.json").read_text(encoding="utf-8"))
    rows = []
    for trace in local.trace:
        if trace["item_id"] not in targets:
            continue
        expected = targets[trace["item_id"]]
        actual = [[p["kind"] for p in c["proposals"]] for c in trace["clauses"]]
        counts = [c["cardinality"] for c in trace["clauses"]]
        rows.append({"item_id": trace["item_id"], "expected": expected, "predicted": actual, "cardinalities": counts, "correct": len(expected) == len(actual) and all(Counter(a) == Counter(b) and n == len(a) for a, b, n in zip(expected, actual, counts))})
    result["localization"] = {"correct": sum(row["correct"] for row in rows), "total": len(rows), "records": rows}
    result["trace"] = local.trace
    return result


def run() -> None:
    protocol = verify_protocol()
    if OUTPUT.exists():
        raise FileExistsError("evidence already exists; preserve failures")
    # Fail closed before train(), including literal overlap with the new pack.
    suite = json.loads((BOUNDARY / "suite.json").read_text(encoding="utf-8"))
    forbidden = {item["input"].strip().casefold() for items in suite["splits"].values() for item in items}
    for line in (BOUNDARY / "reviewed-lessons.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row["text"].strip().casefold() in forbidden:
            raise ValueError("new evaluation surface in lessons")
    OUTPUT.mkdir(parents=True)
    ledger = AliveStore(OUTPUT / "research-ledger")
    ledger.append_event("RESEARCH_STARTED", {"protocol_sha256": digest(BOUNDARY / "protocol.json"), "activation": "NONE"})
    try:
        receipt = train_frozen_encoder_challenger(PARENT, ENCODER, OUTPUT / "candidate", steps=protocol["steps"], seed=protocol["seed"], extra_jsonl=BOUNDARY / "reviewed-lessons.jsonl", held_out_vocabulary=protocol["reserved_vocabulary"])
        verify_protocol()
        incumbent = reports(PARENT)
        candidate = reports(OUTPUT / "candidate/semantic-breadth.pt")
        write(OUTPUT / "incumbent-ablations.json", incumbent)
        write(OUTPUT / "candidate-ablations.json", candidate)
        support = all(candidate["local"]["splits"][split]["correct"] > candidate[mode]["splits"][split]["correct"] for mode in ["parser", "global"] for split in ["fresh", "composition"])
        support = support and all(candidate["local"]["splits"][split]["correct"] >= candidate["global"]["splits"][split]["correct"] for split in ["retention", "oov"])
        card = {
            "schema": "kev.clause-local-research-card.v1", "activation": "NONE",
            "hypothesis": "SUPPORTED" if support else "NOT_SUPPORTED",
            "production_qualification": "NOT_EVALUATED_RUNTIME_NOT_INTEGRATED",
            "protocol_sha256": digest(BOUNDARY / "protocol.json"),
            "checkpoint_sha256": receipt["challenger_sha256"], "parent_sha256": receipt["parent_sha256"],
            "diagnostic_gate_only": compare_evaluations(incumbent["global"], candidate["local"], retention_epsilon=0),
            "reports": {name: digest(OUTPUT / name) for name in ["incumbent-ablations.json", "candidate-ablations.json"]},
            "score_status": "UNCALIBRATED", "training_loss_is_promotion_feature": False,
        }
        write(OUTPUT / "research-card.json", card)
        ledger.append_event("RESEARCH_COMPLETED", {"card_sha256": digest(OUTPUT / "research-card.json"), "hypothesis": card["hypothesis"], "activation": "NONE"})
        verify_protocol()
        print(json.dumps(card))
    except Exception as error:
        ledger.append_event("RESEARCH_FAILED", {"error_type": type(error).__name__, "error": str(error), "activation": "NONE"})
        raise


def replay() -> None:
    verify_protocol()
    checkpoint = OUTPUT / "candidate/semantic-breadth.pt"
    card = json.loads((OUTPUT / "research-card.json").read_text(encoding="utf-8"))
    if digest(checkpoint) != card["checkpoint_sha256"]:
        raise ValueError("candidate checkpoint changed")
    for name, path in [("incumbent-ablations.json", PARENT), ("candidate-ablations.json", checkpoint)]:
        if digest(OUTPUT / name) != card["reports"][name]:
            raise ValueError("retained report changed")
        saved = json.loads((OUTPUT / name).read_text(encoding="utf-8"))
        actual = reports(path)
        # Absolute checkpoint/suite paths differ across checkouts. Every model
        # prediction, failure, slot, local proposal and metric must still match.
        for mode in ["parser", "global", "local"]:
            for field in ["splits", "records", "raw_failures", "prediction_evidence_sha256"]:
                if actual[mode][field] != saved[mode][field]:
                    raise ValueError(f"replay mismatch: {name}/{mode}/{field}")
        if actual["trace"] != saved["trace"] or actual["localization"] != saved["localization"]:
            raise ValueError("localization replay mismatch")
    write(OUTPUT / "replay-receipt.json", {"status": "PASS", "protocol_sha256": digest(BOUNDARY / "protocol.json"), "card_sha256": digest(OUTPUT / "research-card.json"), "scope": "fresh inference; all raw predictions, failures, frame provenance, clause logits-derived scores and localization exact", "activation": "NONE"})
    print("Exact research replay passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "replay"])
    {"freeze": freeze, "run": run, "replay": replay}[parser.parse_args().action]()

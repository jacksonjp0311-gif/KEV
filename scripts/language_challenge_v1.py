"""Freeze and measure language challenge v1. No training or activation."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re

from kev.clause_proposals import ClauseLocalPredictor
from kev.frames import FRAME_KINDS, _claim_ranges, extract_frames
from kev.model_runtime import CheckpointPredictor, predict_proposal, semantic_frame
from kev.uc51a3.alive import AliveStore
from scripts.clause_local_experiment import ENCODER, PARENT, ROOT, digest, write

BOUNDARY = ROOT / "experiments/language-challenge-v1-boundary"
OUTPUT = ROOT / "experiments/language-challenge-v1-evidence"
CANDIDATE = ROOT / "experiments/clause-local-v1-evidence/candidate/semantic-breadth.pt"
MODES = ("parser", "symbolic_all_kinds", "genesis_global", "trained_global", "trained_local")


def frame(kind, subject, number, operator="LT"):
    slots = {"subject": subject.replace(" ", "_"), "value": number, "unit": "ms"}
    if kind != "OBSERVATION":
        slots["operator"] = operator
    return {"kind": kind, "relation": "MEASUREMENT" if kind == "OBSERVATION" else "THRESHOLD", "slots": slots}


def authored_items():
    items = []

    def add(family, text, expected, rationale, pair=None):
        items.append({"id": f"language-v1-{len(items):03d}", "family": family, "input": text,
                      "expected": expected, "rationale": rationale, "pair_id": pair})

    for text, kind, subject, n, op in [
        ("Our desired outcome is for checkout latency to fall below 47 ms.", "GOAL", "checkout latency", 47, "LT"),
        ("For cache delay, 53 ms is an upper bound that must not be exceeded.", "CONSTRAINT", "cache delay", 53, "LTE"),
        ("The latest reading puts dispatch latency at 71 ms.", "OBSERVATION", "dispatch latency", 71, "LT"),
        ("A successful run would bring render delay down to less than 39 ms.", "GOAL", "render delay", 39, "LT"),
    ]:
        add("unfamiliar_phrasing", text, [frame(kind, subject, n, op)], "Explicit entity and number; unfamiliar wording preserves the stated kind and comparator.")
    for subject, n, limit in [("checkout latency", 71, 47), ("cache delay", 83, 53)]:
        observation = frame("OBSERVATION", subject, n)
        add("resolved_reference", f"Observed {subject} is {n} ms. Keep it below {limit} ms.",
            [observation, frame("GOAL", subject, limit)], "Only one named measured entity; 'it' resolves to that entity.")
        add("resolved_reference", f"Observed {subject} is {n} ms. That same {subject} must remain below {limit} ms.",
            [observation, frame("CONSTRAINT", subject, limit)], "Explicit same-entity reference; the requirement is not a new entity.")
    for noun, n in [("checkout latency", 71), ("dispatch latency", 83)]:
        add("ambiguous_reference", f"Observed {noun} is {n} ms; observed cache delay is 59 ms. One of these must stay below 43 ms, but I have not specified which.",
            [frame("OBSERVATION", noun, n), frame("OBSERVATION", "cache delay", 59)],
            "Preserve both observations. Do not bind the unassigned threshold to either entity; clarification is required.")
        add("ambiguous_reference", "Keep it below 43 ms. I have not identified what it refers to.", [],
            "No grounded referent. Abstain from a concrete threshold frame and request clarification; this benchmark scores abstention, not a clarification UI.")
    # Distinct final surfaces for the two no-context controls; no target changes.
    items[-1]["input"] = "It must remain below 43 ms. The subject has not been identified."
    for text, expected in [
        ("Please aim for checkout latency below 47 ms; cache delay must stay at most 53 ms.", [frame("GOAL", "checkout latency", 47), frame("CONSTRAINT", "cache delay", 53, "LTE")]),
        ("Checkout latency must be below 47 ms; checkout latency must be above 61 ms.", [frame("CONSTRAINT", "checkout latency", 47), frame("CONSTRAINT", "checkout latency", 61, "GT")]),
        ("Observed render delay is 71 ms; keep render delay below 39 ms; dispatch latency must be at least 53 ms.", [frame("OBSERVATION", "render delay", 71), frame("GOAL", "render delay", 39), frame("CONSTRAINT", "dispatch latency", 53, "GTE")]),
        ("I would appreciate checkout latency below 47 ms; one nonnegotiable requirement is cache delay below 53 ms.", [frame("GOAL", "checkout latency", 47), frame("CONSTRAINT", "cache delay", 53)]),
    ]:
        add("interacting_constraints", text, expected, "Keep separate kinds, subjects and bounds. Preserve conflicting requirements rather than resolving them by deletion.")
    for text, expected, pair in [
        ("Keep checkout latency below 47 ms; keep cache delay below 53 ms.", [frame("GOAL", "checkout latency", 47), frame("GOAL", "cache delay", 53)], "order"),
        ("Keep cache delay below 53 ms; keep checkout latency below 47 ms.", [frame("GOAL", "checkout latency", 47), frame("GOAL", "cache delay", 53)], "order"),
        ("Observed dispatch latency is 71 ms. It is not true that dispatch latency is 83 ms.", [frame("OBSERVATION", "dispatch latency", 71)], "negation-number"),
        ("Observed dispatch latency is 83 ms. It is not true that dispatch latency is 71 ms.", [frame("OBSERVATION", "dispatch latency", 83)], "negation-number"),
    ]:
        add("minimal_pairs", text, expected, "Order does not change the frame multiset; swapping the asserted and negated numbers changes the sole observation.", pair)
    for kind, subject, n, text in [
        ("GOAL", "checkout latency", 47, "Keep checkout latency below 47 ms."),
        ("CONSTRAINT", "cache delay", 53, "Cache delay must be below 53 ms."),
        ("OBSERVATION", "dispatch latency", 71, "Observed dispatch latency is 71 ms."),
        ("GOAL", "render delay", 39, "Keep render delay below 39 ms."),
    ]:
        add("explicit_controls", text, [frame(kind, subject, n)], "Ordinary explicit entity/value control; report separately from challenge items.")
    return items


def validate_items(items):
    ids, surfaces = set(), set()
    for item in items:
        text = item["input"]
        if item["id"] in ids or text.casefold() in surfaces:
            raise ValueError("duplicate challenge id or surface")
        ids.add(item["id"])
        surfaces.add(text.casefold())
        support = []
        for expected in item["expected"]:
            slots = expected["slots"]
            if slots["subject"].replace("_", " ") not in text.casefold():
                raise ValueError("target entity has no literal source support")
            if not re.search(rf"\b{re.escape(str(slots['value']))}\b", text):
                raise ValueError("target number has no literal source support")
            subject = slots["subject"].replace("_", " ")
            start = text.casefold().index(subject)
            match = re.search(rf"\b{re.escape(str(slots['value']))}\b", text)
            support.append({"subject": [start, start + len(subject)], "number": list(match.span())})
        item["target_support_spans"] = support
    return items


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def score(expected, predicted):
    wanted, actual = Counter(map(canonical, expected)), Counter(map(canonical, predicted))
    return {"correct": wanted == actual,
            "missing": [json.loads(v) for v in (wanted - actual).elements()],
            "unsupported": [json.loads(v) for v in (actual - wanted).elements()]}


def freeze():
    if BOUNDARY.exists():
        raise FileExistsError("boundary already frozen")
    items = validate_items(authored_items())
    training = sorted((ROOT / "training/reviewed").glob("*.jsonl")) + [ROOT / "experiments/clause-local-v1-boundary/reviewed-lessons.jsonl"]
    seen = {json.loads(line)["text"].strip().casefold().rstrip(".") for path in training for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}
    if any(item["input"].strip().casefold().rstrip(".") in seen for item in items):
        raise ValueError("new item overlaps retained training text")
    historical = ROOT / "evals/frozen/public-audit-v7-260.json"
    retention = json.loads(historical.read_text(encoding="utf-8"))["splits"]["retention"]
    items += [{"id": row["id"], "family": "retention", "input": row["input"], "expected": row["expected"]} for row in retention]
    write(BOUNDARY / "suite.json", {"schema": "kev.language-challenge.v1", "frozen": True, "items": items,
        "review": "SAME_PARTY_SYNTHETIC_SEMANTIC_REVIEW_NOT_INDEPENDENT", "training_use": "FORBIDDEN",
        "limitations": "24 authored diagnostic cases plus 40 retained controls; correlated pairs, not a representative population or an OOV promotion split. Existing schemas only. Ambiguous references require abstention; no automatic clarification capability is claimed."})
    paths = [Path(__file__), *sorted((ROOT / "kev").rglob("*.py")), PARENT, CANDIDATE, ENCODER, historical, ROOT / "models/registry.json", *training, BOUNDARY / "suite.json", ROOT / "experiments/clause-local-v1-boundary/suite.json"]
    write(BOUNDARY / "protocol.json", {"schema": "kev.language-challenge-protocol.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "training": "NONE", "activation": "NONE", "modes": MODES,
        "measurement": "Exact frame multiset; missing and unsupported frame counts; per-family counts. Empty output earns no credit on positive targets. Ambiguous targets permit only grounded explicit frames. No promotion decision.",
        "interpretation": "Any claim of learned advantage requires better challenge exact match than symbolic_all_kinds without higher unsupported-frame count or lower retention. A tie is no advantage. Outcomes are diagnostic only.",
        "hashes": {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}})
    print(json.dumps({"protocol_sha256": digest(BOUNDARY / "protocol.json"), "cases": len(items)}))


def verify_inputs(inference=False):
    protocol = json.loads((BOUNDARY / "protocol.json").read_text(encoding="utf-8"))
    for relative, sha in protocol["hashes"].items():
        if ROOT / relative == ENCODER and not inference:
            continue
        if digest(ROOT / relative) != sha:
            raise ValueError(f"frozen input changed: {relative}")
    return protocol


def measure():
    verify_inputs(inference=True)
    items = json.loads((BOUNDARY / "suite.json").read_text(encoding="utf-8"))["items"]
    genesis = CheckpointPredictor(PARENT)
    trained = CheckpointPredictor(CANDIDATE, encoder_manifest=ENCODER)
    local = ClauseLocalPredictor(lambda text: predict_proposal(trained.model, text), trained.sha256)
    models = {"genesis_global": genesis, "trained_global": trained, "trained_local": local}
    reports = {}
    for mode in MODES:
        records = []
        for item in items:
            safe_input = {"id": item["id"], "input": item["input"]}
            if mode in models:
                output = models[mode](safe_input)
            else:
                proposal = None if mode == "parser" else {"kinds": list(FRAME_KINDS), "cardinality": len(_claim_ranges(item["input"]))}
                frames = extract_frames(item["input"], source_type="EVALUATION", source_id=item["id"], proposal_result=proposal, timestamp="1970-01-01T00:00:00Z")
                output = {"predicted": [semantic_frame(f) for f in frames], "frames": [f.to_dict() for f in frames]}
            records.append({**item, "output": output, **score(item["expected"], output["predicted"])})
        families = {}
        for family in sorted({r["family"] for r in records}):
            rows = [r for r in records if r["family"] == family]
            families[family] = {"correct": sum(r["correct"] for r in rows), "total": len(rows), "missing_frames": sum(len(r["missing"]) for r in rows), "unsupported_frames": sum(len(r["unsupported"]) for r in rows)}
        reports[mode] = {"families": families, "records": records, "raw_failures": [r for r in records if not r["correct"]], "score_status": "UNCALIBRATED", "activation": "NONE"}
        if mode == "trained_local":
            reports[mode]["trace"] = local.trace
    return reports


def run():
    verify_inputs(inference=True)
    if OUTPUT.exists():
        raise FileExistsError("preserve existing evidence")
    store = AliveStore(OUTPUT / "ledger")
    store.append_event("RESEARCH_EVALUATION_STARTED", {"protocol_sha256": digest(BOUNDARY / "protocol.json"), "training": "NONE", "activation": "NONE"})
    try:
        reports = measure()
        for mode, report in reports.items():
            write(OUTPUT / f"{mode}.json", report)
        summary = {"schema": "kev.language-challenge-summary.v1", "activation": "NONE", "training": "NONE", "families": {m: r["families"] for m, r in reports.items()}}
        write(OUTPUT / "summary.json", summary)
        verify_inputs(inference=True)
        store.append_event("RESEARCH_EVALUATION_COMPLETED", {"summary_sha256": digest(OUTPUT / "summary.json"), "activation": "NONE"})
        write(OUTPUT / "inventory.json", {"protocol_sha256": digest(BOUNDARY / "protocol.json"), "files": {p.relative_to(OUTPUT).as_posix(): digest(p) for p in sorted(OUTPUT.rglob("*")) if p.is_file() and p.name != ".state.lock"}})
        print(json.dumps(summary))
    except Exception as error:
        store.append_event("RESEARCH_EVALUATION_FAILED", {"type": type(error).__name__, "message": str(error), "activation": "NONE"})
        raise


def verify(inference=False):
    verify_inputs(inference=inference)
    inventory = json.loads((OUTPUT / "inventory.json").read_text(encoding="utf-8"))
    if digest(BOUNDARY / "protocol.json") != inventory["protocol_sha256"]:
        raise ValueError("protocol identity mismatch")
    for relative, sha in inventory["files"].items():
        if digest(OUTPUT / relative) != sha:
            raise ValueError(f"evidence changed: {relative}")
    ledger = AliveStore._verify_ledger_bytes((OUTPUT / "ledger/ledger.jsonl").read_bytes())
    if not ledger["valid"]:
        raise ValueError("ledger invalid")
    anchor = json.loads((OUTPUT / "ledger/ledger-head.json").read_text(encoding="utf-8"))
    if any(anchor[key] != ledger[key] for key in ["head", "events"]) or anchor["ledger_size"] != (OUTPUT / "ledger/ledger.jsonl").stat().st_size:
        raise ValueError("ledger anchor mismatch")
    if inference:
        for mode, report in measure().items():
            if report != json.loads((OUTPUT / f"{mode}.json").read_text(encoding="utf-8")):
                raise ValueError(f"fresh inference differs: {mode}")
    return {"status": "PASS", "inference_replayed": inference, "protocol_sha256": inventory["protocol_sha256"], "inventory_sha256": digest(OUTPUT / "inventory.json"), "activation": "NONE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "run", "verify", "replay"])
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.action in {"freeze", "run"}:
        {"freeze": freeze, "run": run}[args.action]()
    else:
        result = verify(inference=args.action == "replay")
        if args.receipt:
            write(args.receipt, result)
        print(json.dumps(result))

"""Read-only evidence verification, with optional fresh local-model inference.

The original frozen driver preserves its first replay receipt exclusively.
This verifier permits repeated replay without overwriting that receipt.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.clause_local_experiment import BOUNDARY, ENCODER, OUTPUT, PARENT, ROOT, digest, reports, write
from kev.uc51a3.alive import AliveStore

PROTOCOL_SHA256 = "f776c41a65bba23beeff41c027049f54965373e5f3177ca9073c48d49cf0d41f"


def verify(*, inference: bool = False) -> dict:
    if digest(BOUNDARY / "protocol.json") != PROTOCOL_SHA256:
        raise ValueError("protocol changed")
    protocol = json.loads((BOUNDARY / "protocol.json").read_text(encoding="utf-8"))
    for relative, expected in protocol["hashes"].items():
        path = ROOT / relative
        if path == ENCODER and not inference:
            # CI has no encoder weights. This is artifact validation, not inference.
            continue
        if digest(path) != expected:
            raise ValueError(f"frozen input changed: {relative}")
    inventory = json.loads((OUTPUT / "inventory.json").read_text(encoding="utf-8"))
    for relative, expected in inventory["files"].items():
        if digest(OUTPUT / relative) != expected:
            raise ValueError(f"evidence changed: {relative}")
    ledger_path = OUTPUT / "research-ledger/ledger.jsonl"
    ledger = AliveStore._verify_ledger_bytes(ledger_path.read_bytes())
    anchor = json.loads((OUTPUT / "research-ledger/ledger-head.json").read_text())
    if not ledger["valid"] or any(ledger[key] != anchor[key] for key in ["head", "events"]) or anchor["ledger_size"] != ledger_path.stat().st_size:
        raise ValueError("invalid research ledger or anchor")
    card = json.loads((OUTPUT / "research-card.json").read_text(encoding="utf-8"))
    candidate_path = OUTPUT / "candidate/semantic-breadth.pt"
    if digest(candidate_path) != card["checkpoint_sha256"] or digest(PARENT) != card["parent_sha256"]:
        raise ValueError("checkpoint identity mismatch")
    if card["activation"] != "NONE":
        raise ValueError("research cannot activate")
    if inference:
        for name, checkpoint in [("incumbent-ablations.json", PARENT), ("candidate-ablations.json", candidate_path)]:
            saved = json.loads((OUTPUT / name).read_text(encoding="utf-8"))
            actual = reports(checkpoint)
            for mode in ["parser", "global", "local"]:
                for field in ["splits", "records", "raw_failures", "prediction_evidence_sha256"]:
                    if actual[mode][field] != saved[mode][field]:
                        raise ValueError(f"replay mismatch: {name}/{mode}/{field}")
            for field in ["trace", "localization"]:
                if actual[field] != saved[field]:
                    raise ValueError(f"replay mismatch: {name}/{field}")
    return {"status": "PASS", "inference_replayed": inference, "protocol_sha256": PROTOCOL_SHA256, "ledger": ledger, "activation": "NONE"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--seal", action="store_true", help="Create initial inventory exclusively; never refresh it")
    args = parser.parse_args()
    if args.seal:
        write(OUTPUT / "inventory.json", {
            "schema": "kev.research-inventory.v1",
            "files": {str(path.relative_to(OUTPUT)).replace('\\', '/'): digest(path) for path in sorted(OUTPUT.rglob("*")) if path.is_file() and path.name != ".state.lock"},
            "verifier_sha256": digest(Path(__file__)),
        })
    result = verify(inference=args.inference)
    if args.receipt:
        write(args.receipt, result)
    print(json.dumps(result))

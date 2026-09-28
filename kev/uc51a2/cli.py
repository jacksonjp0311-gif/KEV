from __future__ import annotations
import argparse
import json
import hashlib
from pathlib import Path
from .semantic_breadth import load, predict, train, INTENTS

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models/v051a2/semantic-breadth.pt"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["status", "answer", "train"])
    ap.add_argument("--message", default="")
    ap.add_argument("--output")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seed", type=int, default=51401)
    ap.add_argument("--parent", default=str(MODEL))
    ap.add_argument("--lessons")
    a = ap.parse_args()
    if a.mode == "status":
        print(
            json.dumps(
                {
                    "version": "0.51.0-alpha.2",
                    "model": str(MODEL),
                    "model_present": MODEL.is_file(),
                    "sha256": sha(MODEL) if MODEL.is_file() else None,
                    "intents": list(INTENTS),
                    "activation": "NONE",
                    "authority": "NONE",
                    "historical_score_status": "NOT_REPLAYABLE_FROM_PUBLIC_V0.51_ARTIFACTS",
                    "claim_boundary": "legacy research semantic classifier; use kev doctor and MODEL_WEIGHTS.md for current evidence",
                },
                indent=2,
            )
        )
    elif a.mode == "answer":
        if not a.message.strip():
            raise SystemExit("--message required")
        r = predict(load(MODEL), a.message)
        r.pop("embedding", None)
        r.update({"authority": "NONE", "activation": "NONE", "semantic_only": True})
        print(json.dumps(r, indent=2))
    else:
        if not a.output:
            raise SystemExit("--output required")
        print(
            json.dumps(train(a.parent, a.output, a.steps, a.seed, a.lessons), indent=2)
        )

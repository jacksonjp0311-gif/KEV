from __future__ import annotations

import argparse
import json
from pathlib import Path

from kev.evolution import DEFAULT_HELD_OUT
from kev.sentence_training import train_frozen_encoder_challenger
from kev.uc51a2.semantic_breadth import INTENTS, train as train_semantic

from .alive import AliveRuntime, AliveStore, VERSION


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=(
            "status",
            "chat",
            "new-session",
            "teach",
            "review",
            "export-lessons",
            "verify-ledger",
            "train-candidate",
        ),
    )
    parser.add_argument("--state-dir")
    parser.add_argument("--message", default="")
    parser.add_argument("--session")
    parser.add_argument("--intent")
    parser.add_argument("--text")
    parser.add_argument("--lesson-id")
    parser.add_argument("--reviewed-by")
    parser.add_argument("--permission")
    parser.add_argument("--output")
    parser.add_argument("--parent")
    parser.add_argument("--encoder-manifest")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=52021)
    args = parser.parse_args()
    store = AliveStore(args.state_dir)

    if args.mode == "status":
        state = store.read()
        output = {
            "version": VERSION,
            "state_dir": str(store.dir),
            "active_session": state["active_session"],
            "facts": len(state["facts"]),
            "typed_memory": len(state["typed_memory"]),
            "reviewed_lessons": sum(
                item.get("status") == "REVIEWED" for item in state["reviewed_lessons"]
            ),
            "draft_lessons": sum(
                item.get("status") == "DRAFT" for item in state["reviewed_lessons"]
            ),
            "semantic_model": state.get("semantic_model"),
            "semantic_model_sha256": state.get("semantic_model_sha256"),
            "ledger": store.verify_ledger(),
            "authority": "NONE",
        }
    elif args.mode == "new-session":
        output = {"session_id": store.new_session()}
    elif args.mode == "chat":
        output = AliveRuntime(args.state_dir).chat(args.message, args.session)
    elif args.mode == "teach":
        if args.intent not in INTENTS:
            raise SystemExit("valid --intent required: " + ",".join(INTENTS))
        output = store.add_lesson(args.intent, args.text or "", source="cli")
    elif args.mode == "review":
        output = store.review_lesson(
            args.lesson_id or "", args.reviewed_by or "", args.permission or ""
        )
    elif args.mode == "export-lessons":
        if not args.output:
            raise SystemExit("--output required")
        output = store.export_lessons(args.output)
    elif args.mode == "verify-ledger":
        output = store.verify_ledger()
    else:
        if not args.output:
            raise SystemExit("--output required")
        state = store.read()
        parent = Path(args.parent or state.get("semantic_model") or "")
        if not parent.is_file():
            raise SystemExit("valid --parent or active semantic model required")
        lesson_path = store.dir / "reviewed-lessons.export.jsonl"
        store.export_lessons(lesson_path)
        held_out = [
            line.strip()
            for line in DEFAULT_HELD_OUT.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if args.encoder_manifest:
            output = train_frozen_encoder_challenger(
                parent,
                args.encoder_manifest,
                args.output,
                steps=args.steps,
                seed=args.seed,
                extra_jsonl=lesson_path,
                held_out_vocabulary=held_out,
            )
        else:
            output = train_semantic(
                parent,
                args.output,
                args.steps,
                args.seed,
                lesson_path,
                held_out_vocabulary=held_out,
            )
    print(json.dumps(output, indent=2, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()

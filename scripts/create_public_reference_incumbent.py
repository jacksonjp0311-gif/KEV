"""Create the immutable, untrained v0.52 public reference incumbent once.

This is a genesis artifact, not a reconstruction of the unavailable v0.51
research checkpoint and not evidence of capability improvement.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch

from kev.uc51a2.semantic_breadth import INTENTS, MAX_FRAMES, _new_model, sha256_file


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "models" / "public"
REGISTRY = ROOT / "models" / "registry.json"


def main() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)
    manifest_path = PUBLIC / "incumbent-manifest.json"
    if manifest_path.exists() or REGISTRY.exists():
        raise SystemExit(
            "public reference artifacts already exist; refusing to replace them"
        )
    model = _new_model(52001)
    checkpoint = {
        "schema": "kev.semantic-frames.v052",
        "state_dict": model.state_dict(),
        "parameters": model.parameter_count,
        "initialization_seed": 52001,
        "intents": INTENTS,
        "max_frames": MAX_FRAMES,
        "parent_sha256": None,
        "lessons_sha256": None,
        "steps": 0,
        "objective": None,
        "temperature": None,
        "score_status": "UNCALIBRATED",
        "artifact_status": "INCUMBENT",
        "provenance": "deterministic untrained public genesis reference",
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="genesis-", suffix=".pt", dir=PUBLIC
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(checkpoint, handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary = Path(temporary_name)
        digest = sha256_file(temporary)
        final = PUBLIC / f"semantic-breadth-genesis-sha256-{digest}.pt"
        if final.exists():
            raise FileExistsError(final)
        os.replace(temporary, final)
    finally:
        if Path(temporary_name).exists():
            Path(temporary_name).unlink()

    relative = final.relative_to(ROOT).as_posix()
    manifest = {
        "schema": "kev.model-manifest.v1",
        "name": "v0.52 public genesis reference",
        "path": relative,
        "sha256": digest,
        "parent_sha256": None,
        "artifact_status": "INCUMBENT",
        "score_status": "UNCALIBRATED",
        "parameters": model.parameter_count,
        "training_steps": 0,
        "training_data": None,
        "claim_boundary": "untrained bootstrap artifact; not a capability claim",
    }
    registry = {
        "schema": "kev.model-registry.v1",
        "active": manifest,
        "history": [],
        "activation_receipt": "GENESIS_PUBLIC_REFERENCE",
    }
    with manifest_path.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY.open("x", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

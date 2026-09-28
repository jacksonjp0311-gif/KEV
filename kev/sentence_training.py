"""Reviewed-only training for an external, frozen local sentence encoder.

The encoder never enters the challenger checkpoint and never receives a
gradient.  Its manifest and exact files are verified before one embedding
cache is computed.  Only the projection, frame-kind head, and cardinality head
are optimized.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.encoders import (
    FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
    FrozenSentenceEncoderProposal,
    load_supported_local_encoder,
    verify_local_encoder_manifest,
)
from kev.uc51a2 import semantic_breadth as semantic_training
from kev.uc51a2.semantic_breadth import (
    CARDINALITY_WEIGHT,
    CONTRASTIVE_WEIGHT,
    INTENTS,
    MAX_FRAMES,
    PINNED_HELD_OUT_VOCABULARY,
    PINNED_HELD_OUT_VOCABULARY_FILE_SHA256,
    REPOSITORY_ROOT,
    _canonical_json_sha256,
    _frozen_evaluation_exclusions,
    _paraphrase_group_manifest,
    _reviewed_rows_from_text,
    _sha256_bytes,
    _targets,
    required_held_out_vocabulary,
    supervised_contrastive_loss,
)


TRAINING_SOURCE_PATHS = (
    ("FROZEN_ENCODER_TRAINER", Path(__file__).resolve()),
    ("ENCODER_RUNTIME", (Path(__file__).parent / "encoders.py").resolve()),
    ("SHARED_SEMANTIC_TRAINING", Path(semantic_training.__file__).resolve()),
)
_LOADED_TRAINING_SOURCE_HASHES = {
    role: hashlib.sha256(path.read_bytes()).hexdigest()
    for role, path in TRAINING_SOURCE_PATHS
}


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _training_source_manifest() -> list[dict[str, str]]:
    """Hash every Python source that defines this trainer's loss and inputs."""

    return [
        {
            "role": role,
            "path": _portable_path(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for role, path in TRAINING_SOURCE_PATHS
    ]


def _verified_training_source_manifest(
    expected: list[dict[str, str]] | None = None,
) -> list[dict[str, str]]:
    """Reject source swaps after import or during one training invocation."""

    current = _training_source_manifest()
    for entry in current:
        if entry["sha256"] != _LOADED_TRAINING_SOURCE_HASHES.get(entry["role"]):
            raise RuntimeError(
                f"training source {entry['role']} changed after module import"
            )
    if expected is not None and current != expected:
        raise RuntimeError(
            "training source manifest changed during challenger training"
        )
    return current


def _copy_parent_heads(
    model: FrozenSentenceEncoderProposal,
    parent: Mapping[str, Any],
    encoder_manifest_sha256: str,
) -> dict[str, Any]:
    """Copy every shape-compatible typed head from the incumbent lineage."""

    schema = str(parent.get("schema", "UNKNOWN"))
    if schema not in {"kev.semantic-frames.v052", FROZEN_SENTENCE_CHECKPOINT_SCHEMA}:
        raise ValueError(f"unsupported parent checkpoint schema: {schema}")
    parent_intents = parent.get("intents")
    if (
        not isinstance(parent_intents, (list, tuple))
        or tuple(parent_intents) != INTENTS
    ):
        raise ValueError("parent checkpoint intent order is incompatible")
    if parent.get("max_frames") != MAX_FRAMES:
        raise ValueError("parent checkpoint maximum frame cardinality is incompatible")
    copied: list[str] = []
    with torch.no_grad():
        if schema == "kev.semantic-frames.v052":
            state = parent.get("state_dict")
            if not isinstance(state, Mapping):
                raise ValueError("semantic parent checkpoint lacks state_dict")
            mappings = (
                (model.frame_head, "head"),
                (model.cardinality_head, "cardinality_head"),
            )
            for destination, prefix in mappings:
                weight = state.get(f"{prefix}.weight")
                bias = state.get(f"{prefix}.bias")
                if not isinstance(weight, torch.Tensor) or not isinstance(
                    bias, torch.Tensor
                ):
                    raise ValueError(f"semantic parent lacks {prefix} parameters")
                if (
                    weight.shape != destination.weight.shape
                    or bias.shape != destination.bias.shape
                ):
                    raise ValueError(f"semantic parent {prefix} shape is incompatible")
                destination.weight.copy_(weight)
                destination.bias.copy_(bias)
                copied.append(prefix)
        elif schema == FROZEN_SENTENCE_CHECKPOINT_SCHEMA:
            state = parent.get("proposal_state_dict")
            if not isinstance(state, Mapping):
                raise ValueError("frozen sentence parent lacks proposal_state_dict")
            reference = parent.get("encoder_manifest")
            if not isinstance(reference, Mapping):
                raise ValueError(
                    "frozen sentence parent lacks encoder manifest binding"
                )
            for destination_name, destination in (
                ("frame_head", model.frame_head),
                ("cardinality_head", model.cardinality_head),
            ):
                destination.load_state_dict(
                    {
                        "weight": state[f"{destination_name}.weight"],
                        "bias": state[f"{destination_name}.bias"],
                    },
                    strict=True,
                )
                copied.append(destination_name)
            if reference.get("sha256") == encoder_manifest_sha256:
                model.projection.load_state_dict(
                    {
                        "weight": state["projection.weight"],
                        "bias": state["projection.bias"],
                    },
                    strict=True,
                )
                copied.append("projection")
    return {
        "from_schema": schema,
        "to_schema": FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
        "copied_components": copied,
        "new_projection_initialization": (
            "COPIED_FROM_SAME_ENCODER_PARENT"
            if "projection" in copied
            else "DETERMINISTIC_XAVIER_UNIFORM"
        ),
    }


def _reset_trainable_layers(model: FrozenSentenceEncoderProposal, seed: int) -> None:
    """Initialize trainable layers independently of encoder/library construction."""

    with torch.random.fork_rng():
        torch.manual_seed(seed)
        for layer in (model.projection, model.frame_head, model.cardinality_head):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)


def _precompute_embeddings(
    model: FrozenSentenceEncoderProposal,
    texts: list[str],
    *,
    batch_size: int,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("encoder_batch_size must be positive")
    chunks: list[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        chunks.append(model.encode_texts(texts[start : start + batch_size]).cpu())
    result = torch.cat(chunks, dim=0).contiguous()
    if result.shape != (len(texts), model.encoder_dim):
        raise RuntimeError("precomputed encoder feature shape is inconsistent")
    if not torch.isfinite(result).all():
        raise RuntimeError("precomputed encoder features contain non-finite values")
    return result


def train_frozen_encoder_challenger(
    parent_path: str | Path,
    encoder_manifest: str | Path,
    out_dir: str | Path,
    *,
    steps: int = 800,
    seed: int = 52022,
    extra_jsonl: str | Path | None = None,
    held_out_vocabulary: Iterable[str] = (),
    encoder_batch_size: int = 32,
) -> dict[str, Any]:
    """Write one immutable frozen-encoder challenger and its evidence receipt."""

    if not extra_jsonl:
        raise RuntimeError("reviewed JSONL lessons are required")
    if steps <= 0:
        raise ValueError("steps must be positive")
    training_source_manifest = _verified_training_source_manifest()
    training_source_manifest_sha256 = _canonical_json_sha256(training_source_manifest)
    source_hashes = {
        entry["role"]: entry["sha256"] for entry in training_source_manifest
    }

    parent_file = Path(parent_path).resolve()
    manifest_file = Path(encoder_manifest).resolve()
    lesson_file = Path(extra_jsonl).resolve()
    output_directory = Path(out_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "semantic-breadth.pt"
    receipt_path = output_directory / "training-receipt.json"
    embeddings_path = output_directory / "frozen-embeddings.pt"
    if any(path.exists() for path in (checkpoint_path, receipt_path, embeddings_path)):
        raise FileExistsError("challenger output is immutable and already exists")

    manifest = verify_local_encoder_manifest(manifest_file)
    (
        required_held_out,
        required_held_out_artifacts,
        required_held_out_artifacts_sha256,
    ) = required_held_out_vocabulary()
    required_held_out_terms = sorted(required_held_out)
    required_held_out_terms_sha256 = _canonical_json_sha256(required_held_out_terms)
    pinned_relative = PINNED_HELD_OUT_VOCABULARY.relative_to(REPOSITORY_ROOT).as_posix()
    pinned_file_hash = next(
        artifact["sha256"]
        for artifact in required_held_out_artifacts
        if artifact["path"] == pinned_relative
    )
    held_out = sorted(
        required_held_out
        | {
            str(term).casefold().strip()
            for term in held_out_vocabulary
            if str(term).strip()
        }
    )
    held_out_hash = _canonical_json_sha256(held_out)
    forbidden_texts, exclusion_artifacts = _frozen_evaluation_exclusions()
    lesson_bytes = lesson_file.read_bytes()
    rows = _reviewed_rows_from_text(
        lesson_bytes.decode("utf-8"), held_out, forbidden_texts
    )
    parent_bytes = parent_file.read_bytes()
    parent_hash = _sha256_bytes(parent_bytes)
    lessons_hash = _sha256_bytes(lesson_bytes)
    parent = torch.load(io.BytesIO(parent_bytes), map_location="cpu", weights_only=True)
    if not isinstance(parent, Mapping):
        raise ValueError("parent checkpoint must contain a mapping")

    model = load_supported_local_encoder(
        manifest_file,
        expected_manifest_sha256=manifest["manifest_sha256"],
    )
    if model.artifact_metadata.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise RuntimeError(
            "loaded encoder manifest identity differs from the initially verified "
            "training manifest"
        )
    _reset_trainable_layers(model, seed)
    transition = _copy_parent_heads(model, parent, manifest["manifest_sha256"])
    model.train()
    encoder_features = _precompute_embeddings(
        model,
        [row["text"] for row in rows],
        batch_size=encoder_batch_size,
    )
    embedding_body = (
        b"kev.frozen-embeddings.v1\0float32-le\0"
        + json.dumps(list(encoder_features.shape), separators=(",", ":")).encode(
            "ascii"
        )
        + b"\0"
        + encoder_features.numpy().astype("<f4", copy=False).tobytes()
    )
    embedding_values_sha256 = hashlib.sha256(embedding_body).hexdigest()
    embedding_cache = {
        "schema": "kev.frozen-embeddings.v1",
        "encoder_manifest_sha256": manifest["manifest_sha256"],
        "lessons_sha256": lessons_hash,
        "shape": list(encoder_features.shape),
        "dtype": "float32-le",
        "values_sha256": embedding_values_sha256,
        "embeddings": encoder_features,
    }
    with embeddings_path.open("xb") as handle:
        torch.save(embedding_cache, handle)
    embeddings_file_sha256 = _sha256_bytes(embeddings_path.read_bytes())

    frame_targets, cardinality_targets, contrastive_groups = _targets(rows)
    paraphrase_groups = _paraphrase_group_manifest(rows)
    paraphrase_groups_sha256 = _canonical_json_sha256(paraphrase_groups)
    contrastive_positive_pairs = sum(
        group["rows"] * (group["rows"] - 1) // 2 for group in paraphrase_groups
    )
    objective = "L_relation + 0.20 L_cardinality + 0.35 L_contrastive"
    learning_rate = 5e-4
    weight_decay = 2e-4
    gradient_norm_clip = 1.0
    optimizer_config = {
        "name": "AdamW",
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_norm_clip": gradient_norm_clip,
    }
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    parameter_scope = {
        "total": model.parameter_count,
        "trainable": sum(parameter.numel() for parameter in parameters),
        "frozen": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
        "checkpoint": sum(
            value.numel() for value in model.proposal_state_dict().values()
        ),
    }
    optimizer = torch.optim.AdamW(
        parameters, lr=learning_rate, weight_decay=weight_decay
    )
    history: list[dict[str, float | int]] = []
    torch.set_num_threads(2)
    for step in range(steps):
        embeddings, frame_logits, cardinality_logits = model(encoder_features)
        relation_loss = F.binary_cross_entropy_with_logits(frame_logits, frame_targets)
        cardinality_loss = F.cross_entropy(cardinality_logits, cardinality_targets)
        contrastive_loss = supervised_contrastive_loss(embeddings, contrastive_groups)
        loss = (
            relation_loss
            + CARDINALITY_WEIGHT * cardinality_loss
            + CONTRASTIVE_WEIGHT * contrastive_loss
        )
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(parameters, gradient_norm_clip)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "relation": float(relation_loss.detach()),
                "cardinality": float(cardinality_loss.detach()),
                "contrastive": float(contrastive_loss.detach()),
            }
        )

    exclusion_hash = _canonical_json_sha256(
        sorted(
            (
                {"role": artifact["role"], "sha256": artifact["sha256"]}
                for artifact in exclusion_artifacts
            ),
            key=lambda item: (item["role"], item["sha256"]),
        )
    )
    _verified_training_source_manifest(training_source_manifest)
    trainer_source_hash = source_hashes["FROZEN_ENCODER_TRAINER"]
    encoder_runtime_source_hash = source_hashes["ENCODER_RUNTIME"]
    training_contract_source_hash = source_hashes["SHARED_SEMANTIC_TRAINING"]
    training_inputs = {
        "parent_sha256": parent_hash,
        "lessons_sha256": lessons_hash,
        "held_out_vocabulary_sha256": held_out_hash,
        "pinned_held_out_vocabulary_file_sha256": pinned_file_hash,
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "training_exclusions_sha256": exclusion_hash,
        "paraphrase_groups_sha256": paraphrase_groups_sha256,
        "encoder_manifest_sha256": manifest["manifest_sha256"],
        "frozen_embedding_values_sha256": embedding_values_sha256,
        "trainer_source_sha256": trainer_source_hash,
        "encoder_runtime_source_sha256": encoder_runtime_source_hash,
        "training_contract_source_sha256": training_contract_source_hash,
        "training_source_manifest": training_source_manifest,
        "training_source_manifest_sha256": training_source_manifest_sha256,
        "seed": seed,
        "steps": steps,
        "objective": objective,
        "optimizer": optimizer_config,
        "torch_num_threads": 2,
    }
    training_inputs_sha256 = _canonical_json_sha256(training_inputs)
    manifest_reference = {
        "path": _portable_path(manifest_file),
        "sha256": manifest["manifest_sha256"],
        "name": manifest["name"],
        "source": manifest["source"],
        "license": manifest["license"],
        "network_policy": "LOCAL_ONLY",
    }
    checkpoint = {
        "schema": FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
        "model_family": "FROZEN_SENTENCE_ENCODER_PROPOSAL",
        "proposal_state_dict": model.proposal_state_dict(),
        "encoder_manifest": manifest_reference,
        "parameters": parameter_scope,
        "seed": seed,
        "intents": INTENTS,
        "max_frames": MAX_FRAMES,
        "parent_sha256": parent_hash,
        "lessons_sha256": lessons_hash,
        "held_out_vocabulary_sha256": held_out_hash,
        "pinned_held_out_vocabulary_file_sha256": pinned_file_hash,
        "required_held_out_vocabulary_artifacts": required_held_out_artifacts,
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "training_exclusions_sha256": exclusion_hash,
        "paraphrase_groups_sha256": paraphrase_groups_sha256,
        "contrastive_positive_pairs": contrastive_positive_pairs,
        "frozen_embedding_values_sha256": embedding_values_sha256,
        "trainer_source_sha256": trainer_source_hash,
        "encoder_runtime_source_sha256": encoder_runtime_source_hash,
        "training_contract_source_sha256": training_contract_source_hash,
        "training_source_manifest": training_source_manifest,
        "training_source_manifest_sha256": training_source_manifest_sha256,
        "training_inputs_sha256": training_inputs_sha256,
        "training_scope": "projection_frame_and_cardinality_heads_only",
        "architecture_transition": transition,
        "steps": steps,
        "objective": objective,
        "optimizer": optimizer_config,
        "torch_num_threads": 2,
        "temperature": None,
        "score_status": "UNCALIBRATED",
        "artifact_status": "CHALLENGER",
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    checkpoint_hash = _sha256_bytes(checkpoint_path.read_bytes())
    environment: dict[str, str] = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
    }
    for package in ("transformers", "tokenizers", "safetensors"):
        try:
            module = __import__(package)
            environment[package] = str(module.__version__)
        except (ImportError, AttributeError):
            environment[package] = "UNAVAILABLE"
    receipt = {
        "schema": "kev.training-receipt.v1",
        "artifact_status": "CHALLENGER",
        "activation": "NONE",
        "model_family": "FROZEN_SENTENCE_ENCODER_PROPOSAL",
        "parent_path": str(parent_file),
        "parent_sha256": parent_hash,
        "architecture_transition": transition,
        "lessons_path": str(lesson_file),
        "lessons_sha256": lessons_hash,
        "held_out_vocabulary": held_out,
        "held_out_vocabulary_sha256": held_out_hash,
        "required_held_out_vocabulary": required_held_out_terms,
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "required_held_out_vocabulary_artifacts": required_held_out_artifacts,
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "pinned_held_out_vocabulary_path": str(PINNED_HELD_OUT_VOCABULARY),
        "pinned_held_out_vocabulary_expected_file_sha256": (
            PINNED_HELD_OUT_VOCABULARY_FILE_SHA256
        ),
        "pinned_held_out_vocabulary_file_sha256": pinned_file_hash,
        "training_exclusion_artifacts": exclusion_artifacts,
        "training_exclusions_sha256": exclusion_hash,
        "paraphrase_groups": paraphrase_groups,
        "paraphrase_groups_sha256": paraphrase_groups_sha256,
        "contrastive_positive_pairs": contrastive_positive_pairs,
        "encoder_artifact": {
            **manifest_reference,
            "files": manifest["files"],
            "permission": manifest["permission"],
            "loader": manifest["loader"],
        },
        "frozen_embedding_cache": {
            "path": str(embeddings_path),
            "file_sha256": embeddings_file_sha256,
            "values_sha256": embedding_values_sha256,
            "shape": list(encoder_features.shape),
            "encoder_batch_size": encoder_batch_size,
        },
        "trainer_source_sha256": trainer_source_hash,
        "encoder_runtime_source_sha256": encoder_runtime_source_hash,
        "training_contract_source_sha256": training_contract_source_hash,
        "training_source_manifest": training_source_manifest,
        "training_source_manifest_sha256": training_source_manifest_sha256,
        "training_inputs": training_inputs,
        "training_inputs_sha256": training_inputs_sha256,
        "training_scope": "projection_frame_and_cardinality_heads_only",
        "parameter_scope": parameter_scope,
        "challenger_path": str(checkpoint_path),
        "challenger_sha256": checkpoint_hash,
        "seed": seed,
        "steps": steps,
        "examples": len(rows),
        "objective": objective,
        "optimizer": optimizer_config,
        "torch_num_threads": 2,
        "loss_history": history,
        "promotion_features": [],
        "score_status": "UNCALIBRATED",
        "environment": environment,
    }
    with receipt_path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return receipt

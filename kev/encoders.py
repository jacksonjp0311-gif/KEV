"""Optional frozen sentence-encoder front-end for KEV frame proposals.

The encoder supplies language features only.  It has no state, tool, or model
activation authority.  Only the projection and small proposal heads are
trainable, which keeps broader language separate from cognitive authority.
"""

from __future__ import annotations

from collections.abc import Sequence
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Mapping
from typing import Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F

from kev.uc51a2.semantic_breadth import EMB, INTENTS, MAX_FRAMES
from kev.artifacts import validate_sha256, verify_file_sha256


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_SENTENCE_CHECKPOINT_SCHEMA = "kev.frozen-sentence-frames.v1"
HF_BERT_MEAN_POOLING_LOADER = "hf-bert-mean-pooling-normalize-v1"


class LocalSentenceEncoder(Protocol):
    """Minimal interface expected from a permission-approved local encoder."""

    def encode(self, texts: Sequence[str]) -> torch.Tensor: ...


class FrozenSentenceEncoderProposal(nn.Module):
    """Frozen local encoder with trainable projection and typed proposal heads."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int,
        *,
        artifact_metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if encoder_dim <= 0:
            raise ValueError("encoder_dim must be positive")
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.artifact_metadata = copy.deepcopy(artifact_metadata or {})
        self.checkpoint_meta: dict[str, Any] = {}
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()
        self.projection = nn.Linear(encoder_dim, EMB)
        self.frame_head = nn.Linear(EMB, len(INTENTS))
        self.cardinality_head = nn.Linear(EMB, MAX_FRAMES + 1)

    def train(self, mode: bool = True) -> "FrozenSentenceEncoderProposal":
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(
        self, encoder_features: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if encoder_features.ndim != 2 or encoder_features.shape[1] != self.encoder_dim:
            raise ValueError(
                f"encoder features must have shape [batch,{self.encoder_dim}]"
            )
        embedding = F.normalize(self.projection(encoder_features), dim=-1)
        return embedding, self.frame_head(embedding), self.cardinality_head(embedding)

    def encode_texts(self, texts: Sequence[str]) -> torch.Tensor:
        """Run a supplied local encoder without constructing a gradient graph."""

        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("texts must be a non-empty sequence of strings")
        encode = getattr(self.encoder, "encode", None)
        with torch.no_grad():
            raw: Any = (
                encode(list(texts)) if callable(encode) else self.encoder(list(texts))
            )
            features = torch.as_tensor(raw, dtype=torch.float32)
        if features.ndim == 1 and len(texts) == 1:
            features = features[None, :]
        if features.ndim != 2 or features.shape != (len(texts), self.encoder_dim):
            raise ValueError(
                f"local encoder returned {tuple(features.shape)}, expected "
                f"({len(texts)},{self.encoder_dim})"
            )
        return features

    def forward_texts(
        self, texts: Sequence[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self(self.encode_texts(texts))

    def proposals(
        self, texts: Sequence[str], threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """Produce uncalibrated kind/cardinality proposals, never state writes."""

        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold must be between zero and one")
        with torch.no_grad():
            embeddings, logits, cardinality_logits = self.forward_texts(texts)
            temperature = _temperature(self.checkpoint_meta)
            scaled = logits if temperature is None else logits / temperature
            probabilities = torch.sigmoid(scaled)
            cardinalities = cardinality_logits.argmax(dim=1)
        results: list[dict[str, Any]] = []
        for row, cardinality, embedding in zip(
            probabilities, cardinalities, embeddings
        ):
            ranked = sorted(
                (
                    (INTENTS[index], round(float(score), 6))
                    for index, score in enumerate(row)
                ),
                key=lambda item: (-item[1], item[0]),
            )
            count = int(cardinality)
            selected = (
                ranked[: min(count, MAX_FRAMES)]
                if count > 0
                else [item for item in ranked if item[1] >= threshold]
            )
            results.append(
                {
                    "proposals": [
                        {"kind": kind, "score": score} for kind, score in selected
                    ],
                    "cardinality": count,
                    "score_status": (
                        "CALIBRATED" if temperature is not None else "UNCALIBRATED"
                    ),
                    "temperature": temperature,
                    "embedding": embedding.tolist(),
                    "authority": "PROPOSAL_ONLY",
                    "encoder_artifact": copy.deepcopy(self.artifact_metadata),
                }
            )
        return results

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def proposal_state_dict(self) -> dict[str, torch.Tensor]:
        """Return only trainable proposal bytes; frozen encoder bytes stay external."""

        state: dict[str, torch.Tensor] = {}
        for prefix, module in (
            ("projection", self.projection),
            ("frame_head", self.frame_head),
            ("cardinality_head", self.cardinality_head),
        ):
            for name, value in module.state_dict().items():
                state[f"{prefix}.{name}"] = value.detach().cpu().clone()
        return state

    def load_proposal_state_dict(self, state: Mapping[str, Any]) -> None:
        """Load exactly the projection and typed heads from a small checkpoint."""

        if not isinstance(state, Mapping):
            raise ValueError("proposal_state_dict must be a mapping")
        expected = {
            "projection.weight",
            "projection.bias",
            "frame_head.weight",
            "frame_head.bias",
            "cardinality_head.weight",
            "cardinality_head.bias",
        }
        supplied = set(state)
        if supplied != expected:
            missing = sorted(expected - supplied)
            unexpected = sorted(supplied - expected)
            raise ValueError(
                "proposal state keys do not match the frozen-encoder head schema: "
                f"missing={missing}, unexpected={unexpected}"
            )
        self.projection.load_state_dict(
            {"weight": state["projection.weight"], "bias": state["projection.bias"]},
            strict=True,
        )
        self.frame_head.load_state_dict(
            {"weight": state["frame_head.weight"], "bias": state["frame_head.bias"]},
            strict=True,
        )
        self.cardinality_head.load_state_dict(
            {
                "weight": state["cardinality_head.weight"],
                "bias": state["cardinality_head.bias"],
            },
            strict=True,
        )


EncoderLoader = Callable[[Path, dict[str, Any]], nn.Module]


def _temperature(metadata: Mapping[str, Any]) -> float | None:
    if metadata.get("_kev_calibration_verified") is not True:
        return None
    value = metadata.get("temperature")
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    return None


def verify_local_encoder_manifest(manifest_path: str | Path) -> dict[str, Any]:
    """Verify a permission-approved, local-only sentence encoder artifact.

    This function performs no network access and accepts only relative files
    contained beside the manifest. A caller-provided loader is responsible for
    understanding the model format after every declared byte hash verifies.
    """

    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"encoder manifest is not a file: {path}")
    raw_manifest = path.read_bytes()
    try:
        manifest = json.loads(raw_manifest.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("encoder manifest must be UTF-8 JSON") from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("schema") != "kev.local-encoder.v1"
    ):
        raise ValueError("invalid local encoder manifest schema")
    if manifest.get("network_policy") != "LOCAL_ONLY":
        raise ValueError("encoder manifest must declare network_policy LOCAL_ONLY")
    name = manifest.get("name")
    dimension = manifest.get("embedding_dim")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("encoder manifest requires a name")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("encoder manifest requires a positive embedding_dim")

    permission = manifest.get("permission")
    if not isinstance(permission, dict) or permission.get("status") != "APPROVED":
        raise ValueError("encoder manifest requires approved permission provenance")
    for field in ("reviewed_by", "reviewed_at", "scope"):
        if not isinstance(permission.get(field), str) or not permission[field].strip():
            raise ValueError(f"encoder permission requires {field}")
    license_record = manifest.get("license")
    if not isinstance(license_record, dict) or not isinstance(
        license_record.get("identifier"), str
    ):
        raise ValueError("encoder manifest requires a license identifier")
    if not license_record["identifier"].strip():
        raise ValueError("encoder license identifier cannot be empty")

    loader = manifest.get("loader")
    if loader is not None:
        if not isinstance(loader, dict) or not isinstance(loader.get("id"), str):
            raise ValueError(
                "encoder loader must identify an allowlisted implementation"
            )
        if loader["id"] == HF_BERT_MEAN_POOLING_LOADER:
            required_strings = (
                "config_path",
                "model_path",
                "tokenizer_path",
                "pooling_config_path",
                "modules_path",
            )
            for field in required_strings:
                if not isinstance(loader.get(field), str) or not loader[field].strip():
                    raise ValueError(f"encoder loader requires {field}")
            maximum = loader.get("max_length")
            if (
                isinstance(maximum, bool)
                or not isinstance(maximum, int)
                or not 1 <= maximum <= 512
            ):
                raise ValueError("encoder loader max_length must be in 1..512")
            if loader.get("pooling") != "MEAN" or loader.get("normalize") is not True:
                raise ValueError(
                    "supported encoder loader requires mean pooling and normalization"
                )

    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("encoder manifest requires at least one hashed local file")
    root = path.parent.resolve()
    verified_files: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, record in enumerate(files):
        if not isinstance(record, dict):
            raise ValueError(f"encoder file {index} must be an object")
        raw = record.get("path")
        if not isinstance(raw, str) or not raw.strip() or "://" in raw:
            raise ValueError(f"encoder file {index} must use a local relative path")
        relative = Path(raw)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"encoder file {index} escapes the manifest directory")
        candidate = (root / relative).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"encoder file {index} escapes the manifest directory"
            ) from error
        normalized = relative.as_posix()
        if normalized in seen:
            raise ValueError(f"encoder file {index} duplicates {normalized!r}")
        seen.add(normalized)
        expected = validate_sha256(
            str(record.get("sha256", "")), field="encoder file sha256"
        )
        verify_file_sha256(candidate, expected)
        verified_files.append(
            {
                "path": normalized,
                "sha256": expected,
                "size_bytes": candidate.stat().st_size,
            }
        )

    if isinstance(loader, dict) and loader.get("id") == HF_BERT_MEAN_POOLING_LOADER:
        declared = {record["path"] for record in verified_files}
        referenced = {
            str(loader[field]).replace("\\", "/")
            for field in (
                "config_path",
                "model_path",
                "tokenizer_path",
                "pooling_config_path",
                "modules_path",
            )
        }
        if not referenced <= declared:
            raise ValueError("encoder loader references an undeclared local file")

    source = manifest.get("source")
    if loader is not None:
        if not isinstance(source, dict):
            raise ValueError("supported encoder manifest requires source provenance")
        for field in ("provider", "repository", "revision", "url"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"encoder source requires {field}")
        revision = source["revision"].casefold()
        if len(revision) != 40 or any(
            character not in "0123456789abcdef" for character in revision
        ):
            raise ValueError(
                "encoder source revision must be a 40-character commit hash"
            )

    return {
        "schema": manifest["schema"],
        "name": name.strip(),
        "embedding_dim": dimension,
        "network_policy": "LOCAL_ONLY",
        "permission": copy.deepcopy(permission),
        "license": copy.deepcopy(license_record),
        "source": copy.deepcopy(source),
        "loader": copy.deepcopy(loader),
        "files": verified_files,
        "manifest_path": str(path),
        "manifest_sha256": hashlib.sha256(raw_manifest).hexdigest(),
    }


def load_verified_local_encoder(
    manifest_path: str | Path,
    loader: EncoderLoader,
) -> FrozenSentenceEncoderProposal:
    """Load verified local bytes through trusted caller code and freeze them."""

    if not callable(loader):
        raise TypeError("loader must be callable")
    verified = verify_local_encoder_manifest(manifest_path)
    root = Path(verified["manifest_path"]).parent
    encoder = loader(root, copy.deepcopy(verified))
    if not isinstance(encoder, nn.Module):
        raise TypeError("local encoder loader must return torch.nn.Module")
    return FrozenSentenceEncoderProposal(
        encoder,
        int(verified["embedding_dim"]),
        artifact_metadata={
            "name": verified["name"],
            "manifest_sha256": verified["manifest_sha256"],
            "network_policy": verified["network_policy"],
            "permission": verified["permission"],
            "license": verified["license"],
            "source": verified["source"],
            "loader": verified["loader"],
            "files": verified["files"],
        },
    )


class HuggingFaceBertMeanEncoder(nn.Module):
    """A fixed BERT encoder using the model card's mean-pool + normalize recipe."""

    def __init__(
        self,
        transformer: nn.Module,
        tokenizer: Any,
        *,
        max_length: int,
        pad_token_id: int,
    ) -> None:
        super().__init__()
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(
            pad_id=self.pad_token_id,
            pad_token="[PAD]",
        )
        encoded = self.tokenizer.encode_batch(list(texts))
        input_ids = torch.tensor([item.ids for item in encoded], dtype=torch.long)
        attention_mask = torch.tensor(
            [item.attention_mask for item in encoded], dtype=torch.long
        )
        token_type_ids = torch.tensor(
            [item.type_ids for item in encoded], dtype=torch.long
        )
        with torch.no_grad():
            output = self.transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).last_hidden_state
            expanded_mask = attention_mask.unsqueeze(-1).to(output.dtype)
            pooled = (output * expanded_mask).sum(dim=1) / expanded_mask.sum(
                dim=1
            ).clamp_min(1e-9)
            return F.normalize(pooled, p=2, dim=1).cpu()


def _read_verified_file(
    root: Path, verified: Mapping[str, Any], relative_path: str
) -> bytes:
    """Read and hash the exact bytes subsequently consumed by the safe loader."""

    normalized = Path(relative_path).as_posix()
    records = {
        str(record["path"]): str(record["sha256"]) for record in verified["files"]
    }
    if normalized not in records:
        raise ValueError(f"encoder file is not declared by the manifest: {normalized}")
    path = (root / normalized).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError("encoder file escapes the verified root") from error
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != records[normalized]:
        raise ValueError(
            f"encoder file SHA-256 mismatch while loading {normalized}: "
            f"expected {records[normalized]}, got {actual}"
        )
    return payload


def _load_hf_bert_mean_encoder(root: Path, verified: dict[str, Any]) -> nn.Module:
    loader = verified.get("loader")
    if not isinstance(loader, dict) or loader.get("id") != HF_BERT_MEAN_POOLING_LOADER:
        raise ValueError("encoder manifest does not select the supported BERT loader")
    try:
        from safetensors.torch import load as load_safetensors
        from tokenizers import Tokenizer  # type: ignore[import-untyped]
        from transformers import BertConfig, BertModel
    except ImportError as error:
        raise RuntimeError(
            "the frozen sentence encoder requires the optional 'semantic-encoder' "
            "dependencies; install kev-cognitive-runtime[semantic-encoder]"
        ) from error

    config_payload = _read_verified_file(root, verified, loader["config_path"])
    model_payload = _read_verified_file(root, verified, loader["model_path"])
    tokenizer_payload = _read_verified_file(root, verified, loader["tokenizer_path"])
    pooling_payload = _read_verified_file(root, verified, loader["pooling_config_path"])
    modules_payload = _read_verified_file(root, verified, loader["modules_path"])

    config_data = json.loads(config_payload.decode("utf-8"))
    pooling = json.loads(pooling_payload.decode("utf-8"))
    modules = json.loads(modules_payload.decode("utf-8"))
    if config_data.get("model_type") != "bert" or config_data.get("architectures") != [
        "BertModel"
    ]:
        raise ValueError("only a pinned BertModel configuration is accepted")
    if int(config_data.get("hidden_size", 0)) != int(verified["embedding_dim"]):
        raise ValueError("encoder hidden size differs from manifest embedding_dim")
    if not (
        pooling.get("pooling_mode_mean_tokens") is True
        and pooling.get("pooling_mode_cls_token") is False
        and pooling.get("pooling_mode_max_tokens") is False
        and isinstance(modules, list)
        and [record.get("type") for record in modules]
        == [
            "sentence_transformers.models.Transformer",
            "sentence_transformers.models.Pooling",
            "sentence_transformers.models.Normalize",
        ]
    ):
        raise ValueError("encoder bundle does not match mean-pool + normalize recipe")

    configuration = BertConfig.from_dict(config_data)
    transformer = BertModel(configuration)
    state = load_safetensors(model_payload)
    incompatible = transformer.load_state_dict(state, strict=False)
    allowed_unexpected = {"embeddings.position_ids"}
    if (
        incompatible.missing_keys
        or not set(incompatible.unexpected_keys) <= allowed_unexpected
    ):
        raise ValueError(
            "encoder safetensors do not exactly match the pinned BertModel: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    transformer.eval()
    tokenizer = Tokenizer.from_str(tokenizer_payload.decode("utf-8"))
    return HuggingFaceBertMeanEncoder(
        transformer,
        tokenizer,
        max_length=int(loader["max_length"]),
        pad_token_id=int(config_data.get("pad_token_id", 0)),
    )


def load_supported_local_encoder(
    manifest_path: str | Path,
    *,
    expected_manifest_sha256: str | None = None,
) -> FrozenSentenceEncoderProposal:
    """Load one allowlisted local encoder without network or dynamic model code."""

    verified = verify_local_encoder_manifest(manifest_path)
    if expected_manifest_sha256 is not None:
        expected = validate_sha256(
            expected_manifest_sha256, field="expected_manifest_sha256"
        )
        if verified["manifest_sha256"] != expected:
            raise ValueError(
                "encoder manifest changed between verification and loading: "
                f"expected {expected}, got {verified['manifest_sha256']}"
            )
    loader = verified.get("loader")
    if not isinstance(loader, dict) or loader.get("id") != HF_BERT_MEAN_POOLING_LOADER:
        raise ValueError("local encoder loader is not allowlisted")
    root = Path(verified["manifest_path"]).parent
    encoder = _load_hf_bert_mean_encoder(root, verified)
    return FrozenSentenceEncoderProposal(
        encoder,
        int(verified["embedding_dim"]),
        artifact_metadata={
            key: copy.deepcopy(verified[key])
            for key in (
                "name",
                "manifest_sha256",
                "network_policy",
                "permission",
                "license",
                "source",
                "loader",
                "files",
            )
        },
    )


def resolve_encoder_manifest_reference(
    checkpoint: Mapping[str, Any], override: str | Path | None = None
) -> Path:
    reference = checkpoint.get("encoder_manifest")
    if not isinstance(reference, Mapping):
        raise ValueError("frozen sentence checkpoint requires encoder_manifest")
    raw_path = override if override is not None else reference.get("path")
    if not isinstance(raw_path, (str, Path)) or not str(raw_path):
        raise ValueError("frozen sentence checkpoint requires an encoder manifest path")
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    payload = path.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    expected = validate_sha256(
        str(reference.get("sha256", "")), field="encoder_manifest.sha256"
    )
    if actual != expected:
        raise ValueError(
            f"encoder manifest SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return path


def load_frozen_sentence_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    encoder_manifest: str | Path | None = None,
) -> FrozenSentenceEncoderProposal:
    """Materialize a small proposal-head checkpoint over its pinned local encoder."""

    if checkpoint.get("schema") != FROZEN_SENTENCE_CHECKPOINT_SCHEMA:
        raise ValueError("invalid frozen sentence checkpoint schema")
    if tuple(checkpoint.get("intents", ())) != INTENTS:
        raise ValueError("frozen sentence checkpoint intent order mismatch")
    if checkpoint.get("max_frames") != MAX_FRAMES:
        raise ValueError("frozen sentence checkpoint cardinality schema mismatch")
    manifest_path = resolve_encoder_manifest_reference(checkpoint, encoder_manifest)
    reference = checkpoint["encoder_manifest"]
    model = load_supported_local_encoder(
        manifest_path,
        expected_manifest_sha256=str(reference.get("sha256", "")),
    )
    if model.artifact_metadata.get("manifest_sha256") != reference.get("sha256"):
        raise ValueError("loaded encoder manifest does not match checkpoint binding")
    model.load_proposal_state_dict(checkpoint.get("proposal_state_dict", {}))
    model.checkpoint_meta = {
        key: copy.deepcopy(value)
        for key, value in checkpoint.items()
        if key != "proposal_state_dict"
    }
    model.eval()
    return model

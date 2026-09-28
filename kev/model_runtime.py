from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch

from kev.calibration_evidence import (
    CalibrationReceiptArtifact,
    validate_runtime_calibration,
)
from kev.encoders import (
    FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
    FrozenSentenceEncoderProposal,
    load_frozen_sentence_checkpoint,
)
from kev.frames import Frame, extract_frames
from kev.uc51a2.semantic_breadth import (
    SemanticBreadth,
    features,
    load_bytes,
    predict_frame_kinds,
)


def semantic_frame(frame: Frame) -> dict[str, Any]:
    """Project a frame to the surface/provenance-independent eval shape."""

    return {
        "kind": frame.kind,
        "relation": frame.relation,
        "slots": {name: slot.value for name, slot in sorted(frame.slots.items())},
    }


def _checkpoint_mapping(checkpoint_bytes: bytes) -> Mapping[str, Any]:
    value = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(value, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return value


def load_proposal_model_bytes(
    checkpoint_bytes: bytes,
    *,
    encoder_manifest: str | Path | None = None,
    semantic_loader: Callable[[bytes], SemanticBreadth] | None = None,
) -> SemanticBreadth | FrozenSentenceEncoderProposal:
    """Load either proposal architecture from one captured checkpoint payload."""

    checkpoint = _checkpoint_mapping(checkpoint_bytes)
    model: SemanticBreadth | FrozenSentenceEncoderProposal
    if checkpoint.get("schema") == FROZEN_SENTENCE_CHECKPOINT_SCHEMA:
        model = load_frozen_sentence_checkpoint(
            checkpoint, encoder_manifest=encoder_manifest
        )
    else:
        model = (semantic_loader or load_bytes)(checkpoint_bytes)
    _clear_model_calibration(model)
    return model


def _clear_model_calibration(
    model: SemanticBreadth | FrozenSentenceEncoderProposal,
) -> bool:
    """Remove unverified embedded calibration metadata from a loaded model."""

    metadata = model.checkpoint_meta
    embedded = bool(metadata.get("_kev_embedded_calibration_present")) or (
        metadata.get("temperature") is not None
    )
    metadata["_kev_embedded_calibration_present"] = embedded
    metadata["temperature"] = None
    metadata["score_status"] = "UNCALIBRATED"
    metadata.pop("_kev_calibration_verified", None)
    metadata.pop("_kev_calibration_receipt_sha256", None)
    return embedded


def configure_checkpoint_calibration(
    model: SemanticBreadth | FrozenSentenceEncoderProposal,
    checkpoint_path: str | Path,
    checkpoint_bytes: bytes,
    *,
    calibration_receipt: CalibrationReceiptArtifact
    | Mapping[str, Any]
    | str
    | Path
    | None = None,
    calibration_fit_path: str | Path | None = None,
    promotion_suite_path: str | Path | None = None,
    reject_invalid: bool = False,
) -> tuple[CalibrationReceiptArtifact | None, str | None]:
    """Apply a temperature only after validating all retained evidence.

    Missing evidence and invalid auto-discovered sibling evidence downgrade the
    model to uncalibrated output. Callers that explicitly supply evidence may
    request rejection through ``reject_invalid``.
    """

    embedded = _clear_model_calibration(model)
    try:
        evidence = validate_runtime_calibration(
            checkpoint_path,
            checkpoint_bytes,
            receipt=calibration_receipt,
            calibration_fit_path=calibration_fit_path,
            promotion_suite_path=promotion_suite_path,
        )
    except (OSError, TypeError, ValueError) as error:
        if reject_invalid:
            raise ValueError(f"invalid calibration evidence: {error}") from error
        return None, f"CALIBRATION_RECEIPT_INVALID: {error}"
    if evidence is None:
        return (
            None,
            "CALIBRATION_RECEIPT_MISSING" if embedded else None,
        )

    metadata = model.checkpoint_meta
    metadata["temperature"] = evidence.temperature
    metadata["score_status"] = "CALIBRATED"
    metadata["_kev_calibration_verified"] = True
    metadata["_kev_calibration_receipt_sha256"] = evidence.canonical_sha256
    return evidence, None


def predict_proposal(
    model: SemanticBreadth | FrozenSentenceEncoderProposal, text: str
) -> dict[str, Any]:
    if model.checkpoint_meta.get("_kev_calibration_verified") is not True:
        _clear_model_calibration(model)
    if isinstance(model, FrozenSentenceEncoderProposal):
        return model.proposals([text])[0]
    return predict_frame_kinds(model, text)


def proposal_logits(
    model: SemanticBreadth | FrozenSentenceEncoderProposal,
    texts: Sequence[str],
) -> torch.Tensor:
    """Return unscaled frame logits for calibration without training the encoder."""

    if not texts:
        raise ValueError("texts must be non-empty")
    with torch.no_grad():
        if isinstance(model, FrozenSentenceEncoderProposal):
            _, logits, _ = model.forward_texts(texts)
            return logits.detach()
        inputs = torch.stack([features(text) for text in texts])
        _, logits = model(inputs)
        return logits.detach()


class CheckpointPredictor:
    """Connect a checkpoint to model-neutral frozen-suite evaluation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        encoder_manifest: str | Path | None = None,
        calibration_receipt: CalibrationReceiptArtifact
        | Mapping[str, Any]
        | str
        | Path
        | None = None,
        calibration_fit_path: str | Path | None = None,
        promotion_suite_path: str | Path | None = None,
    ) -> None:
        self.path = Path(checkpoint_path).resolve()
        checkpoint_bytes = self.path.read_bytes()
        self.sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
        self.model = load_proposal_model_bytes(
            checkpoint_bytes, encoder_manifest=encoder_manifest
        )
        self.calibration, self.calibration_error = configure_checkpoint_calibration(
            self.model,
            self.path,
            checkpoint_bytes,
            calibration_receipt=calibration_receipt,
            calibration_fit_path=calibration_fit_path,
            promotion_suite_path=promotion_suite_path,
            reject_invalid=(
                calibration_receipt is not None
                or calibration_fit_path is not None
                or promotion_suite_path is not None
            ),
        )

    def __call__(self, item: Mapping[str, Any]) -> dict[str, Any]:
        text = str(item.get("input", item.get("text", "")))
        proposal = predict_proposal(self.model, text)
        proposal_result = {
            "kinds": [record["kind"] for record in proposal["proposals"]],
            "scores": {
                record["kind"]: record["score"] for record in proposal["proposals"]
            },
            "cardinality": proposal["cardinality"],
        }
        if item.get("projection") == "kinds":
            # Label-only/OOV audits measure the neural proposal head directly;
            # they never create state and therefore require no invented slots.
            predicted: list[Any] = sorted(set(proposal_result["kinds"]))
        else:
            frames = extract_frames(
                text,
                source_type="EVALUATION",
                source_id=str(item.get("id", "unknown")),
                actor="frozen-suite",
                model_hash=self.sha256,
                proposal_result=proposal_result,
                timestamp="1970-01-01T00:00:00Z",
            )
            predicted = [semantic_frame(frame) for frame in frames]
        selected_scores = [record["score"] for record in proposal["proposals"]]
        confidence = (
            sum(selected_scores) / len(selected_scores) if selected_scores else None
        )
        return {
            "predicted": predicted,
            "confidence": confidence,
            "score_status": proposal["score_status"],
            "temperature": proposal["temperature"],
            "model_sha256": self.sha256,
        }


def checkpoint_predictor(
    checkpoint_path: str | Path,
    *,
    encoder_manifest: str | Path | None = None,
    calibration_receipt: CalibrationReceiptArtifact
    | Mapping[str, Any]
    | str
    | Path
    | None = None,
    calibration_fit_path: str | Path | None = None,
    promotion_suite_path: str | Path | None = None,
) -> CheckpointPredictor:
    return CheckpointPredictor(
        checkpoint_path,
        encoder_manifest=encoder_manifest,
        calibration_receipt=calibration_receipt,
        calibration_fit_path=calibration_fit_path,
        promotion_suite_path=promotion_suite_path,
    )

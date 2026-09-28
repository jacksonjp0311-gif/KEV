from __future__ import annotations

import hashlib
import io
import json
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


BASE = (
    "COPY_VALUE",
    "SUPERSEDES",
    "MAGNITUDE",
    "NEGATE",
    "REFERENCE",
    "ACTIVE_SELECTION",
    "RUN_STATUS",
    "RECEIPT_VALUE",
    "EVIDENCE_CONSISTENCY",
)
NEW = ("GOAL", "CONSTRAINT", "OBSERVATION", "PREDICTION")
INTENTS = BASE + NEW
D = 1024
EMB = 80
MAX_FRAMES = 6
CONTRASTIVE_WEIGHT = 0.35
CARDINALITY_WEIGHT = 0.20
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PINNED_HELD_OUT_VOCABULARY = (
    REPOSITORY_ROOT / "evals" / "frozen" / "held-out-vocabulary-v1.txt"
)
PINNED_HELD_OUT_VOCABULARY_FILE_SHA256 = (
    "e7b7c7ec703e378a9d3ff419b00f9b67e370c07ff2c773d8f6262ec83f2dba3a"
)
FROZEN_EVAL_DIRECTORY = REPOSITORY_ROOT / "evals" / "frozen"
PINNED_EVALUATION_MANIFEST = FROZEN_EVAL_DIRECTORY / "manifest-v6.json"
PINNED_EVALUATION_MANIFEST_FILE_SHA256 = (
    "07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce"
)

CANON = {
    "goal": "GOAL",
    "objective": "GOAL",
    "aim": "GOAL",
    "target": "GOAL",
    "desired": "GOAL",
    "want": "GOAL",
    "constraint": "CONSTRAINT",
    "requirement": "CONSTRAINT",
    "must": "CONSTRAINT",
    "cannot": "CONSTRAINT",
    "forbidden": "CONSTRAINT",
    "limit": "CONSTRAINT",
    "observed": "OBSERVATION",
    "observation": "OBSERVATION",
    "measured": "OBSERVATION",
    "actual": "OBSERVATION",
    "recorded": "OBSERVATION",
    "predict": "PREDICTION",
    "prediction": "PREDICTION",
    "expect": "PREDICTION",
    "forecast": "PREDICTION",
    "anticipate": "PREDICTION",
    "likely": "PREDICTION",
    "old": "OLD",
    "former": "OLD",
    "obsolete": "OLD",
    "replace": "UPDATE",
    "replaced": "UPDATE",
    "superseded": "UPDATE",
    "current": "CURRENT",
    "now": "CURRENT",
    "absolute": "MAGNITUDE",
    "magnitude": "MAGNITUDE",
    "distance": "MAGNITUDE",
    "negate": "NEGATE",
    "inverse": "NEGATE",
    "opposite": "NEGATE",
    "active": "ACTIVE",
    "serving": "ACTIVE",
    "inactive": "INACTIVE",
    "candidate": "INACTIVE",
    "receipt": "RECEIPT",
    "result": "RESULT",
    "output": "RESULT",
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _pinned_evaluation_manifest() -> tuple[dict[str, Any], dict[str, Any]]:
    """Load the code-pinned manifest that closes the frozen input inventory."""

    payload = PINNED_EVALUATION_MANIFEST.read_bytes()
    actual = _sha256_bytes(payload)
    if actual != PINNED_EVALUATION_MANIFEST_FILE_SHA256:
        raise ValueError(
            "pinned evaluation manifest SHA-256 mismatch: "
            f"expected {PINNED_EVALUATION_MANIFEST_FILE_SHA256}, got {actual}"
        )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("pinned evaluation manifest must be UTF-8 JSON") from error
    if (
        not isinstance(value, dict)
        or value.get("schema") != "kev.eval-manifest.v1"
        or value.get("frozen") is not True
        or value.get("version") != 6
        or not isinstance(value.get("artifacts"), Mapping)
    ):
        raise ValueError("invalid pinned evaluation manifest-v6")
    return value, {
        "path": PINNED_EVALUATION_MANIFEST.relative_to(REPOSITORY_ROOT).as_posix(),
        "sha256": actual,
    }


def _manifest_target(raw_reference: Mapping[str, Any], *, role: str) -> Path:
    raw_value = raw_reference.get("path")
    if not isinstance(raw_value, str) or not raw_value.strip():
        raise ValueError(f"pinned manifest artifact {role!r} lacks a path")
    raw_path = Path(raw_value)
    if raw_path.is_absolute() or ".." in raw_path.parts:
        raise ValueError(
            f"pinned manifest artifact path escapes repository: {raw_path}"
        )
    target = (REPOSITORY_ROOT / raw_path).resolve()
    try:
        target.relative_to(REPOSITORY_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"pinned manifest artifact path escapes repository: {raw_path}"
        ) from error
    return target


def required_held_out_vocabulary() -> tuple[set[str], list[dict[str, Any]], str]:
    """Load every manifest-declared frozen held-out vocabulary artifact.

    Direct trainer calls use this same boundary as ``kev evolve``. A caller may
    add exclusions, but it cannot omit an older or current frozen vocabulary.
    The legacy v1 digest remains code-pinned as an independent bootstrap check.
    """

    if not PINNED_HELD_OUT_VOCABULARY.is_file():
        raise FileNotFoundError(
            f"pinned held-out vocabulary is required: {PINNED_HELD_OUT_VOCABULARY}"
        )
    bootstrap_bytes = PINNED_HELD_OUT_VOCABULARY.read_bytes()
    bootstrap_hash = _sha256_bytes(bootstrap_bytes)
    if bootstrap_hash != PINNED_HELD_OUT_VOCABULARY_FILE_SHA256:
        raise ValueError(
            "pinned held-out vocabulary digest does not match the frozen v1 artifact"
        )

    declarations: dict[Path, dict[str, Any]] = {}
    manifest, manifest_record = _pinned_evaluation_manifest()
    for role, raw_reference in manifest["artifacts"].items():
        if not isinstance(raw_reference, Mapping) or not isinstance(
            raw_reference.get("path"), str
        ):
            continue
        raw_path = Path(raw_reference["path"])
        if not re.fullmatch(r"held-out-vocabulary-v\d+\.txt", raw_path.name):
            continue
        target = _manifest_target(raw_reference, role=str(role))
        expected = str(raw_reference.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid held-out vocabulary hash in manifest-v6")
        record = declarations.setdefault(
            target,
            {"sha256": expected, "declared_by": []},
        )
        if record["sha256"] != expected:
            raise ValueError(f"conflicting frozen hashes for {target}")
        record["declared_by"].append({**manifest_record, "role": str(role)})

    discovered = {
        path.resolve()
        for path in FROZEN_EVAL_DIRECTORY.glob("held-out-vocabulary-v*.txt")
    }
    declared = set(declarations)
    if discovered != declared:
        unpinned = sorted(str(path) for path in discovered - declared)
        missing = sorted(str(path) for path in declared - discovered)
        raise ValueError(
            "frozen held-out vocabulary inventory mismatch: "
            f"unpinned={unpinned}, missing={missing}"
        )
    if PINNED_HELD_OUT_VOCABULARY.resolve() not in declared:
        raise ValueError("code-pinned v1 held-out vocabulary is absent from manifests")

    required_terms: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for path in sorted(declared, key=lambda value: value.as_posix()):
        payload = path.read_bytes()
        actual = _sha256_bytes(payload)
        if actual != declarations[path]["sha256"]:
            raise ValueError(
                f"held-out vocabulary SHA-256 mismatch for {path}: "
                f"expected {declarations[path]['sha256']}, got {actual}"
            )
        try:
            terms = sorted(
                {
                    line.casefold().strip()
                    for line in payload.decode("utf-8").splitlines()
                    if line.strip()
                }
            )
        except UnicodeDecodeError as error:
            raise ValueError(f"held-out vocabulary must be UTF-8: {path}") from error
        if not terms:
            raise ValueError(f"held-out vocabulary is empty: {path}")
        required_terms.update(terms)
        artifacts.append(
            {
                "path": path.relative_to(REPOSITORY_ROOT).as_posix(),
                "sha256": actual,
                "size_bytes": len(payload),
                "terms": terms,
                "terms_sha256": _canonical_json_sha256(terms),
                "declared_by": sorted(
                    declarations[path]["declared_by"],
                    key=lambda item: (item["path"], item["role"]),
                ),
            }
        )
    artifacts_sha256 = _canonical_json_sha256(artifacts)
    return required_terms, artifacts, artifacts_sha256


def _h(value: str) -> int:
    """Stable feature bucket; Python's randomized hash is never used."""

    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:4], "big") % D


def tokens(text: str) -> list[str]:
    output: list[str] = []
    for word in re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*|-?\d+(?:\.\d+)?", text.casefold()):
        if re.fullmatch(r"-?\d+(?:\.\d+)?", word):
            output.append("NUMBER")
        else:
            output.append(CANON.get(word, word))
    return output


def features(text: str) -> torch.Tensor:
    canonical = tokens(text)
    vector = torch.zeros(D)
    for word in canonical:
        vector[_h("u:" + word)] += 1
    for left, right in zip(canonical, canonical[1:]):
        vector[_h("b:" + left + "_" + right)] += 1
    return torch.log1p(vector)


class SemanticBreadth(nn.Module):
    """Small neural proposal front-end; it has no state or action authority."""

    def __init__(self) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(D, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, EMB),
        )
        # The historical relation head is reused as an independent frame-kind
        # proposal head. Softmax is retained only for the legacy one-label API.
        self.head = nn.Linear(EMB, len(INTENTS))
        self.cardinality_head = nn.Linear(EMB, MAX_FRAMES + 1)
        self.checkpoint_meta: dict[str, Any] = {}

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = F.normalize(self.enc(inputs), dim=-1)
        return embedding, self.head(embedding)

    def forward_frames(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding, frame_logits = self.forward(inputs)
        return embedding, frame_logits, self.cardinality_head(embedding)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def _new_model(seed: int = 52001) -> SemanticBreadth:
    # Deterministic initialization matters when an older checkpoint lacks the
    # cardinality head. Such a checkpoint remains UNCALIBRATED and is never
    # silently treated as a new incumbent.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        return SemanticBreadth()


def _model_from_checkpoint(checkpoint: Mapping[str, Any]) -> SemanticBreadth:
    model = _new_model(int(checkpoint.get("initialization_seed", 52001)))
    incompatible = model.load_state_dict(checkpoint["state_dict"], strict=False)
    model.checkpoint_meta = {
        key: value for key, value in checkpoint.items() if key != "state_dict"
    }
    if incompatible.missing_keys:
        model.checkpoint_meta["missing_keys"] = list(incompatible.missing_keys)
        model.checkpoint_meta["score_status"] = "UNCALIBRATED"
    if incompatible.unexpected_keys:
        model.checkpoint_meta["unexpected_keys"] = list(incompatible.unexpected_keys)
    model.eval()
    return model


def load_bytes(checkpoint_bytes: bytes) -> SemanticBreadth:
    """Load exactly the supplied checkpoint bytes without reopening a path."""

    if not isinstance(checkpoint_bytes, bytes):
        raise TypeError("checkpoint_bytes must be bytes")
    checkpoint = torch.load(
        io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    return _model_from_checkpoint(checkpoint)


def load(path: str | Path) -> SemanticBreadth:
    return load_bytes(Path(path).read_bytes())


def _temperature(model: SemanticBreadth) -> float | None:
    value = model.checkpoint_meta.get("temperature")
    if (
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
    ):
        return float(value)
    return None


def predict(model: SemanticBreadth, text: str) -> dict[str, Any]:
    with torch.no_grad():
        embedding, logits = model(features(text)[None, :])
        temperature = _temperature(model)
        scaled = logits if temperature is None else logits / temperature
        probabilities = torch.softmax(scaled, 1)[0]
        index = int(probabilities.argmax())
    return {
        "intent": INTENTS[index],
        "score": float(probabilities[index]),
        "score_status": "CALIBRATED" if temperature is not None else "UNCALIBRATED",
        "temperature": temperature,
        "embedding": embedding[0].tolist(),
    }


def predict_frame_kinds(
    model: SemanticBreadth, text: str, threshold: float = 0.5
) -> dict[str, Any]:
    """Return proposals only. Symbolic validation decides what enters state."""

    with torch.no_grad():
        embedding, logits, cardinality_logits = model.forward_frames(
            features(text)[None, :]
        )
        temperature = _temperature(model)
        scaled = logits if temperature is None else logits / temperature
        probabilities = torch.sigmoid(scaled)[0]
        cardinality = int(cardinality_logits[0].argmax())

    # Quantization keeps public evaluation evidence stable across supported CPU
    # math backends without presenting these uncalibrated scores as precise.
    ranked = sorted(
        (
            (INTENTS[index], round(float(score), 6))
            for index, score in enumerate(probabilities)
        ),
        key=lambda item: (-item[1], item[0]),
    )
    selected = [item for item in ranked if item[1] >= threshold]
    if cardinality > 0:
        selected = ranked[: min(cardinality, MAX_FRAMES)]
    return {
        "proposals": [{"kind": kind, "score": score} for kind, score in selected],
        "cardinality": cardinality,
        "score_status": "CALIBRATED" if temperature is not None else "UNCALIBRATED",
        "temperature": temperature,
        "embedding": embedding[0].tolist(),
    }


def _canonical_target_frame(
    frame: Mapping[str, Any], line_number: int
) -> dict[str, Any]:
    """Project a reviewed frame to the exact semantics used for pair checks."""

    kind = frame.get("kind")
    if kind not in INTENTS:
        raise ValueError(f"lesson line {line_number} has a frame with an unknown kind")
    relation = frame.get("relation")
    slots = frame.get("slots")
    if not isinstance(relation, str) or not relation.strip():
        raise ValueError(
            f"lesson line {line_number} paraphrase frames require an exact relation"
        )
    if not isinstance(slots, Mapping) or not slots:
        raise ValueError(
            f"lesson line {line_number} paraphrase frames require typed slots"
        )
    canonical_slots: dict[str, Any] = {}
    for name, slot in slots.items():
        if not isinstance(name, str):
            raise ValueError(f"lesson line {line_number} has a non-string slot name")
        if not isinstance(slot, Mapping) or "value" not in slot:
            raise ValueError(
                f"lesson line {line_number} paraphrase frames require typed slots"
            )
        slot_type = slot.get("type")
        if not isinstance(slot_type, str) or not slot_type.strip():
            raise ValueError(
                f"lesson line {line_number} paraphrase frames require typed slots"
            )
        # This mirrors Frame.semantic_key(): source spelling and provenance are
        # separate, while both the declared type and exact value are semantic.
        canonical_slots[name] = {
            "type": slot_type.strip(),
            "value": slot["value"],
        }
    target = {
        "kind": kind,
        "relation": relation.strip(),
        "slots": canonical_slots,
    }
    try:
        json.dumps(target, ensure_ascii=False, allow_nan=False, sort_keys=True)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"lesson line {line_number} has a non-JSON frame target"
        ) from error
    return target


def _target_hash(target_frames: list[dict[str, Any]]) -> str:
    ordered = sorted(
        target_frames,
        key=lambda item: json.dumps(
            item,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
    )
    payload = json.dumps(
        ordered,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _frozen_evaluation_exclusions() -> tuple[set[str], list[dict[str, Any]]]:
    """Load every frozen public/calibration surface as a training exclusion."""

    texts: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    references: dict[Path, str] = {}
    manifest, manifest_record = _pinned_evaluation_manifest()
    for role, reference in manifest["artifacts"].items():
        if not isinstance(reference, Mapping) or not isinstance(
            reference.get("path"), str
        ):
            continue
        target = _manifest_target(reference, role=str(role))
        expected = str(reference.get("sha256", ""))
        if not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError("invalid artifact hash in pinned manifest-v6")
        prior = references.get(target)
        if prior is not None and prior != expected:
            raise ValueError(f"conflicting frozen hashes for {target}")
        references[target] = expected
    artifacts.append(
        {
            "path": str(PINNED_EVALUATION_MANIFEST.resolve()),
            "sha256": manifest_record["sha256"],
            "role": "FROZEN_EVALUATION_MANIFEST",
        }
    )

    def read_declared(path: Path) -> bytes:
        expected = references.get(path.resolve())
        if expected is None:
            raise ValueError(
                f"training exclusion is not hash-pinned by a manifest: {path}"
            )
        content = path.read_bytes()
        actual = _sha256_bytes(content)
        if actual != expected:
            raise ValueError(
                f"training exclusion SHA-256 mismatch for {path}: expected {expected}, got {actual}"
            )
        return content

    read_declared(PINNED_HELD_OUT_VOCABULARY)

    def declared_matching(pattern: str) -> set[Path]:
        return {
            path
            for path in references
            if path.parent == FROZEN_EVAL_DIRECTORY.resolve() and path.match(pattern)
        }

    def closed_inventory(pattern: str, label: str) -> list[Path]:
        declared = declared_matching(pattern)
        discovered = {
            path.resolve()
            for path in FROZEN_EVAL_DIRECTORY.glob(pattern)
            if path.is_file()
        }
        if discovered != declared:
            unpinned = sorted(str(path) for path in discovered - declared)
            missing = sorted(str(path) for path in declared - discovered)
            raise ValueError(
                f"frozen {label} inventory mismatch: "
                f"unpinned={unpinned}, missing={missing}"
            )
        return sorted(declared, key=lambda path: path.as_posix())

    suites = closed_inventory("public-audit-v*-260.json", "promotion-suite")
    calibration_files = closed_inventory("calibration-fit-*.jsonl", "calibration")
    known_failure_files = closed_inventory("*known-failures*.json", "known-failure")
    if not suites:
        raise FileNotFoundError("pinned manifest declares no public evaluation suite")
    for path in suites:
        content = read_declared(path)
        value = json.loads(content.decode("utf-8"))
        if value.get("frozen") is not True or not isinstance(
            value.get("splits"), Mapping
        ):
            raise ValueError(f"invalid frozen evaluation suite: {path}")
        for rows in value["splits"].values():
            if not isinstance(rows, list):
                raise ValueError(f"invalid split in frozen evaluation suite: {path}")
            for row in rows:
                if isinstance(row, Mapping):
                    text = str(row.get("input", row.get("text", ""))).strip().casefold()
                    if text:
                        texts.add(text)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_bytes(content),
                "role": "FROZEN_PROMOTION_SUITE_TRAINING_EXCLUSION",
            }
        )
    for path in calibration_files:
        content = read_declared(path)
        for line in content.decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            text = str(row.get("text", row.get("input", ""))).strip().casefold()
            if text:
                texts.add(text)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_bytes(content),
                "role": "CALIBRATION_FIT_TRAINING_EXCLUSION",
            }
        )
    for path in known_failure_files:
        content = read_declared(path)
        value = json.loads(content.decode("utf-8"))
        if value.get("frozen") is not True or not isinstance(value.get("cases"), list):
            raise ValueError(f"invalid frozen known-failure audit: {path}")
        for row in value["cases"]:
            if not isinstance(row, Mapping):
                continue
            primary = str(row.get("input", "")).strip().casefold()
            if primary:
                texts.add(primary)
            reverse = row.get("reverse_control")
            if isinstance(reverse, Mapping):
                control = str(reverse.get("input", "")).strip().casefold()
                if control:
                    texts.add(control)
        artifacts.append(
            {
                "path": str(path.resolve()),
                "sha256": _sha256_bytes(content),
                "role": "DEVELOPMENT_INFLUENCED_AUDIT_TRAINING_EXCLUSION",
            }
        )
    return texts, artifacts


def _reviewed_rows_from_text(
    lesson_text: str,
    held_out_terms: Iterable[str] = (),
    forbidden_texts: Iterable[str] = (),
) -> list[dict[str, Any]]:
    held_out = {term.casefold().strip() for term in held_out_terms if term.strip()}
    forbidden = {text.casefold().strip() for text in forbidden_texts if text.strip()}
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(lesson_text.splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        status = row.get("status", row.get("review_status"))
        if status != "REVIEWED":
            raise ValueError(f"lesson line {line_number} is not explicitly REVIEWED")
        reviewed_by_value = row.get("reviewed_by")
        reviewed_at_value = row.get("reviewed_at")
        if not isinstance(reviewed_by_value, str) or not reviewed_by_value.strip():
            raise ValueError(f"lesson line {line_number} lacks review provenance")
        if not isinstance(reviewed_at_value, str) or not reviewed_at_value.strip():
            raise ValueError(f"lesson line {line_number} lacks review provenance")
        reviewed_by = reviewed_by_value.strip()
        reviewed_at = reviewed_at_value.strip()
        try:
            parsed_reviewed_at = datetime.fromisoformat(
                reviewed_at[:-1] + "+00:00"
                if reviewed_at.endswith("Z")
                else reviewed_at
            )
        except ValueError as exc:
            raise ValueError(
                f"lesson line {line_number} reviewed_at must be valid ISO-8601 UTC"
            ) from exc
        if (
            parsed_reviewed_at.tzinfo is None
            or parsed_reviewed_at.utcoffset() != timedelta(0)
        ):
            raise ValueError(
                f"lesson line {line_number} reviewed_at must be valid ISO-8601 UTC"
            )
        permission_value = row.get("permission")
        if not isinstance(permission_value, str) or not permission_value.strip():
            raise ValueError(f"lesson line {line_number} lacks permission provenance")
        permission = permission_value.strip()
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError(f"lesson line {line_number} has empty text")
        if text.casefold() in forbidden:
            raise ValueError(
                f"lesson line {line_number} overlaps a frozen evaluation, calibration, "
                "or known-failure audit item"
            )
        surface = set(re.findall(r"[a-zA-Z][a-zA-Z0-9_-]*", text.casefold()))
        overlap = sorted(surface & held_out)
        if overlap:
            raise ValueError(
                f"lesson line {line_number} contains held-out vocabulary: {','.join(overlap)}"
            )
        supplied_frames = row.get("frames")
        target_frames: list[dict[str, Any]] = []
        if supplied_frames is not None:
            if not isinstance(supplied_frames, list) or not supplied_frames:
                raise ValueError(
                    f"lesson line {line_number} frames must be a non-empty list"
                )
            for frame in supplied_frames:
                if not isinstance(frame, Mapping):
                    raise ValueError(
                        f"lesson line {line_number} has a non-object frame"
                    )
                target_frames.append(_canonical_target_frame(frame, line_number))

        raw_kinds = row.get("frame_kinds")
        if raw_kinds is None:
            if target_frames:
                raw_kinds = [frame["kind"] for frame in target_frames]
            elif row.get("intent"):
                raw_kinds = [row["intent"]]
        if not isinstance(raw_kinds, list):
            raise ValueError(f"lesson line {line_number} frame_kinds must be a list")
        if not raw_kinds or any(kind not in INTENTS for kind in raw_kinds):
            raise ValueError(
                f"lesson line {line_number} has unknown or empty frame kinds"
            )
        if len(raw_kinds) > MAX_FRAMES:
            raise ValueError(
                f"lesson line {line_number} exceeds maximum frame cardinality"
            )
        if target_frames and set(raw_kinds) != {
            frame["kind"] for frame in target_frames
        }:
            raise ValueError(
                f"lesson line {line_number} frame_kinds disagree with frames"
            )

        kinds = list(dict.fromkeys(raw_kinds))
        cardinality = len(target_frames) if target_frames else len(raw_kinds)
        if cardinality < 1 or cardinality > MAX_FRAMES:
            raise ValueError(f"lesson line {line_number} has invalid frame cardinality")
        semantic_targets = target_frames or [{"kind": kind} for kind in raw_kinds]
        group = row.get("paraphrase_group")
        if group is not None and (not isinstance(group, str) or not group.strip()):
            raise ValueError(
                f"lesson line {line_number} has an invalid paraphrase_group"
            )
        row = dict(row)
        row["reviewed_by"] = reviewed_by
        row["reviewed_at"] = reviewed_at
        row["permission"] = permission
        row["text"] = text
        row["frame_kinds"] = kinds
        row["frame_cardinality"] = cardinality
        row["semantic_target_sha256"] = _target_hash(semantic_targets)
        row["_structured_frame_targets"] = target_frames
        row["paraphrase_group"] = group.strip() if isinstance(group, str) else None
        rows.append(row)
    if not rows:
        raise ValueError("no reviewed lessons")

    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = row["paraphrase_group"]
        if group is not None:
            groups.setdefault(group, []).append(row)
    target_groups: dict[str, str] = {}
    for group, members in sorted(groups.items()):
        if len(members) < 2:
            raise ValueError(
                f"paraphrase_group {group!r} requires at least two reviewed rows"
            )
        if any(not member["_structured_frame_targets"] for member in members):
            raise ValueError(
                f"paraphrase_group {group!r} requires exact relation and slot targets"
            )
        signatures = {member["semantic_target_sha256"] for member in members}
        if len(signatures) != 1:
            raise ValueError(
                f"paraphrase_group {group!r} does not map to one exact frame set"
            )
        signature = next(iter(signatures))
        prior_group = target_groups.get(signature)
        if prior_group is not None:
            raise ValueError(
                "one exact semantic target cannot appear under multiple "
                f"paraphrase_group IDs: {prior_group!r}, {group!r}"
            )
        target_groups[signature] = group
    target_occurrences: dict[str, list[str | None]] = {}
    for row in rows:
        target_occurrences.setdefault(row["semantic_target_sha256"], []).append(
            row["paraphrase_group"]
        )
    for assignments in target_occurrences.values():
        if len(assignments) > 1 and any(group is None for group in assignments):
            raise ValueError(
                "a repeated exact semantic target must map to exactly one non-null "
                "paraphrase_group; ungrouped duplicates would become contrastive negatives"
            )
    group_indexes = {name: index for index, name in enumerate(sorted(groups))}
    for row in rows:
        group = row["paraphrase_group"]
        row["_contrastive_group_index"] = group_indexes.get(group, -1)
    return rows


def _reviewed_rows(
    path: str | Path,
    held_out_terms: Iterable[str] = (),
    forbidden_texts: Iterable[str] = (),
) -> list[dict[str, Any]]:
    return _reviewed_rows_from_text(
        Path(path).read_text(encoding="utf-8"), held_out_terms, forbidden_texts
    )


def _paraphrase_group_manifest(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group = row.get("paraphrase_group")
        if isinstance(group, str):
            groups.setdefault(group, []).append(row)
    result: list[dict[str, Any]] = []
    for group, members in sorted(groups.items()):
        lesson_ids = sorted(str(member.get("id", "")) for member in members)
        target_sha256 = members[0]["semantic_target_sha256"]
        body = {
            "group": group,
            "lesson_ids": lesson_ids,
            "rows": len(members),
            "semantic_target_sha256": target_sha256,
        }
        body["group_sha256"] = hashlib.sha256(
            json.dumps(
                body,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        result.append(body)
    return result


def _targets(
    rows: list[dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    frame_targets = torch.zeros((len(rows), len(INTENTS)), dtype=torch.float32)
    cardinalities = torch.zeros(len(rows), dtype=torch.long)
    contrastive_groups = torch.full((len(rows),), -1, dtype=torch.long)
    for row_index, row in enumerate(rows):
        kinds = row["frame_kinds"]
        for kind in kinds:
            frame_targets[row_index, INTENTS.index(kind)] = 1.0
        cardinalities[row_index] = row["frame_cardinality"]
        contrastive_groups[row_index] = row["_contrastive_group_index"]
    return frame_targets, cardinalities, contrastive_groups


def supervised_contrastive_loss(
    embeddings: torch.Tensor, contrastive_groups: torch.Tensor
) -> torch.Tensor:
    """Pull only explicitly reviewed paraphrases together.

    Ungrouped rows, even when their frame-kind vectors match, are never
    positive pairs. This prevents different values, polarities, or relations
    from being collapsed merely because they share a broad kind.
    """

    count = embeddings.shape[0]
    if count < 2:
        return embeddings.sum() * 0.0
    if contrastive_groups.shape != (count,):
        raise ValueError(
            "contrastive_groups must contain one group index per embedding"
        )
    contrastive_groups = contrastive_groups.to(embeddings.device)
    similarities = embeddings @ embeddings.T
    diagonal = torch.eye(count, dtype=torch.bool, device=embeddings.device)
    assigned = contrastive_groups >= 0
    same = (
        (contrastive_groups[:, None] == contrastive_groups[None, :])
        & assigned[:, None]
        & assigned[None, :]
        & ~diagonal
    )
    different = ~same & ~diagonal
    losses: list[torch.Tensor] = []
    if same.any():
        losses.append((1.0 - similarities[same]).mean())
    if different.any():
        losses.append(F.relu(similarities[different] - 0.20).mean())
    return torch.stack(losses).mean() if losses else embeddings.sum() * 0.0


def freeze_encoder_train_projection_and_heads(model: SemanticBreadth) -> dict[str, int]:
    """Freeze the feature encoder; train only projection and proposal heads."""

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    projection = model.enc[-1]
    if not isinstance(projection, nn.Linear):
        raise TypeError("semantic encoder must end in a linear projection")
    for module in (projection, model.head, model.cardinality_head):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return {
        "total": sum(parameter.numel() for parameter in model.parameters()),
        "trainable": sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "frozen": sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        ),
    }


def train(
    parent_path: str | Path,
    out_dir: str | Path,
    steps: int = 800,
    seed: int = 51401,
    extra_jsonl: str | Path | None = None,
    held_out_vocabulary: Iterable[str] = (),
) -> dict[str, Any]:
    """Train an immutable challenger exclusively from reviewed lessons."""

    if not extra_jsonl:
        raise RuntimeError("reviewed JSONL lessons are required")
    if steps <= 0:
        raise ValueError("steps must be positive")

    parent = Path(parent_path).resolve()
    lessons = Path(extra_jsonl).resolve()
    output_directory = Path(out_dir).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_directory / "semantic-breadth.pt"
    receipt_path = output_directory / "training-receipt.json"
    if checkpoint_path.exists() or receipt_path.exists():
        raise FileExistsError("challenger output is immutable and already exists")

    (
        required_held_out,
        required_held_out_artifacts,
        required_held_out_artifacts_sha256,
    ) = required_held_out_vocabulary()
    required_held_out_terms = sorted(required_held_out)
    required_held_out_terms_sha256 = _canonical_json_sha256(required_held_out_terms)
    pinned_relative = PINNED_HELD_OUT_VOCABULARY.relative_to(REPOSITORY_ROOT).as_posix()
    pinned_held_out_file_hash = next(
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
    lesson_bytes = lessons.read_bytes()
    lesson_text = lesson_bytes.decode("utf-8")
    rows = _reviewed_rows_from_text(lesson_text, held_out, forbidden_texts)
    parent_bytes = parent.read_bytes()
    parent_hash = _sha256_bytes(parent_bytes)
    lessons_hash = _sha256_bytes(lesson_bytes)
    trainer_source_hash = _sha256_bytes(Path(__file__).read_bytes())

    torch.manual_seed(seed)
    torch.set_num_threads(2)
    checkpoint = torch.load(
        io.BytesIO(parent_bytes), map_location="cpu", weights_only=True
    )
    if not isinstance(checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping")
    model = _model_from_checkpoint(checkpoint)
    model.train()
    parameter_scope = freeze_encoder_train_projection_and_heads(model)
    inputs = torch.stack([features(row["text"]) for row in rows])
    frame_targets, cardinality_targets, contrastive_groups = _targets(rows)
    paraphrase_groups = _paraphrase_group_manifest(rows)
    paraphrase_groups_sha256 = hashlib.sha256(
        json.dumps(
            paraphrase_groups,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
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
    training_exclusions_sha256 = _canonical_json_sha256(
        sorted(
            (
                {"role": artifact["role"], "sha256": artifact["sha256"]}
                for artifact in exclusion_artifacts
            ),
            key=lambda item: (item["role"], item["sha256"]),
        )
    )
    training_inputs = {
        "parent_sha256": parent_hash,
        "lessons_sha256": lessons_hash,
        "held_out_vocabulary_sha256": held_out_hash,
        "pinned_held_out_vocabulary_file_sha256": pinned_held_out_file_hash,
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "training_exclusions_sha256": training_exclusions_sha256,
        "paraphrase_groups_sha256": paraphrase_groups_sha256,
        "trainer_source_sha256": trainer_source_hash,
        "seed": seed,
        "steps": steps,
        "objective": objective,
        "optimizer": optimizer_config,
        "torch_num_threads": 2,
    }
    training_inputs_sha256 = _canonical_json_sha256(training_inputs)
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    history: list[dict[str, float | int]] = []

    for step in range(steps):
        embeddings, frame_logits, cardinality_logits = model.forward_frames(inputs)
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
        nn.utils.clip_grad_norm_(model.parameters(), gradient_norm_clip)
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

    checkpoint = {
        "schema": "kev.semantic-frames.v052",
        "state_dict": model.state_dict(),
        "parameters": model.parameter_count,
        "seed": seed,
        "intents": INTENTS,
        "max_frames": MAX_FRAMES,
        "parent_sha256": parent_hash,
        "lessons_sha256": lessons_hash,
        "held_out_vocabulary_sha256": held_out_hash,
        "pinned_held_out_vocabulary_file_sha256": pinned_held_out_file_hash,
        "required_held_out_vocabulary_artifacts": required_held_out_artifacts,
        "required_held_out_vocabulary_artifacts_sha256": (
            required_held_out_artifacts_sha256
        ),
        "required_held_out_vocabulary_terms_sha256": (required_held_out_terms_sha256),
        "training_exclusions_sha256": training_exclusions_sha256,
        "paraphrase_groups_sha256": paraphrase_groups_sha256,
        "contrastive_positive_pairs": contrastive_positive_pairs,
        "trainer_source_sha256": trainer_source_hash,
        "training_inputs_sha256": training_inputs_sha256,
        "training_scope": "projection_and_heads_only",
        "parameter_scope": parameter_scope,
        "steps": steps,
        "objective": objective,
        "optimizer": optimizer_config,
        "torch_num_threads": 2,
        "temperature": None,
        "score_status": "UNCALIBRATED",
        "artifact_status": "CHALLENGER",
        "initialization_seed": model.checkpoint_meta.get("initialization_seed", 52001),
    }
    with checkpoint_path.open("xb") as handle:
        torch.save(checkpoint, handle)
    checkpoint_hash = sha256_file(checkpoint_path)
    receipt = {
        "schema": "kev.training-receipt.v1",
        "artifact_status": "CHALLENGER",
        "activation": "NONE",
        "parent_path": str(parent),
        "parent_sha256": parent_hash,
        "lessons_path": str(lessons),
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
        "pinned_held_out_vocabulary_file_sha256": pinned_held_out_file_hash,
        "training_exclusion_artifacts": exclusion_artifacts,
        "training_exclusions_sha256": training_exclusions_sha256,
        "paraphrase_groups": paraphrase_groups,
        "paraphrase_groups_sha256": paraphrase_groups_sha256,
        "contrastive_positive_pairs": contrastive_positive_pairs,
        "trainer_source_sha256": trainer_source_hash,
        "training_inputs": training_inputs,
        "training_inputs_sha256": training_inputs_sha256,
        "training_scope": "projection_and_heads_only",
        "parameter_scope": parameter_scope,
        "challenger_path": str(checkpoint_path),
        "challenger_sha256": checkpoint_hash,
        "seed": seed,
        "steps": steps,
        "examples": len(rows),
        "objective": checkpoint["objective"],
        "optimizer": optimizer_config,
        "torch_num_threads": 2,
        "loss_history": history,
        "promotion_features": [],
        "score_status": "UNCALIBRATED",
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
        },
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return receipt

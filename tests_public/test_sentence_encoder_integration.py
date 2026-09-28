from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import socket

import pytest
import torch

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.calibration import calibrate_checkpoint
from kev.encoders import (
    FROZEN_SENTENCE_CHECKPOINT_SCHEMA,
    load_supported_local_encoder,
)
from kev.model_runtime import CheckpointPredictor
from kev import sentence_training
from kev.sentence_training import train_frozen_encoder_challenger
from kev.uc51a3.alive import AliveRuntime, AliveStore


ROOT = Path(__file__).resolve().parents[1]


def test_sentence_training_source_manifest_binds_shared_semantic_code() -> None:
    manifest = sentence_training._verified_training_source_manifest()
    by_role = {entry["role"]: entry for entry in manifest}

    assert set(by_role) == {
        "FROZEN_ENCODER_TRAINER",
        "ENCODER_RUNTIME",
        "SHARED_SEMANTIC_TRAINING",
    }
    semantic_source = ROOT / "kev" / "uc51a2" / "semantic_breadth.py"
    assert by_role["SHARED_SEMANTIC_TRAINING"]["path"] == (
        "kev/uc51a2/semantic_breadth.py"
    )
    assert by_role["SHARED_SEMANTIC_TRAINING"]["sha256"] == file_sha256(semantic_source)
    assert len(canonical_json_sha256(manifest)) == 64


def test_sentence_training_source_manifest_rejects_a_source_swap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "shared.py"
    source.write_text("VERSION = 1\n", encoding="utf-8")
    role = "SHARED_SEMANTIC_TRAINING"
    monkeypatch.setattr(sentence_training, "TRAINING_SOURCE_PATHS", ((role, source),))
    monkeypatch.setattr(
        sentence_training,
        "_LOADED_TRAINING_SOURCE_HASHES",
        {role: file_sha256(source)},
    )
    expected = sentence_training._verified_training_source_manifest()

    source.write_text("VERSION = 2\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="changed after module import"):
        sentence_training._verified_training_source_manifest(expected)


def test_selected_acquisition_record_pins_replayable_manifest() -> None:
    source_path = (
        ROOT
        / "models"
        / "encoder-sources"
        / "all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json"
    )
    source = json.loads(source_path.read_text(encoding="utf-8"))

    assert file_sha256(source_path) == (
        "a46078dd1c13d9114bff07a891a1a903c1281681cf6dd3002e8d9e5ea8da03e9"
    )
    assert source["source"]["revision"] == ("1110a243fdf4706b3f48f1d95db1a4f5529b4d41")
    assert source["license"]["identifier"] == "Apache-2.0"
    assert source["expected_local_manifest_sha256"] == (
        "cfd8f8bc3413f31ab927ad7d78eb380c3359fdb2893061075d2369dfec81ea57"
    )
    assert next(
        record for record in source["files"] if record["path"] == "model.safetensors"
    )["sha256"] == ("53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db")
    assert "REVIEWER" not in source["replay"]["acquire_command"]
    assert "ISO_8601" not in source["replay"]["acquire_command"]


@pytest.fixture(scope="module")
def tiny_encoder_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tokenizers = pytest.importorskip("tokenizers")
    safetensors = pytest.importorskip("safetensors.torch")
    transformers = pytest.importorskip("transformers")
    directory = tmp_path_factory.mktemp("tiny-local-encoder")

    configuration = transformers.BertConfig(
        vocab_size=9,
        hidden_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        intermediate_size=16,
        max_position_embeddings=32,
        pad_token_id=0,
    )
    configuration.architectures = ["BertModel"]
    transformer = transformers.BertModel(configuration)
    files: dict[str, bytes] = {
        "config.json": (
            json.dumps(configuration.to_dict(), sort_keys=True) + "\n"
        ).encode(),
        "model.safetensors": safetensors.save(transformer.state_dict()),
        "1_Pooling/config.json": json.dumps(
            {
                "word_embedding_dimension": 8,
                "pooling_mode_cls_token": False,
                "pooling_mode_mean_tokens": True,
                "pooling_mode_max_tokens": False,
                "pooling_mode_mean_sqrt_len_tokens": False,
            },
            sort_keys=True,
        ).encode(),
        "modules.json": json.dumps(
            [
                {
                    "idx": 0,
                    "name": "0",
                    "path": "",
                    "type": "sentence_transformers.models.Transformer",
                },
                {
                    "idx": 1,
                    "name": "1",
                    "path": "1_Pooling",
                    "type": "sentence_transformers.models.Pooling",
                },
                {
                    "idx": 2,
                    "name": "2",
                    "path": "2_Normalize",
                    "type": "sentence_transformers.models.Normalize",
                },
            ],
            sort_keys=True,
        ).encode(),
    }
    vocabulary = {
        "[PAD]": 0,
        "[UNK]": 1,
        "[CLS]": 2,
        "[SEP]": 3,
        "[MASK]": 4,
        "reduce": 5,
        "latency": 6,
        "never": 7,
        "production": 8,
    }
    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece(vocabulary))
    tokenizer.normalizer = tokenizers.normalizers.BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    files["tokenizer.json"] = tokenizer.to_str().encode()

    records: list[dict[str, str]] = []
    for relative, payload in files.items():
        destination = directory / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)
        records.append(
            {"path": relative, "sha256": hashlib.sha256(payload).hexdigest()}
        )
    manifest = {
        "schema": "kev.local-encoder.v1",
        "name": "tiny-offline-test-encoder",
        "embedding_dim": 8,
        "network_policy": "LOCAL_ONLY",
        "source": {
            "provider": "test-fixture",
            "repository": "local/tiny-bert",
            "revision": "0" * 40,
            "url": "https://example.invalid/local-test-fixture",
        },
        "permission": {
            "status": "APPROVED",
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-09-28T00:00:00Z",
            "scope": "offline integration test only",
        },
        "license": {"identifier": "TEST-ONLY"},
        "loader": {
            "id": "hf-bert-mean-pooling-normalize-v1",
            "config_path": "config.json",
            "model_path": "model.safetensors",
            "tokenizer_path": "tokenizer.json",
            "pooling_config_path": "1_Pooling/config.json",
            "modules_path": "modules.json",
            "max_length": 16,
            "pooling": "MEAN",
            "normalize": True,
            "dynamic_remote_code": False,
            "safetensors_only": True,
        },
        "files": records,
    }
    manifest_path = directory / "encoder-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def test_supported_runtime_is_offline_and_encoder_stays_frozen(
    tiny_encoder_bundle: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_network(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("runtime loader attempted network access")

    monkeypatch.setattr(socket, "create_connection", forbidden_network)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    model = load_supported_local_encoder(tiny_encoder_bundle)
    features = model.encode_texts(["reduce latency", "never production"])

    assert features.shape == (2, 8)
    assert torch.allclose(features.norm(dim=1), torch.ones(2), atol=1e-6)
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert all(
        parameter.requires_grad
        for module in (model.projection, model.frame_head, model.cardinality_head)
        for parameter in module.parameters()
    )


def _public_parent() -> Path:
    registry = json.loads((ROOT / "models" / "registry.json").read_text())
    return ROOT / registry["active"]["path"]


def _reviewed_lessons(path: Path) -> None:
    rows = [
        {
            "id": "tiny-lesson-1",
            "status": "REVIEWED",
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-09-28T00:00:00Z",
            "permission": "test-authored",
            "text": "Please keep the staging switch disabled",
            "frame_kinds": ["CONSTRAINT"],
        },
        {
            "id": "tiny-lesson-2",
            "status": "REVIEWED",
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-09-28T00:00:00Z",
            "permission": "test-authored",
            "text": "Aim to reduce the service delay",
            "frame_kinds": ["GOAL"],
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"reviewed_by": "   "}, "lacks review provenance"),
        ({"permission": "   "}, "lacks permission provenance"),
        ({"reviewed_at": "not-a-timestamp"}, "valid ISO-8601 UTC"),
        (
            {"reviewed_at": "2026-09-28T00:00:00-04:00"},
            "valid ISO-8601 UTC",
        ),
    ],
)
def test_direct_sentence_training_rejects_invalid_review_provenance(
    tiny_encoder_bundle: Path,
    tmp_path: Path,
    updates: dict[str, object],
    match: str,
) -> None:
    lessons = tmp_path / "invalid-provenance.jsonl"
    row: dict[str, object] = {
        "id": "invalid-provenance",
        "status": "REVIEWED",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-09-28T00:00:00Z",
        "permission": "test-authored",
        "text": "Keep the result stable",
        "frame_kinds": ["CONSTRAINT"],
    }
    row.update(updates)
    lessons.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "candidate"

    with pytest.raises(ValueError, match=match):
        train_frozen_encoder_challenger(
            _public_parent(),
            tiny_encoder_bundle,
            output,
            steps=1,
            extra_jsonl=lessons,
        )

    assert not (output / "semantic-breadth.pt").exists()
    assert not (output / "training-receipt.json").exists()
    assert not (output / "frozen-embeddings.pt").exists()


def test_direct_sentence_training_rejects_v2_held_out_surface_forms(
    tiny_encoder_bundle: Path, tmp_path: Path
) -> None:
    lessons = tmp_path / "held-out-v2.jsonl"
    row = {
        "id": "held-out-v2",
        "status": "REVIEWED",
        "reviewed_by": "test-reviewer",
        "reviewed_at": "2026-09-28T00:00:00Z",
        "permission": "test-authored",
        "text": "Keep the result attested",
        "frame_kinds": ["CONSTRAINT"],
    }
    lessons.write_text(json.dumps(row) + "\n", encoding="utf-8")
    output = tmp_path / "candidate"

    with pytest.raises(ValueError, match="held-out vocabulary: attested"):
        train_frozen_encoder_challenger(
            _public_parent(),
            tiny_encoder_bundle,
            output,
            steps=1,
            extra_jsonl=lessons,
        )

    assert not (output / "semantic-breadth.pt").exists()
    assert not (output / "training-receipt.json").exists()
    assert not (output / "frozen-embeddings.pt").exists()


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("intents", ["PREDICTION"], "intent order"),
        ("max_frames", 5, "maximum frame cardinality"),
    ],
)
def test_parent_head_copy_rejects_incompatible_semantic_metadata_before_copy(
    tiny_encoder_bundle: Path,
    field: str,
    value: object,
    match: str,
) -> None:
    parent = torch.load(_public_parent(), map_location="cpu", weights_only=True)
    parent[field] = value
    model = load_supported_local_encoder(tiny_encoder_bundle)
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.proposal_state_dict().items()
    }

    with pytest.raises(ValueError, match=match):
        sentence_training._copy_parent_heads(
            model,
            parent,
            file_sha256(tiny_encoder_bundle),
        )

    after = model.proposal_state_dict()
    assert all(torch.equal(before[name], after[name]) for name in before)


def test_sentence_training_rejects_manifest_swap_before_artifact_writes(
    tiny_encoder_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied = tmp_path / "encoder"
    shutil.copytree(tiny_encoder_bundle.parent, copied)
    manifest_path = copied / tiny_encoder_bundle.name
    lessons = tmp_path / "reviewed.jsonl"
    _reviewed_lessons(lessons)
    output = tmp_path / "candidate"
    real_verify = sentence_training.verify_local_encoder_manifest
    swapped = False

    def verify_then_swap(path: str | Path) -> dict[str, object]:
        nonlocal swapped
        verified = real_verify(path)
        if not swapped:
            value = json.loads(manifest_path.read_text(encoding="utf-8"))
            value["name"] = "swapped-after-initial-verification"
            manifest_path.write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            swapped = True
        return verified

    monkeypatch.setattr(
        sentence_training, "verify_local_encoder_manifest", verify_then_swap
    )
    with pytest.raises(ValueError, match="changed between verification and loading"):
        train_frozen_encoder_challenger(
            _public_parent(),
            manifest_path,
            output,
            steps=1,
            extra_jsonl=lessons,
        )

    assert not (output / "semantic-breadth.pt").exists()
    assert not (output / "training-receipt.json").exists()
    assert not (output / "frozen-embeddings.pt").exists()


def test_sentence_training_rejects_loaded_manifest_identity_mismatch(
    tiny_encoder_bundle: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lessons = tmp_path / "reviewed.jsonl"
    _reviewed_lessons(lessons)
    output = tmp_path / "candidate"
    real_load = sentence_training.load_supported_local_encoder

    def mismatched_loader(
        path: str | Path, *, expected_manifest_sha256: str | None = None
    ):
        model = real_load(path, expected_manifest_sha256=expected_manifest_sha256)
        model.artifact_metadata["manifest_sha256"] = "0" * 64
        return model

    monkeypatch.setattr(
        sentence_training, "load_supported_local_encoder", mismatched_loader
    )
    with pytest.raises(RuntimeError, match="identity differs"):
        train_frozen_encoder_challenger(
            _public_parent(),
            tiny_encoder_bundle,
            output,
            steps=1,
            extra_jsonl=lessons,
        )

    assert not (output / "semantic-breadth.pt").exists()
    assert not (output / "training-receipt.json").exists()
    assert not (output / "frozen-embeddings.pt").exists()


def test_small_checkpoint_trains_loads_calibrates_and_runs_alive(
    tiny_encoder_bundle: Path, tmp_path: Path
) -> None:
    lessons = tmp_path / "reviewed.jsonl"
    _reviewed_lessons(lessons)
    receipt = train_frozen_encoder_challenger(
        _public_parent(),
        tiny_encoder_bundle,
        tmp_path / "trained",
        steps=1,
        seed=52022,
        extra_jsonl=lessons,
        encoder_batch_size=2,
    )
    checkpoint_path = Path(receipt["challenger_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["schema"] == FROZEN_SENTENCE_CHECKPOINT_SCHEMA
    assert set(checkpoint["proposal_state_dict"]) == {
        "projection.weight",
        "projection.bias",
        "frame_head.weight",
        "frame_head.bias",
        "cardinality_head.weight",
        "cardinality_head.bias",
    }
    assert checkpoint["encoder_manifest"]["sha256"] == file_sha256(tiny_encoder_bundle)
    assert receipt["architecture_transition"]["copied_components"] == [
        "head",
        "cardinality_head",
    ]
    assert receipt["promotion_features"] == []
    assert receipt["frozen_embedding_cache"]["values_sha256"]
    assert [
        Path(artifact["path"]).name
        for artifact in receipt["required_held_out_vocabulary_artifacts"]
    ] == ["held-out-vocabulary-v1.txt", "held-out-vocabulary-v2.txt"]
    assert (
        checkpoint["required_held_out_vocabulary_artifacts"]
        == receipt["required_held_out_vocabulary_artifacts"]
    )
    assert checkpoint["trainer_source_sha256"] == receipt["trainer_source_sha256"]
    assert (
        checkpoint["encoder_runtime_source_sha256"]
        == (receipt["encoder_runtime_source_sha256"])
    )
    semantic_source_sha256 = file_sha256(
        ROOT / "kev" / "uc51a2" / "semantic_breadth.py"
    )
    assert checkpoint["training_contract_source_sha256"] == semantic_source_sha256
    assert receipt["training_contract_source_sha256"] == semantic_source_sha256
    assert receipt["training_inputs"]["training_contract_source_sha256"] == (
        semantic_source_sha256
    )
    assert checkpoint["training_source_manifest"] == receipt["training_source_manifest"]
    assert (
        checkpoint["training_source_manifest"]
        == receipt["training_inputs"]["training_source_manifest"]
    )
    assert checkpoint["training_source_manifest_sha256"] == canonical_json_sha256(
        checkpoint["training_source_manifest"]
    )
    assert (
        checkpoint["training_source_manifest_sha256"]
        == receipt["training_source_manifest_sha256"]
    )
    assert (
        checkpoint["training_source_manifest_sha256"]
        == receipt["training_inputs"]["training_source_manifest_sha256"]
    )

    predictor = CheckpointPredictor(checkpoint_path)
    prediction = predictor(
        {"id": "probe", "input": "reduce latency", "projection": "kinds"}
    )
    assert prediction["model_sha256"] == file_sha256(checkpoint_path)
    assert prediction["score_status"] == "UNCALIBRATED"

    state_dir = tmp_path / "state"
    store = AliveStore(state_dir)
    incumbent_state = store.read()
    store.record_model_decision(
        "MODEL_QUALIFIED",
        {"run_id": "synthetic-runtime-test"},
        active_model={
            "path": str(checkpoint_path),
            "sha256": file_sha256(checkpoint_path),
        },
        expected_incumbent={
            "path": incumbent_state["semantic_model"],
            "sha256": incumbent_state["semantic_model_sha256"],
            "generation": incumbent_state["semantic_model_generation"],
        },
    )
    runtime = AliveRuntime(state_dir)
    assert runtime.model_error is None
    assert runtime._proposal("reduce latency")["model_sha256"] == file_sha256(
        checkpoint_path
    )

    calibration = tmp_path / "calibration.jsonl"
    calibration.write_text(
        json.dumps(
            {
                "id": "fit-1",
                "split": "calibration_fit",
                "text": "never production",
                "frame_kinds": ["CONSTRAINT"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    promotion = tmp_path / "promotion.json"
    promotion.write_text(
        json.dumps(
            {
                "schema": "kev.eval-suite.v1",
                "id": "tiny-promotion",
                "frozen": True,
                "splits": {
                    "fresh": [
                        {"id": "fresh-1", "input": "fresh probe", "expected": []}
                    ],
                    "retention": [
                        {
                            "id": "retention-1",
                            "input": "retention probe",
                            "expected": [],
                        }
                    ],
                    "oov": [{"id": "oov-1", "input": "oov probe", "expected": []}],
                },
            }
        ),
        encoding="utf-8",
    )
    calibration_receipt = calibrate_checkpoint(
        checkpoint_path,
        calibration,
        tmp_path / "calibrated",
        promotion_suite=promotion,
        steps=1,
    )
    calibrated = CheckpointPredictor(calibration_receipt["output_path"])
    calibrated_prediction = calibrated(
        {"id": "probe", "input": "reduce latency", "projection": "kinds"}
    )
    assert calibrated_prediction["score_status"] == "CALIBRATED"
    assert calibrated_prediction["temperature"] == pytest.approx(
        calibration_receipt["temperature"]
    )

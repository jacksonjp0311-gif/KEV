from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from kev.encoders import (
    FrozenSentenceEncoderProposal,
    load_verified_local_encoder,
    verify_local_encoder_manifest,
)


class DummyLocalEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))

    def encode(self, texts):
        return (
            torch.tensor(
                [[len(text), text.count(" ") + 1, 1.0, 0.0] for text in texts],
                dtype=torch.float32,
            )
            * self.scale
        )


def test_local_sentence_encoder_stays_frozen_and_only_proposes():
    encoder = DummyLocalEncoder()
    model = FrozenSentenceEncoderProposal(encoder, encoder_dim=4)
    model.train()

    assert model.encoder.training is False
    assert all(not parameter.requires_grad for parameter in model.encoder.parameters())
    assert model.trainable_parameter_count > 0

    results = model.proposals(["reduce latency", "never modify production"])
    assert len(results) == 2
    assert all(result["authority"] == "PROPOSAL_ONLY" for result in results)
    assert all(result["score_status"] == "UNCALIBRATED" for result in results)
    assert all(0 <= result["cardinality"] <= 6 for result in results)


def test_local_encoder_output_shape_is_enforced():
    model = FrozenSentenceEncoderProposal(DummyLocalEncoder(), encoder_dim=5)
    try:
        model.forward_texts(["one"])
    except ValueError as error:
        assert "expected (1,5)" in str(error)
    else:
        raise AssertionError("incorrect local encoder shape must be rejected")


def _manifest(tmp_path: Path, **updates) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    model_file = tmp_path / "encoder.bin"
    model_file.write_bytes(b"permission-clean-local-encoder-fixture")
    value = {
        "schema": "kev.local-encoder.v1",
        "name": "test-local-encoder",
        "embedding_dim": 4,
        "network_policy": "LOCAL_ONLY",
        "permission": {
            "status": "APPROVED",
            "reviewed_by": "test-reviewer",
            "reviewed_at": "2026-09-28T00:00:00Z",
            "scope": "local KEV proposal testing",
        },
        "license": {"identifier": "TEST-ONLY"},
        "files": [
            {
                "path": "encoder.bin",
                "sha256": hashlib.sha256(model_file.read_bytes()).hexdigest(),
            }
        ],
    }
    value.update(updates)
    path = tmp_path / "encoder-manifest.json"
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_verified_local_encoder_manifest_freezes_loaded_encoder(tmp_path: Path):
    manifest = _manifest(tmp_path)
    loaded = load_verified_local_encoder(
        manifest, lambda _root, _record: DummyLocalEncoder()
    )

    assert all(not parameter.requires_grad for parameter in loaded.encoder.parameters())
    assert loaded.artifact_metadata["network_policy"] == "LOCAL_ONLY"
    assert len(loaded.artifact_metadata["manifest_sha256"]) == 64
    proposal = loaded.proposals(["a local sentence"])[0]
    assert proposal["encoder_artifact"]["name"] == "test-local-encoder"
    assert proposal["authority"] == "PROPOSAL_ONLY"


def test_local_encoder_bytes_are_verified_before_loader_runs(tmp_path: Path):
    manifest = _manifest(tmp_path)
    (tmp_path / "encoder.bin").write_bytes(b"tampered")
    called = False

    def loader(_root, _record):
        nonlocal called
        called = True
        return DummyLocalEncoder()

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_verified_local_encoder(manifest, loader)
    assert called is False


def test_local_encoder_manifest_requires_permission_and_local_paths(tmp_path: Path):
    missing_permission = _manifest(tmp_path / "permission", permission={})
    with pytest.raises(ValueError, match="approved permission"):
        verify_local_encoder_manifest(missing_permission)

    remote = _manifest(
        tmp_path / "remote",
        files=[{"path": "https://example.invalid/model.bin", "sha256": "0" * 64}],
    )
    with pytest.raises(ValueError, match="local relative path"):
        verify_local_encoder_manifest(remote)

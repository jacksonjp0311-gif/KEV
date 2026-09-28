from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256, json_artifact_hashes
from kev.evaluation import evaluate_challenger, load_frozen_suite
from kev.model_runtime import CheckpointPredictor
from scripts import build_public_eval_pack_v6


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "evals/frozen/public-audit-v6-260.json"
MANIFEST_PATH = ROOT / "evals/frozen/manifest-v6.json"
AUDIT_PATH = ROOT / "evals/evidence/v5-fresh-template-audit.json"
BASELINE_PATH = ROOT / "evals/evidence/v6-genesis-baseline.json"
CHECKPOINT_PATH = (
    ROOT / "models/public/semantic-breadth-genesis-sha256-"
    "8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt"
)

FROZEN_HASHES = {
    "evals/evidence/v5-fresh-template-audit.json": (
        "b307dfafbe7d15c27a0b3d9d63a928aab05c1d99125877b8d0db74737b23ee65"
    ),
    "evals/frozen/public-audit-v6-260.json": (
        "a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029"
    ),
    "evals/frozen/manifest-v6.json": (
        "07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce"
    ),
    "evals/evidence/v6-genesis-baseline.json": (
        "881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4"
    ),
    "evals/frozen/public-audit-v5-260.json": (
        "e913f483bdce2e3ea1a435f0c139a846874a73660665c69b8723a9670348451c"
    ),
    "evals/frozen/manifest-v5.json": (
        "4b2cb72aa1a891e1673225be4ff9c5989cc1165ad79203938ffdbe4f794e3e94"
    ),
    "evals/evidence/v5-genesis-baseline.json": (
        "dded5f48db1af32efa3cc6bd3ffb61912fea65662e3da1ce21732cc6fd0eddc1"
    ),
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _items(suite: dict) -> dict[tuple[str, str], dict]:
    return {
        (split, item["id"]): item
        for split, rows in suite["splits"].items()
        for item in rows
    }


def test_v6_boundary_preserves_v5_and_changes_exactly_six_inputs():
    for relative, expected in FROZEN_HASHES.items():
        assert file_sha256(ROOT / relative) == expected

    v5 = _json(ROOT / "evals/frozen/public-audit-v5-260.json")
    v6 = _json(SUITE_PATH)
    old_items = _items(v5)
    new_items = _items(v6)
    assert old_items.keys() == new_items.keys()
    changed = set()
    for key, old in old_items.items():
        new = new_items[key]
        if new == old:
            continue
        expected = dict(old)
        expected["input"] = build_public_eval_pack_v6.REPLACEMENTS[old["id"]]
        assert new == expected
        changed.add(old["id"])
    assert changed == set(build_public_eval_pack_v6.REPLACEMENTS)
    assert sum(old_items[key] == new_items[key] for key in old_items) == 254

    build_public_eval_pack_v6.validate_suite(v6)
    assert build_public_eval_pack_v6.build_suite() == v6
    suite = load_frozen_suite(SUITE_PATH)
    assert suite.canonical_sha256 == (
        "3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c"
    )
    assert {name: len(rows) for name, rows in suite.splits.items()} == {
        "fresh": 80,
        "retention": 40,
        "oov": 40,
        "composition": 80,
        "calibration": 20,
    }


def test_v5_audit_finding_pins_repetitions_and_no_intervening_training():
    finding = _json(AUDIT_PATH)
    assert finding == build_public_eval_pack_v6.build_audit_finding()
    assert finding["finding"]["digit_normalized_template_repetition_count"] == 6
    assert finding["finding"]["exact_surface_overlap_count"] == 0
    assert {item["item_id"] for item in finding["finding"]["items"]} == set(
        build_public_eval_pack_v6.REPLACEMENTS
    )
    attestation = finding["intervening_change_attestation"]
    assert attestation == {
        "challenger_training_started": False,
        "model_weights_changed": False,
        "active_checkpoint_changed": False,
        "active_checkpoint_sha256_before_and_after": (
            "8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6"
        ),
        "statement": (
            "No challenger training or model change occurred between the v5 freeze "
            "and this finding."
        ),
    }
    assert finding["resolution"]["v5_bytes_preserved"] is True
    assert finding["resolution"]["action"] == (
        "CREATE_V6_WITH_SIX_NEW_SURFACES; DO_NOT_PATCH_V5"
    )


def test_v6_manifest_pins_every_artifact_and_reuses_v2_holdouts():
    manifest = _json(MANIFEST_PATH)
    assert manifest == build_public_eval_pack_v6.build_manifest()
    assert manifest["version"] == 6
    assert manifest["preserved_generations"] == [1, 2, 3, 4, 5]
    assert manifest["freshness_boundary"] == {
        "source": "evals/evidence/v5-fresh-template-audit.json",
        "replacement_count": 6,
        "replacement_overlap_check": (
            "ZERO_EXACT_AND_DIGIT_NORMALIZED_TEMPLATE_OVERLAP_WITH_"
            "V1_V5_AND_CALIBRATION_FIT_V2"
        ),
        "unchanged_item_count": 254,
        "challenger_training_before_freeze": False,
    }
    assert manifest["artifacts"]["calibration_fit"]["path"] == (
        "evals/frozen/calibration-fit-v2.jsonl"
    )
    assert manifest["artifacts"]["held_out_vocabulary"]["path"] == (
        "evals/frozen/held-out-vocabulary-v2.txt"
    )
    for reference in manifest["artifacts"].values():
        path = ROOT / reference["path"]
        assert path.stat().st_size == reference["size_bytes"]
        assert file_sha256(path) == reference["sha256"]
        if "canonical_sha256" in reference:
            assert (
                json_artifact_hashes(path)["canonical_json_sha256"]
                == reference["canonical_sha256"]
            )


def test_v6_builder_refuses_to_refresh_frozen_outputs():
    before = {
        path: file_sha256(path) for path in (AUDIT_PATH, SUITE_PATH, MANIFEST_PATH)
    }
    with pytest.raises(SystemExit, match="refusing to refresh"):
        build_public_eval_pack_v6.main()
    assert {path: file_sha256(path) for path in before} == before


def test_v6_genesis_baseline_replays_byte_exact_with_raw_failures():
    predictor = CheckpointPredictor(CHECKPOINT_PATH)
    report = evaluate_challenger(
        SUITE_PATH,
        incumbent_predictor=predictor,
        challenger_predictor=predictor,
        incumbent_path=CHECKPOINT_PATH,
        challenger_path=CHECKPOINT_PATH,
        incumbent_sha256=predictor.sha256,
        challenger_sha256=predictor.sha256,
    )
    encoded = (
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")
    assert hashlib.sha256(encoded).hexdigest() == (
        "881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4"
    )
    assert report == _json(BASELINE_PATH)
    assert report["report_sha256"] == (
        "e6c768c496048afbdcf56554700ac2cc19e9c09d013139e2c7c53cdb3a9f658e"
    )
    assert report["decision"]["decision"] == "REJECT"
    assert report["decision"]["reason_codes"] == [
        "FRESH_TIE",
        "COMPOSITION_TIE",
    ]
    assert report["incumbent"]["raw_failures"]["count"] == 119
    assert len(report["incumbent"]["audit_metrics"]) == 28
    assert report["incumbent"]["splits"]["fresh"]["correct"] == 48
    assert report["incumbent"]["splits"]["retention"]["correct"] == 40
    assert report["incumbent"]["splits"]["oov"]["correct"] == 3
    assert report["incumbent"]["splits"]["composition"]["correct"] == 50
    assert report["incumbent"]["splits"]["calibration"]["correct"] == 0
    body = dict(report)
    claimed = body.pop("report_sha256")
    assert canonical_json_sha256(body) == claimed

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from kev.artifacts import canonical_json_sha256, file_sha256, json_artifact_hashes
from kev.calibration import assert_disjoint_suites
from kev.evaluation import evaluate_challenger, load_frozen_suite
from kev.model_runtime import CheckpointPredictor
from scripts import build_public_eval_pack_v5


ROOT = Path(__file__).resolve().parents[1]
SUITE_PATH = ROOT / "evals/frozen/public-audit-v5-260.json"
MANIFEST_PATH = ROOT / "evals/frozen/manifest-v5.json"
HELD_OUT_PATH = ROOT / "evals/frozen/held-out-vocabulary-v2.txt"
CALIBRATION_PATH = ROOT / "evals/frozen/calibration-fit-v2.jsonl"
BASELINE_PATH = ROOT / "evals/evidence/v5-genesis-baseline.json"
CHECKPOINT_PATH = (
    ROOT / "models/public/semantic-breadth-genesis-sha256-"
    "8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt"
)

FROZEN_HASHES = {
    "evals/frozen/public-audit-v5-260.json": (
        "e913f483bdce2e3ea1a435f0c139a846874a73660665c69b8723a9670348451c"
    ),
    "evals/frozen/manifest-v5.json": (
        "4b2cb72aa1a891e1673225be4ff9c5989cc1165ad79203938ffdbe4f794e3e94"
    ),
    "evals/frozen/held-out-vocabulary-v2.txt": (
        "087da62e8129dd259c1b2662f0439eb7d6387714528af14444a4575c66e76798"
    ),
    "evals/frozen/calibration-fit-v2.jsonl": (
        "d0e7d4fcffef2c3ccd376acb2c77a1fb90289224caa046cddc88c9656ace101c"
    ),
    "evals/frozen/public-audit-v4-260.json": (
        "6e3ef02d4abc1101ee95d069e80e51963baa911b0ebd4305e9ec536ba0c57322"
    ),
    "evals/frozen/manifest-v4.json": (
        "21ae7711e3f3ad5323bc6a8a5669a5f073f7dedb646a61320dabfe487041888d"
    ),
    "evals/evidence/v4-genesis-baseline-audit-gated.json": (
        "e4675d7f2bd4370bcaa28a25cd213a7e1aecae46d43abca875a62660d815a003"
    ),
    "models/public/incumbent-evidence-v8.json": (
        "763b1826de9e54d9d4fb9bdd2e974fa67bfbf61b52be3fda43ed523026a2ed79"
    ),
}


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_v5_boundary_is_frozen_independent_label_clean_and_hash_pinned():
    for relative, expected in FROZEN_HASHES.items():
        assert file_sha256(ROOT / relative) == expected

    suite = load_frozen_suite(SUITE_PATH)
    manifest = _json(MANIFEST_PATH)
    assert suite.suite_id == "kev-public-audit-v5-260"
    assert suite.file_sha256 == FROZEN_HASHES["evals/frozen/public-audit-v5-260.json"]
    assert suite.canonical_sha256 == (
        "1cb5431b83e93f284bb1277ceea4c6e27ad1249f6a5c6fb24df4e22371e42698"
    )
    assert {name: len(rows) for name, rows in suite.splits.items()} == {
        "fresh": 80,
        "retention": 40,
        "oov": 40,
        "composition": 80,
        "calibration": 20,
    }
    assert suite.data["provenance"]["authored_without_training_corpus_access"] is True
    assert manifest["version"] == 5
    assert manifest["preserved_generations"] == [1, 2, 3, 4]
    assert manifest["training_boundary"] == {
        "suite": "FORBIDDEN_IN_TRAIN",
        "calibration_fit": "TEMPERATURE_ONLY_FORBIDDEN_IN_WEIGHT_TRAIN",
        "held_out_vocabulary": "FORBIDDEN_SURFACE_FORMS_IN_TRAIN",
    }

    for reference in manifest["artifacts"].values():
        path = ROOT / reference["path"]
        assert path.stat().st_size == reference["size_bytes"]
        assert file_sha256(path) == reference["sha256"]
        if "canonical_sha256" in reference:
            assert (
                json_artifact_hashes(path)["canonical_json_sha256"]
                == reference["canonical_sha256"]
            )

    build_public_eval_pack_v5.validate_suite(suite.data)
    assert build_public_eval_pack_v5.build_suite() == suite.data
    assert build_public_eval_pack_v5.build_manifest() == manifest


def test_v5_composition_and_held_out_contracts_are_explicit():
    suite = _json(SUITE_PATH)
    composition = suite["splits"]["composition"]
    assert {item["audit"] for item in composition} == (
        build_public_eval_pack_v5.REQUIRED_COMPOSITION_AUDITS
    )
    assert all(len(item["expected"]) >= 2 for item in composition)
    assert all(
        item["projection"] == "semantic_frames" for item in suite["splits"]["retention"]
    )

    held_out = tuple(HELD_OUT_PATH.read_text(encoding="utf-8").splitlines())
    assert held_out == build_public_eval_pack_v5.HELD_OUT_TERMS
    oov_text = "\n".join(item["input"].casefold() for item in suite["splits"]["oov"])
    non_oov_text = "\n".join(
        item["input"].casefold()
        for name, rows in suite["splits"].items()
        if name != "oov"
        for item in rows
    )
    assert all(oov_text.count(term) == 4 for term in held_out)
    assert all(term not in non_oov_text for term in held_out)


def test_v5_calibration_fit_is_label_clean_balanced_and_disjoint():
    rows = [
        json.loads(line)
        for line in CALIBRATION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    suite = _json(SUITE_PATH)
    build_public_eval_pack_v5.validate_calibration_fit(rows, suite)
    assert rows == build_public_eval_pack_v5.build_calibration_fit()
    assert_disjoint_suites(CALIBRATION_PATH, SUITE_PATH)
    assert len(rows) == 52


def test_v5_builder_refuses_to_refresh_any_frozen_output():
    before = {
        path: file_sha256(path)
        for path in (SUITE_PATH, MANIFEST_PATH, HELD_OUT_PATH, CALIBRATION_PATH)
    }
    with pytest.raises(SystemExit, match="refusing to refresh"):
        build_public_eval_pack_v5.main()
    assert {path: file_sha256(path) for path in before} == before


def test_v5_genesis_baseline_replays_byte_exact_with_raw_failures():
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
        "dded5f48db1af32efa3cc6bd3ffb61912fea65662e3da1ce21732cc6fd0eddc1"
    )
    assert report == _json(BASELINE_PATH)
    assert report["report_sha256"] == (
        "3d358cbedad47745c8520a3fa475dedd169654a8a972d95477afac94bce1d3f0"
    )
    assert report["decision"]["decision"] == "REJECT"
    assert report["decision"]["reason_codes"] == ["FRESH_TIE", "COMPOSITION_TIE"]
    assert report["incumbent"]["raw_failures"]["count"] == 119
    assert len(report["incumbent"]["audit_metrics"]) == 28
    assert report["incumbent"]["splits"]["fresh"]["correct"] == 48
    assert report["incumbent"]["splits"]["retention"]["correct"] == 40
    assert report["incumbent"]["splits"]["oov"]["correct"] == 3
    assert report["incumbent"]["splits"]["composition"]["correct"] == 50
    assert report["incumbent"]["splits"]["calibration"]["correct"] == 0
    card_body = dict(report)
    claimed_hash = card_body.pop("report_sha256")
    assert canonical_json_sha256(card_body) == claimed_hash

"""Record the v7 baseline and advance evidence, without changing model weights.

Run only after the runtime and independently authored v7 pack are frozen.
Historical baseline cards and the registry snapshot are retained verbatim.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from kev.artifacts import file_sha256
from kev.evaluation import evaluate_challenger, load_frozen_suite
from kev.evolution import _atomic_json
from kev.model_runtime import CheckpointPredictor


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "models/registry.json"
SUITE = ROOT / "evals/frozen/public-audit-v7-260.json"
MANIFEST = ROOT / "evals/frozen/manifest-v7.json"
BASELINE = ROOT / "evals/evidence/v7-genesis-baseline.json"
EVIDENCE = ROOT / "models/public/incumbent-evidence-v11.json"
SNAPSHOT = ROOT / "models/public/registry-before-v7.json"
GENESIS = "8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6"
V6_COMMIT = "1f5d9f322876d3e9ad7500b543521a64a3de5be1"
MANIFEST_SHA256 = "f8784d55629f86377ca80aadf52c838e40b20c497c28653b8605d5e505f6a036"


def _source_hashes() -> dict[str, str]:
    paths = {Path(__file__).resolve(), *(ROOT / "kev").rglob("*.py")}
    return {
        path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted(paths)
    }


def _assert_unchanged(expected: dict[Path, str], sources: dict[str, str]) -> None:
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"baseline input changed during measurement: {path}")
    if _source_hashes() != sources:
        raise ValueError("runtime source inventory or bytes changed during measurement")


def _write_new(path: Path, value: object) -> None:
    payload = (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> None:
    if any(path.exists() for path in (BASELINE, EVIDENCE, SNAPSHOT)):
        raise SystemExit("v7 baseline evidence already exists; refusing to refresh")
    sources = _source_hashes()
    registry_bytes = REGISTRY.read_bytes()
    registry = json.loads(registry_bytes)
    active = registry["active"]
    checkpoint = ROOT / active["path"]
    if active["sha256"] != GENESIS or file_sha256(checkpoint) != GENESIS:
        raise ValueError("the expected genesis incumbent has changed")
    previous_path = ROOT / active["evidence_manifest_path"]
    previous_bytes = previous_path.read_bytes()
    previous_sha256 = hashlib.sha256(previous_bytes).hexdigest()
    if previous_sha256 != active["evidence_manifest_sha256"]:
        raise ValueError("previous evidence hash mismatch")
    previous = json.loads(previous_bytes)
    manifest_bytes = MANIFEST.read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != MANIFEST_SHA256:
        raise ValueError("v7 evaluation manifest SHA-256 mismatch")
    manifest = json.loads(manifest_bytes)
    captured = {
        REGISTRY: hashlib.sha256(registry_bytes).hexdigest(),
        checkpoint: GENESIS,
        previous_path: previous_sha256,
        MANIFEST: MANIFEST_SHA256,
    }
    for reference in manifest["artifacts"].values():
        path = ROOT / reference["path"]
        if file_sha256(path) != reference["sha256"]:
            raise ValueError(f"manifest artifact mismatch: {reference['path']}")
        captured[path] = reference["sha256"]
    promotion = manifest["artifacts"]["promotion_suite"]
    if (ROOT / promotion["path"]).resolve() != SUITE.resolve():
        raise ValueError("v7 manifest promotion suite path mismatch")
    suite = load_frozen_suite(
        SUITE,
        expected_file_sha256=promotion["sha256"],
        expected_canonical_sha256=promotion["canonical_sha256"],
    )
    predictor = CheckpointPredictor(checkpoint)
    if predictor.sha256 != GENESIS:
        raise ValueError("loaded predictor differs from captured genesis incumbent")
    _assert_unchanged(captured, sources)
    card = evaluate_challenger(
        suite,
        incumbent_predictor=predictor,
        challenger_predictor=predictor,
        incumbent_path=checkpoint,
        challenger_path=checkpoint,
        incumbent_sha256=GENESIS,
        challenger_sha256=GENESIS,
    )
    # No frozen output is written until every measured input and source has
    # been checked against the identities captured before model inference.
    _assert_unchanged(captured, sources)
    _write_new(BASELINE, card)
    evidence = {
        **previous,
        "evaluated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "claim_boundary": (
            "untrained genesis weights measured with the v7 composition runtime; "
            "runtime changes and weight improvement are separate claims"
        ),
        "eval_card_path": BASELINE.relative_to(ROOT).as_posix(),
        "eval_card_sha256": file_sha256(BASELINE),
        "eval_manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "eval_manifest_sha256": MANIFEST_SHA256,
        "suite_path": SUITE.relative_to(ROOT).as_posix(),
        "suite_sha256": suite.file_sha256,
        "suite_canonical_sha256": suite.canonical_sha256,
        "report_sha256": card["report_sha256"],
        "raw_failure_count": card["incumbent"]["raw_failures"]["count"],
        "audit_family_gate_count": len(card["incumbent"]["audit_metrics"]),
        "metrics": {
            f"{split}_exact_match": metrics["accuracy"]
            for split, metrics in card["incumbent"]["splits"].items()
        },
        "previous_evidence_manifest_path": previous_path.relative_to(ROOT).as_posix(),
        "previous_evidence_manifest_sha256": previous_sha256,
        "previous_runtime_git_commit": V6_COMMIT,
        "runtime_source_sha256": sources,
        "weight_update": False,
    }
    evidence.pop("freshness_audit_finding_path", None)
    evidence.pop("freshness_audit_finding_sha256", None)
    _write_new(EVIDENCE, evidence)
    with SNAPSHOT.open("xb") as handle:
        handle.write(registry_bytes)
    active.update(
        evidence_manifest_path=EVIDENCE.relative_to(ROOT).as_posix(),
        evidence_manifest_sha256=file_sha256(EVIDENCE),
        eval_card_path=BASELINE.relative_to(ROOT).as_posix(),
        eval_card_sha256=file_sha256(BASELINE),
        claim_boundary="untrained bootstrap artifact; v7 baseline is not a weight improvement",
    )
    if REGISTRY.read_bytes() != registry_bytes:
        raise ValueError(
            "registry changed while measuring baseline; evidence preserved"
        )
    _atomic_json(REGISTRY, registry)
    print(
        json.dumps(
            {
                "baseline_sha256": file_sha256(BASELINE),
                "evidence_sha256": file_sha256(EVIDENCE),
                "registry_sha256": file_sha256(REGISTRY),
                "scores": {
                    name: {key: metrics[key] for key in ("correct", "total")}
                    for name, metrics in card["incumbent"]["splits"].items()
                },
                "checkpoint_sha256": GENESIS,
                "weight_update": False,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

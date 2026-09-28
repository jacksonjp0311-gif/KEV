from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from kev import __version__
from kev.artifacts import (
    canonical_json_sha256,
    file_sha256,
    validate_sha256,
    verify_file_sha256,
)
from kev.evaluation import (
    evaluate_challenger,
    frozen_suite,
    load_frozen_suite,
    replay_historical_aggregate,
)
from kev.evolution import (
    DEFAULT_CALIBRATION,
    DEFAULT_EVAL_MANIFEST,
    DEFAULT_HELD_OUT,
    DEFAULT_REGISTRY,
    DEFAULT_SUITE,
    evolve,
)
from kev.frames import extract_frames
from kev.model_runtime import CheckpointPredictor


ROOT = Path(__file__).resolve().parents[1]


def _required_text(record: Mapping[str, Any], field: str) -> str:
    value = record.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _repo_artifact(root: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty repository-relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise ValueError(f"{field} must be repository-relative")
    resolved_root = root.resolve()
    resolved = (resolved_root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes the repository root") from exc
    return resolved


def _read_json_object(
    path: Path,
    *,
    label: str,
    expected_sha256: str | None = None,
) -> tuple[dict[str, Any], str]:
    raw = path.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None:
        expected = validate_sha256(expected_sha256, field=f"{label}_sha256")
        if actual != expected:
            raise ValueError(
                f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
            )
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, actual


def _same_artifact(left: Path, right: Path, *, label: str) -> None:
    if left.resolve() != right.resolve():
        raise ValueError(f"{label} path does not match the evidence-pinned path")


def _verify_public_evidence_chain(
    *,
    root: Path,
    registry_path: Path,
    expected_manifest_path: Path,
    expected_suite_path: Path,
) -> dict[str, Any]:
    """Verify registry → evidence → artifacts without trusting manifest contents first."""

    registry, _ = _read_json_object(registry_path, label="model registry")
    active = registry.get("active")
    if not isinstance(active, Mapping):
        raise ValueError("model registry requires an active object")

    checkpoint_path = _repo_artifact(root, active.get("path"), field="active.path")
    checkpoint_sha256 = _required_text(active, "sha256")
    verify_file_sha256(checkpoint_path, checkpoint_sha256)

    evidence_path = _repo_artifact(
        root,
        active.get("evidence_manifest_path"),
        field="active.evidence_manifest_path",
    )
    evidence_sha256 = _required_text(active, "evidence_manifest_sha256")
    evidence, _ = _read_json_object(
        evidence_path,
        label="incumbent evidence",
        expected_sha256=evidence_sha256,
    )

    evidence_checkpoint_path = _repo_artifact(
        root,
        evidence.get("checkpoint_path"),
        field="evidence.checkpoint_path",
    )
    _same_artifact(
        checkpoint_path,
        evidence_checkpoint_path,
        label="checkpoint",
    )
    if _required_text(evidence, "checkpoint_sha256") != checkpoint_sha256:
        raise ValueError("incumbent evidence references a different checkpoint hash")

    eval_card_path = _repo_artifact(
        root,
        evidence.get("eval_card_path"),
        field="evidence.eval_card_path",
    )
    active_eval_card_path = _repo_artifact(
        root,
        active.get("eval_card_path"),
        field="active.eval_card_path",
    )
    _same_artifact(eval_card_path, active_eval_card_path, label="evaluation card")
    eval_card_sha256 = _required_text(evidence, "eval_card_sha256")
    if _required_text(active, "eval_card_sha256") != eval_card_sha256:
        raise ValueError("registry and evidence evaluation-card hashes differ")
    eval_card, _ = _read_json_object(
        eval_card_path,
        label="evaluation card",
        expected_sha256=eval_card_sha256,
    )
    claimed_report_sha256 = _required_text(eval_card, "report_sha256")
    evidence_report_sha256 = _required_text(evidence, "report_sha256")
    if claimed_report_sha256 != evidence_report_sha256:
        raise ValueError("evaluation-card report hash differs from incumbent evidence")
    report_body = dict(eval_card)
    report_body.pop("report_sha256")
    actual_report_sha256 = canonical_json_sha256(report_body)
    if actual_report_sha256 != claimed_report_sha256:
        raise ValueError(
            "evaluation-card report SHA-256 does not match its canonical content"
        )

    manifest_path = _repo_artifact(
        root,
        evidence.get("eval_manifest_path"),
        field="evidence.eval_manifest_path",
    )
    _same_artifact(manifest_path, expected_manifest_path, label="evaluation manifest")
    manifest_sha256 = _required_text(evidence, "eval_manifest_sha256")
    manifest, _ = _read_json_object(
        manifest_path,
        label="evaluation manifest",
        expected_sha256=manifest_sha256,
    )

    suite_path = _repo_artifact(
        root,
        evidence.get("suite_path"),
        field="evidence.suite_path",
    )
    _same_artifact(suite_path, expected_suite_path, label="promotion suite")
    suite_sha256 = _required_text(evidence, "suite_sha256")
    suite_data, _ = _read_json_object(
        suite_path,
        label="promotion suite",
        expected_sha256=suite_sha256,
    )
    suite = frozen_suite(suite_data)
    suite_canonical_sha256 = _required_text(evidence, "suite_canonical_sha256")
    if suite.canonical_sha256 != suite_canonical_sha256:
        raise ValueError("promotion-suite canonical SHA-256 mismatch")

    if eval_card.get("suite_canonical_sha256") != suite_canonical_sha256:
        raise ValueError("evaluation card references a different promotion suite")
    for role in ("incumbent", "challenger"):
        role_report = eval_card.get(role)
        role_suite = (
            role_report.get("suite") if isinstance(role_report, Mapping) else None
        )
        if not isinstance(role_suite, Mapping):
            raise ValueError(f"evaluation card requires {role}.suite evidence")
        if (
            role_suite.get("canonical_sha256") != suite_canonical_sha256
            or role_suite.get("file_sha256") != suite_sha256
            or role_suite.get("path") != evidence.get("suite_path")
        ):
            raise ValueError(
                f"evaluation card {role} suite evidence does not match active evidence"
            )

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("evaluation manifest requires an artifacts object")
    promotion_ref = artifacts.get("promotion_suite")
    if not isinstance(promotion_ref, Mapping):
        raise ValueError("evaluation manifest requires a promotion_suite reference")
    if (
        promotion_ref.get("path") != evidence.get("suite_path")
        or promotion_ref.get("sha256") != suite_sha256
        or promotion_ref.get("canonical_sha256") != suite_canonical_sha256
    ):
        raise ValueError(
            "evaluation manifest promotion suite differs from active evidence"
        )

    known_path = _repo_artifact(
        root,
        evidence.get("known_failures_path"),
        field="evidence.known_failures_path",
    )
    known_sha256 = _required_text(evidence, "known_failures_sha256")
    known, _ = _read_json_object(
        known_path,
        label="known failures",
        expected_sha256=known_sha256,
    )
    if evidence.get("known_failures_promotion_eligible") is not False:
        raise ValueError(
            "incumbent evidence must exclude known failures from promotion"
        )
    if known.get("promotion_eligible") is not False:
        raise ValueError("known failures must declare promotion_eligible=false")
    known_ref = artifacts.get("frame_parser_known_failures_v1")
    if not isinstance(known_ref, Mapping) or (
        known_ref.get("path") != evidence.get("known_failures_path")
        or known_ref.get("sha256") != known_sha256
        or known_ref.get("promotion_eligible") is not False
    ):
        raise ValueError(
            "evaluation manifest known-failure reference differs from active evidence"
        )

    for prefix in ("pre_audit_gate_eval_card", "previous_evidence_manifest"):
        path_field = f"{prefix}_path"
        hash_field = f"{prefix}_sha256"
        if path_field in evidence or hash_field in evidence:
            linked_path = _repo_artifact(
                root,
                evidence.get(path_field),
                field=f"evidence.{path_field}",
            )
            verify_file_sha256(linked_path, _required_text(evidence, hash_field))

    return {
        "active": dict(active),
        "artifacts": dict(artifacts),
        "checkpoint_path": checkpoint_path,
        "evidence_manifest_sha256": evidence_sha256,
        "eval_manifest_sha256": manifest_sha256,
        "known_failures_sha256": known_sha256,
        "report_sha256": actual_report_sha256,
        "suite_canonical_sha256": suite_canonical_sha256,
        "suite_sha256": suite_sha256,
    }


def _verify_manifest_artifacts(root: Path, artifacts: Mapping[str, Any]) -> None:
    for name, raw_reference in artifacts.items():
        if not isinstance(raw_reference, Mapping):
            raise ValueError(f"manifest artifact {name!r} must be an object")
        path = _repo_artifact(
            root,
            raw_reference.get("path"),
            field=f"manifest.artifacts.{name}.path",
        )
        verify_file_sha256(path, _required_text(raw_reference, "sha256"))
        expected_size = raw_reference.get("size_bytes")
        if expected_size is not None and path.stat().st_size != expected_size:
            raise ValueError(f"manifest artifact {name!r} size mismatch")


def _json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def _write_report(path: str | Path, value: Any) -> dict[str, str]:
    output = Path(path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    return {"path": str(output), "sha256": file_sha256(output)}


def _auto_temperature(checkpoint: str | Path) -> Path | None:
    candidate = Path(checkpoint).resolve().parent / "calibration-receipt.json"
    return candidate if candidate.is_file() else None


def command_eval(args: argparse.Namespace) -> int:
    suite = load_frozen_suite(args.suite)
    incumbent = CheckpointPredictor(
        args.incumbent, encoder_manifest=args.incumbent_encoder_manifest
    )
    challenger = CheckpointPredictor(
        args.challenger, encoder_manifest=args.challenger_encoder_manifest
    )
    report = evaluate_challenger(
        suite,
        incumbent_predictor=incumbent,
        challenger_predictor=challenger,
        incumbent_path=incumbent.path,
        challenger_path=challenger.path,
        incumbent_sha256=incumbent.sha256,
        challenger_sha256=challenger.sha256,
        incumbent_temperature=args.incumbent_temperature
        or _auto_temperature(incumbent.path),
        challenger_temperature=args.challenger_temperature
        or _auto_temperature(challenger.path),
        retention_epsilon=args.retention_epsilon,
    )
    if args.output:
        _write_report(args.output, report)
    _json(report)
    return 0


def command_evolve(args: argparse.Namespace) -> int:
    result = evolve(
        args.lessons,
        state_dir=args.state_dir,
        registry_path=args.registry,
        suite_path=args.suite,
        calibration_path=None if args.no_calibration else args.calibration_suite,
        held_out_vocabulary_path=args.held_out_vocabulary,
        output_root=args.output_root,
        encoder_manifest_path=args.encoder_manifest,
        steps=args.steps,
        seed=args.seed,
        retention_epsilon=args.retention_epsilon,
    )
    _json(result)
    return 0


def command_frames(args: argparse.Namespace) -> int:
    frames = extract_frames(
        args.message,
        source_type="CLI_LANGUAGE",
        source_id="kev-frames-command",
        actor="user",
    )
    _json(
        {
            "schema": "kev.frame-extraction.v1",
            "frames": [frame.to_dict() for frame in frames],
            "authority": "NONE",
            "state_mutation": "NONE",
        }
    )
    return 0


def command_historical_replay(_args: argparse.Namespace) -> int:
    _json(replay_historical_aggregate())
    return 0


def command_doctor(_args: argparse.Namespace) -> int:
    checks: list[dict[str, Any]] = []
    chain: dict[str, Any] | None = None
    try:
        chain = _verify_public_evidence_chain(
            root=ROOT,
            registry_path=DEFAULT_REGISTRY,
            expected_manifest_path=DEFAULT_EVAL_MANIFEST,
            expected_suite_path=DEFAULT_SUITE,
        )
        active = chain["active"]
        checks.append(
            {
                "name": "public-incumbent",
                "ok": True,
                "path": str(chain["checkpoint_path"]),
                "sha256": active["sha256"],
                "evidence_manifest_sha256": chain["evidence_manifest_sha256"],
                "eval_manifest_sha256": chain["eval_manifest_sha256"],
                "suite_sha256": chain["suite_sha256"],
                "suite_canonical_sha256": chain["suite_canonical_sha256"],
                "eval_report_sha256": chain["report_sha256"],
                "known_failures_sha256": chain["known_failures_sha256"],
            }
        )
    except Exception as error:
        checks.append({"name": "public-incumbent", "ok": False, "error": str(error)})
    try:
        if chain is None:
            raise ValueError("public evidence chain did not verify")
        _verify_manifest_artifacts(ROOT, chain["artifacts"])
        checks.append(
            {
                "name": "frozen-evaluation-artifacts",
                "ok": True,
                "manifest_sha256": chain["eval_manifest_sha256"],
            }
        )
    except Exception as error:
        checks.append(
            {"name": "frozen-evaluation-artifacts", "ok": False, "error": str(error)}
        )
    result = {
        "schema": "kev.doctor.v1",
        "version": __version__,
        "ok": all(check["ok"] for check in checks),
        "checks": checks,
    }
    result["report_sha256"] = canonical_json_sha256(result)
    _json(result)
    return 0 if result["ok"] else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kev")
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate = commands.add_parser(
        "eval", help="evaluate a challenger without activating it"
    )
    evaluate.add_argument("--challenger", required=True)
    evaluate.add_argument("--incumbent", required=True)
    evaluate.add_argument("--suite", required=True)
    evaluate.add_argument("--retention-epsilon", type=float, default=0.0)
    evaluate.add_argument("--incumbent-temperature")
    evaluate.add_argument("--challenger-temperature")
    evaluate.add_argument("--incumbent-encoder-manifest")
    evaluate.add_argument("--challenger-encoder-manifest")
    evaluate.add_argument("--output")
    evaluate.set_defaults(func=command_eval)

    evolution = commands.add_parser(
        "evolve", help="train and gate one immutable challenger"
    )
    evolution.add_argument("--lessons", required=True)
    evolution.add_argument("--state-dir")
    evolution.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    evolution.add_argument("--suite", default=str(DEFAULT_SUITE))
    evolution.add_argument("--calibration-suite", default=str(DEFAULT_CALIBRATION))
    evolution.add_argument("--held-out-vocabulary", default=str(DEFAULT_HELD_OUT))
    evolution.add_argument("--no-calibration", action="store_true")
    evolution.add_argument("--output-root")
    evolution.add_argument(
        "--encoder-manifest",
        help=(
            "verified local-only frozen sentence encoder manifest; omitted keeps "
            "the incumbent hashed-feature architecture"
        ),
    )
    evolution.add_argument("--steps", type=int, default=400)
    evolution.add_argument("--seed", type=int, default=52021)
    evolution.add_argument("--retention-epsilon", type=float, default=0.0)
    evolution.set_defaults(func=command_evolve)

    frames = commands.add_parser("frames", help="extract frames without mutating state")
    frames.add_argument("--message", required=True)
    frames.set_defaults(func=command_frames)

    replay = commands.add_parser(
        "historical-replay", help="replay the documented 190/260 tie"
    )
    replay.set_defaults(func=command_historical_replay)

    doctor = commands.add_parser(
        "doctor", help="verify public model and frozen eval hashes"
    )
    doctor.set_defaults(func=command_doctor)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema": "kev.cli-error.v1",
                    "error": type(error).__name__,
                    "message": str(error),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

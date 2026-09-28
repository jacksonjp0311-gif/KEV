"""Freeze KEV's v6 evaluation boundary without rewriting v1 through v5.

V6 is a narrowly scoped correction to the independently audited v5 boundary.
An audit found six v5 ``fresh`` surfaces that repeated v4 wording after only
numbers were changed.  No challenger training or model change occurred before
that finding.  This builder therefore:

* preserves the exact v5 suite, manifest, and baseline as development evidence;
* writes a machine-readable audit finding that pins all six repetitions;
* changes only the six affected input surfaces in the v6 item collection; and
* rejects replacements whose exact or digit-normalized template appears in any
  v1-v5 promotion suite or the independent calibration-fit slice.

The builder reads evaluation evidence only.  It never reads a lesson or
training corpus and performs no training.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Iterable

from kev.artifacts import canonical_json_sha256, file_sha256

if __package__:
    from scripts import build_public_eval_pack_v5 as v5
else:
    import build_public_eval_pack_v5 as v5


ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "evals" / "frozen"
EVIDENCE = ROOT / "evals" / "evidence"

V5_SUITE = FROZEN / "public-audit-v5-260.json"
V5_MANIFEST = FROZEN / "manifest-v5.json"
V5_BASELINE = EVIDENCE / "v5-genesis-baseline.json"
V5_AUDIT_FINDING = EVIDENCE / "v5-fresh-template-audit.json"
V6_SUITE = FROZEN / "public-audit-v6-260.json"
V6_MANIFEST = FROZEN / "manifest-v6.json"
CALIBRATION_FIT = FROZEN / "calibration-fit-v2.jsonl"
HELD_OUT = FROZEN / "held-out-vocabulary-v2.txt"
REGISTRY = ROOT / "models" / "registry.json"

V5_SUITE_SHA256 = "e913f483bdce2e3ea1a435f0c139a846874a73660665c69b8723a9670348451c"
V5_SUITE_CANONICAL_SHA256 = (
    "1cb5431b83e93f284bb1277ceea4c6e27ad1249f6a5c6fb24df4e22371e42698"
)
V5_MANIFEST_SHA256 = "4b2cb72aa1a891e1673225be4ff9c5989cc1165ad79203938ffdbe4f794e3e94"
V5_BASELINE_SHA256 = "dded5f48db1af32efa3cc6bd3ffb61912fea65662e3da1ce21732cc6fd0eddc1"
GENESIS_SHA256 = "8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6"
CALIBRATION_FIT_SHA256 = (
    "d0e7d4fcffef2c3ccd376acb2c77a1fb90289224caa046cddc88c9656ace101c"
)
HELD_OUT_SHA256 = "087da62e8129dd259c1b2662f0439eb7d6387714528af14444a4575c66e76798"

AUDITED_AT = "2026-09-28T11:31:10Z"

REPLACEMENTS = {
    "v5-fresh-kind-00-0": (
        "Populate vault-901 using everything currently stored in depot-901."
    ),
    "v5-fresh-kind-00-1": (
        "Let locker-902 receive the complete contents currently held by cabinet-902."
    ),
    "v5-fresh-kind-02-0": "Give the absolute size of negative 903.",
    "v5-fresh-kind-02-1": ("Report the nonnegative distance from -904 to zero."),
    "v5-fresh-kind-03-0": "Change 905 to its additive opposite.",
    "v5-fresh-kind-03-1": ("Return the number that cancels 906 when added to it."),
}

EXPECTED_REPETITIONS = {
    "v5-fresh-kind-00-0": (
        "v4-fresh-semantic-000",
        "v4-fresh-semantic-013",
        "v4-fresh-semantic-026",
        "v4-fresh-semantic-039",
    ),
    "v5-fresh-kind-00-1": (
        "v4-fresh-semantic-000",
        "v4-fresh-semantic-013",
        "v4-fresh-semantic-026",
        "v4-fresh-semantic-039",
    ),
    "v5-fresh-kind-02-0": (
        "v4-fresh-semantic-002",
        "v4-fresh-semantic-015",
        "v4-fresh-semantic-028",
    ),
    "v5-fresh-kind-02-1": (
        "v4-fresh-semantic-002",
        "v4-fresh-semantic-015",
        "v4-fresh-semantic-028",
    ),
    "v5-fresh-kind-03-0": (
        "v4-fresh-semantic-003",
        "v4-fresh-semantic-016",
        "v4-fresh-semantic-029",
    ),
    "v5-fresh-kind-03-1": (
        "v4-fresh-semantic-003",
        "v4-fresh-semantic-016",
        "v4-fresh-semantic-029",
    ),
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_surface(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(normalized.split())


def _digit_normalized_template(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "<N>", _normalized_surface(text))


def _all_items(suite: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    for split, rows in suite["splits"].items():
        for item in rows:
            yield split, item


def _verify_v5_boundary() -> tuple[dict[str, Any], dict[str, Any]]:
    expected = {
        V5_SUITE: V5_SUITE_SHA256,
        V5_MANIFEST: V5_MANIFEST_SHA256,
        V5_BASELINE: V5_BASELINE_SHA256,
        CALIBRATION_FIT: CALIBRATION_FIT_SHA256,
        HELD_OUT: HELD_OUT_SHA256,
    }
    for path, digest in expected.items():
        if file_sha256(path) != digest:
            raise ValueError(f"preserved artifact changed before v6 freeze: {path}")
    suite = _json(V5_SUITE)
    if canonical_json_sha256(suite) != V5_SUITE_CANONICAL_SHA256:
        raise ValueError("preserved v5 canonical suite hash changed")
    registry = _json(REGISTRY)
    if registry["active"]["sha256"] != GENESIS_SHA256:
        raise ValueError("active checkpoint changed before the v5 audit was resolved")
    return suite, registry


def build_audit_finding() -> dict[str, Any]:
    v5_suite, registry = _verify_v5_boundary()
    v4_suite = _json(FROZEN / "public-audit-v4-260.json")
    v4_items = {item["id"]: item for _, item in _all_items(v4_suite)}
    v5_items = {item["id"]: item for _, item in _all_items(v5_suite)}

    repetitions: list[dict[str, Any]] = []
    for item_id, expected_match_ids in EXPECTED_REPETITIONS.items():
        item = v5_items[item_id]
        template = _digit_normalized_template(item["input"])
        matches = []
        for match_id in expected_match_ids:
            prior = v4_items[match_id]
            if _digit_normalized_template(prior["input"]) != template:
                raise ValueError(f"declared repetition no longer matches: {item_id}")
            matches.append({"item_id": match_id, "input": prior["input"]})
        repetitions.append(
            {
                "item_id": item_id,
                "expected_kinds": item["expected"],
                "input": item["input"],
                "digit_normalized_template": template,
                "matched_v4_items": matches,
            }
        )

    return {
        "schema": "kev.eval-audit-finding.v1",
        "id": "v5-fresh-number-only-template-repetitions",
        "status": "FROZEN_DEVELOPMENT_EVIDENCE",
        "audited_at": AUDITED_AT,
        "scope": "v5 fresh surfaces compared with preserved v1-v4 evaluation surfaces",
        "method": {
            "exact_normalization": "Unicode NFKC, casefold, collapse whitespace",
            "template_normalization": (
                "exact normalization plus replace every decimal digit run with <N>"
            ),
        },
        "source_suite": {
            "path": V5_SUITE.relative_to(ROOT).as_posix(),
            "sha256": V5_SUITE_SHA256,
            "canonical_sha256": V5_SUITE_CANONICAL_SHA256,
        },
        "source_manifest": {
            "path": V5_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": V5_MANIFEST_SHA256,
        },
        "finding": {
            "exact_surface_overlap_count": 0,
            "digit_normalized_template_repetition_count": len(repetitions),
            "affected_split": "fresh",
            "items": repetitions,
        },
        "intervening_change_attestation": {
            "challenger_training_started": False,
            "model_weights_changed": False,
            "active_checkpoint_changed": False,
            "active_checkpoint_sha256_before_and_after": registry["active"]["sha256"],
            "statement": (
                "No challenger training or model change occurred between the v5 "
                "freeze and this finding."
            ),
        },
        "resolution": {
            "v5_bytes_preserved": True,
            "v5_status": "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE",
            "action": "CREATE_V6_WITH_SIX_NEW_SURFACES; DO_NOT_PATCH_V5",
            "replacement_count": len(REPLACEMENTS),
        },
    }


def _historical_surfaces() -> tuple[set[str], set[str]]:
    exact: set[str] = set()
    templates: set[str] = set()
    for version in range(1, 6):
        suite = _json(FROZEN / f"public-audit-v{version}-260.json")
        for _, item in _all_items(suite):
            exact.add(_normalized_surface(item["input"]))
            templates.add(_digit_normalized_template(item["input"]))
    for line in CALIBRATION_FIT.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        text = json.loads(line)["text"]
        exact.add(_normalized_surface(text))
        templates.add(_digit_normalized_template(text))
    return exact, templates


def build_suite(*, audit_finding_sha256: str | None = None) -> dict[str, Any]:
    v5_suite, _ = _verify_v5_boundary()
    suite = copy.deepcopy(v5_suite)
    seen_replacements: set[str] = set()
    for item in suite["splits"]["fresh"]:
        if item["id"] in REPLACEMENTS:
            item["input"] = REPLACEMENTS[item["id"]]
            seen_replacements.add(item["id"])
    if seen_replacements != set(REPLACEMENTS):
        raise ValueError("not every declared v6 replacement was applied")

    finding_hash = audit_finding_sha256
    if finding_hash is None:
        finding_hash = _sha256_bytes(_json_bytes(build_audit_finding()))
    suite["id"] = "kev-public-audit-v6-260"
    suite["provenance"] = {
        **suite["provenance"],
        "status": "FROZEN; create v7 rather than changing any byte",
        "created_for": "challenger decisions made only after the v6 boundary was frozen",
        "independence_policy": (
            "v5 was preserved after an independent audit found six number-only "
            "fresh-template repetitions; only those surfaces were replaced before "
            "challenger training, without reading a lesson or training corpus"
        ),
        "preserved_generations": [1, 2, 3, 4, 5],
        "predecessor_suite_sha256": V5_SUITE_SHA256,
        "predecessor_suite_canonical_sha256": V5_SUITE_CANONICAL_SHA256,
        "v5_audit_finding_path": V5_AUDIT_FINDING.relative_to(ROOT).as_posix(),
        "v5_audit_finding_sha256": finding_hash,
        "replacement_count": len(REPLACEMENTS),
        "replacement_overlap_policy": (
            "zero exact or digit-normalized template overlap with v1-v5 suites "
            "and calibration-fit-v2"
        ),
    }
    return suite


def validate_suite(suite: dict[str, Any]) -> None:
    v5_suite, _ = _verify_v5_boundary()
    v5.validate_suite(suite)
    if suite["id"] != "kev-public-audit-v6-260":
        raise ValueError("wrong v6 suite identifier")
    if suite["provenance"]["replacement_count"] != 6:
        raise ValueError("v6 replacement count changed")

    old_items = {(split, item["id"]): item for split, item in _all_items(v5_suite)}
    new_items = {(split, item["id"]): item for split, item in _all_items(suite)}
    if old_items.keys() != new_items.keys():
        raise ValueError("v6 item identifiers or split membership changed")
    changed: set[str] = set()
    for key, old in old_items.items():
        new = new_items[key]
        if old == new:
            continue
        expected = dict(old)
        expected["input"] = REPLACEMENTS[old["id"]]
        if new != expected:
            raise ValueError(f"v6 changed more than the input for {old['id']}")
        changed.add(old["id"])
    if changed != set(REPLACEMENTS):
        raise ValueError("v6 did not change exactly the six audited inputs")

    historical_exact, historical_templates = _historical_surfaces()
    replacement_exact = {_normalized_surface(text) for text in REPLACEMENTS.values()}
    replacement_templates = {
        _digit_normalized_template(text) for text in REPLACEMENTS.values()
    }
    if len(replacement_exact) != 6 or len(replacement_templates) != 6:
        raise ValueError("v6 replacement surfaces or templates are not unique")
    if replacement_exact & historical_exact:
        raise ValueError("a v6 replacement exactly overlaps prior evaluation")
    if replacement_templates & historical_templates:
        raise ValueError("a v6 replacement template overlaps prior evaluation")


def _artifact(
    path: Path,
    *,
    canonical: bool = False,
    content: bytes | None = None,
    value: Any | None = None,
    **extra: Any,
) -> dict[str, Any]:
    raw = path.read_bytes() if content is None else content
    result: dict[str, Any] = {
        "path": path.relative_to(ROOT).as_posix(),
        "sha256": _sha256_bytes(raw),
        "size_bytes": len(raw),
    }
    if canonical:
        canonical_value = json.loads(raw.decode("utf-8")) if value is None else value
        result["canonical_sha256"] = canonical_json_sha256(canonical_value)
    result.update(extra)
    return result


def build_manifest(
    *,
    suite: dict[str, Any] | None = None,
    suite_bytes: bytes | None = None,
    audit_finding: dict[str, Any] | None = None,
    audit_bytes: bytes | None = None,
) -> dict[str, Any]:
    if suite is None:
        suite = _json(V6_SUITE)
    if suite_bytes is None:
        suite_bytes = V6_SUITE.read_bytes()
    if audit_finding is None:
        audit_finding = _json(V5_AUDIT_FINDING)
    if audit_bytes is None:
        audit_bytes = V5_AUDIT_FINDING.read_bytes()

    v5_manifest = _json(V5_MANIFEST)
    artifacts = copy.deepcopy(v5_manifest["artifacts"])
    prior_suite = artifacts.pop("promotion_suite")
    prior_suite["status"] = "PRESERVED_DEVELOPMENT_EVIDENCE_NOT_CURRENT_PROMOTION_SUITE"
    artifacts["preserved_promotion_suite_v5"] = prior_suite
    artifacts.update(
        {
            "promotion_suite": _artifact(
                V6_SUITE,
                canonical=True,
                content=suite_bytes,
                value=suite,
            ),
            "preserved_manifest_v5": _artifact(
                V5_MANIFEST,
                canonical=True,
                status="PRESERVED_PREVIOUS_EVALUATION_MANIFEST",
            ),
            "v5_baseline_self_evaluation": _artifact(
                V5_BASELINE,
                canonical=True,
                status="PRESERVED_PRE_V6_BASELINE_WITH_RAW_FAILURES",
            ),
            "v5_freshness_audit_finding": _artifact(
                V5_AUDIT_FINDING,
                canonical=True,
                content=audit_bytes,
                value=audit_finding,
                promotion_eligible=False,
                status="PRESERVED_AUDIT_FINDING_NO_TRAINING_OCCURRED",
            ),
        }
    )
    return {
        "schema": "kev.eval-manifest.v1",
        "frozen": True,
        "version": 6,
        "preserved_generations": [1, 2, 3, 4, 5],
        "artifacts": artifacts,
        "mutation_policy": "create a new version; never refresh v1 through v6 in place",
        "training_boundary": {
            "suite": "FORBIDDEN_IN_TRAIN",
            "calibration_fit": "TEMPERATURE_ONLY_FORBIDDEN_IN_WEIGHT_TRAIN",
            "held_out_vocabulary": "FORBIDDEN_SURFACE_FORMS_IN_TRAIN",
        },
        "freshness_boundary": {
            "source": V5_AUDIT_FINDING.relative_to(ROOT).as_posix(),
            "replacement_count": 6,
            "replacement_overlap_check": (
                "ZERO_EXACT_AND_DIGIT_NORMALIZED_TEMPLATE_OVERLAP_WITH_"
                "V1_V5_AND_CALIBRATION_FIT_V2"
            ),
            "unchanged_item_count": 254,
            "challenger_training_before_freeze": False,
        },
    }


def main() -> None:
    outputs = (V5_AUDIT_FINDING, V6_SUITE, V6_MANIFEST)
    if any(path.exists() for path in outputs):
        raise SystemExit("v6 artifacts already exist; refusing to refresh them")

    audit_finding = build_audit_finding()
    audit_bytes = _json_bytes(audit_finding)
    audit_sha256 = _sha256_bytes(audit_bytes)
    suite = build_suite(audit_finding_sha256=audit_sha256)
    validate_suite(suite)
    suite_bytes = _json_bytes(suite)
    manifest = build_manifest(
        suite=suite,
        suite_bytes=suite_bytes,
        audit_finding=audit_finding,
        audit_bytes=audit_bytes,
    )
    manifest_bytes = _json_bytes(manifest)

    with V5_AUDIT_FINDING.open("xb") as handle:
        handle.write(audit_bytes)
    with V6_SUITE.open("xb") as handle:
        handle.write(suite_bytes)
    with V6_MANIFEST.open("xb") as handle:
        handle.write(manifest_bytes)

    print(
        json.dumps(
            {
                "audit_finding_sha256": audit_sha256,
                "suite_sha256": _sha256_bytes(suite_bytes),
                "suite_canonical_sha256": canonical_json_sha256(suite),
                "manifest_sha256": _sha256_bytes(manifest_bytes),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

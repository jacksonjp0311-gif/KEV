"""Build the corrected, immutable reviewed corpus for the KEV v5 experiment.

The first independently frozen corpus is preserved and rejected before
training because one exact semantic target appeared under two paraphrase group
IDs.  Since KEV's contrastive objective treats different explicit groups as
negatives, that collision would have supplied a contradictory training signal.

V6 derives only repository-authored synthetic rows from the preserved builder,
merges the colliding groups, re-reviews every row, and enforces the stronger
one-target-to-one-group invariant.  Like its predecessor, it never reads v5
promotion or calibration prompts.  Only the separately supplied v2 vocabulary
holdout is read from the v5 evaluation package.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kev.artifacts import canonical_json_sha256, file_sha256

try:
    from scripts import build_reviewed_training_corpus_v5 as prior
except ImportError:  # pragma: no cover - direct ``python scripts/...`` execution
    import build_reviewed_training_corpus_v5 as prior  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "training" / "reviewed"
CORPUS_NAME = "semantic-frame-paraphrases-v6-reviewed.jsonl"
MANIFEST_NAME = "semantic-frame-paraphrases-v6-manifest.json"
REJECTION_NAME = "semantic-frame-paraphrases-v5-rejection.json"
REVIEWED_AT = "2026-09-28T11:18:33Z"
REVIEWER = "openai-codex-agent:reviewed-corpus-task"
PERMISSION = (
    "repository-authored synthetic text; user-authorized for the corrected KEV "
    "v5 local training experiment; no third-party text; not independent human review"
)

V5_CORPUS_PATH = OUTPUT_DIRECTORY / prior.CORPUS_NAME
V5_CORPUS_SHA256 = "91b49f47804120906537f2b3eafeef5ddf1df1663f114e5c91a1c9fa59ce5dd8"
V5_MANIFEST_PATH = OUTPUT_DIRECTORY / prior.MANIFEST_NAME
V5_MANIFEST_SHA256 = "e783ece3d8aa4ac3d104833345ef771d1aa67536060a81be5e793c9c4d9c3b3d"
V5_BUILDER_PATH = ROOT / "scripts" / "build_reviewed_training_corpus_v5.py"
V5_BUILDER_SHA256 = "7b63bdb809becfde07641affc7ba4ee22387486ed5bb9d5ac9f51004aeea3a32"
COLLIDING_GROUPS = (
    "evidence-consistent-alpha",
    "evidence-consistent-gamma",
)
MERGED_GROUP = "evidence-consistent"


def _verify_predecessor() -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for path, expected, role in (
        (V5_CORPUS_PATH, V5_CORPUS_SHA256, "REJECTED_TRAINING_CORPUS"),
        (V5_MANIFEST_PATH, V5_MANIFEST_SHA256, "REJECTED_CORPUS_MANIFEST"),
        (V5_BUILDER_PATH, V5_BUILDER_SHA256, "PRESERVED_CORPUS_BUILDER"),
    ):
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(
                f"preserved v5 corpus evidence changed at {path}: "
                f"expected {expected}, got {actual}"
            )
        references.append(
            {
                "schema": "kev.artifact-ref.v1",
                "path": path.relative_to(ROOT).as_posix(),
                "role": role,
                "sha256": actual,
                "size_bytes": path.stat().st_size,
            }
        )
    return references


def build_rows() -> list[dict[str, Any]]:
    rows = deepcopy(prior.build_rows())
    for row in rows:
        predecessor_id = str(row["id"])
        predecessor_group = str(row["paraphrase_group"])
        corrected_group = (
            MERGED_GROUP if predecessor_group in COLLIDING_GROUPS else predecessor_group
        )
        row["id"] = predecessor_id.replace("v5-", "v6-", 1)
        row["derived_from_lesson_id"] = predecessor_id
        row["paraphrase_group"] = f"v6-{corrected_group}"
        row["reviewed_at"] = REVIEWED_AT
        row["reviewed_by"] = REVIEWER
        row["permission"] = PERMISSION
        row["authorship"]["authorization"] = (
            "user-authorized corrected KEV v5 experiment"
        )
        row["review"] = {
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "scope": (
                "surface text, exact typed frame-set target, and global "
                "semantic-target-to-group consistency"
            ),
        }
    return rows


def validate_target_group_bijection(rows: Sequence[Mapping[str, Any]]) -> None:
    target_groups: dict[str, set[str]] = {}
    for row in rows:
        frames = row.get("frames")
        group = row.get("paraphrase_group")
        if not isinstance(frames, list) or not isinstance(group, str) or not group:
            raise ValueError("structured frames and paraphrase group are required")
        signature = canonical_json_sha256(frames)
        target_groups.setdefault(signature, set()).add(group)
    collisions = {
        signature: sorted(groups)
        for signature, groups in target_groups.items()
        if len(groups) != 1
    }
    if collisions:
        raise ValueError(
            "one exact semantic target appears under multiple paraphrase_group IDs: "
            f"{json.dumps(collisions, sort_keys=True)}"
        )


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out_terms: Iterable[str],
    forbidden_surfaces: Iterable[str],
) -> None:
    # Reuse the complete structural/coverage check from the preserved builder.
    # Its provenance constants describe v5, so adapt only those metadata fields
    # in an in-memory validation copy; the emitted v6 rows are checked below.
    compatibility_rows = deepcopy(list(rows))
    for row in compatibility_rows:
        row["reviewed_at"] = prior.REVIEWED_AT
        row["reviewed_by"] = prior.REVIEWER
        row["permission"] = prior.PERMISSION
    prior.validate_rows(
        compatibility_rows,
        held_out_terms=held_out_terms,
        forbidden_surfaces=forbidden_surfaces,
    )
    for row in rows:
        if (
            row.get("reviewed_at") != REVIEWED_AT
            or row.get("reviewed_by") != REVIEWER
            or row.get("permission") != PERMISSION
        ):
            raise ValueError("corrected corpus review provenance changed")
    validate_target_group_bijection(rows)


def _rejection_body(predecessor_references: list[dict[str, Any]]) -> dict[str, Any]:
    v5_rows = prior.build_rows()
    collision_members = [
        row for row in v5_rows if row["paraphrase_group"] in COLLIDING_GROUPS
    ]
    signatures = {canonical_json_sha256(row["frames"]) for row in collision_members}
    if len(collision_members) != 6 or len(signatures) != 1:
        raise ValueError("preserved v5 collision no longer replays exactly")
    return {
        "schema": "kev.training-corpus-rejection.v1",
        "id": "kev-v5-reviewed-semantic-frame-corpus-rejection",
        "created_at": REVIEWED_AT,
        "decision": "REJECTED_BEFORE_TRAINING",
        "reason_code": "SEMANTIC_TARGET_MULTIPLE_PARAPHRASE_GROUPS",
        "reason": (
            "one exact EVIDENCE_CONSISTENCY frame target appeared under two "
            "paraphrase group IDs; the contrastive objective would treat "
            "cross-group same-semantics pairs as negatives"
        ),
        "collision": {
            "canonical_frames_sha256": next(iter(signatures)),
            "paraphrase_groups": list(COLLIDING_GROUPS),
            "lesson_ids": sorted(str(row["id"]) for row in collision_members),
            "row_count": len(collision_members),
        },
        "artifacts_preserved": predecessor_references,
        "training": {
            "optimizer_steps": 0,
            "checkpoint_created": False,
            "model_weights_touched": False,
            "training_receipt": None,
        },
        "promotion": {
            "decision": "NOT_APPLICABLE",
            "eval_card": None,
            "ledger_model_event": None,
        },
        "resolution": (
            "preserve v5 bytes and issue a separately versioned v6 corpus with "
            "a global one-target-to-one-group validation"
        ),
    }


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _vocabulary_and_exclusions(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
) -> tuple[set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    v1_terms, v1_reference = prior._vocabulary(
        prior.PINNED_V1_VOCABULARY,
        prior.PINNED_V1_VOCABULARY_SHA256,
    )
    v2_terms, v2_reference = prior._vocabulary(
        held_out_v2_path.resolve(), held_out_v2_sha256
    )
    v1_reference["path"] = "evals/frozen/held-out-vocabulary-v1.txt"
    v2_reference["path"] = "evals/frozen/held-out-vocabulary-v2.txt"
    forbidden, exclusions = prior._frozen_v1_v4_surfaces()
    return v1_terms | v2_terms, [v1_reference, v2_reference], exclusions


def build_artifacts(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
) -> tuple[bytes, bytes, bytes, dict[str, Any]]:
    predecessor_references = _verify_predecessor()
    rejection = _rejection_body(predecessor_references)
    rejection["rejection_canonical_sha256"] = canonical_json_sha256(rejection)
    encoded_rejection = _json_bytes(rejection)

    held_out, vocabulary_references, exclusions = _vocabulary_and_exclusions(
        held_out_v2_path, held_out_v2_sha256
    )
    forbidden, _ = prior._frozen_v1_v4_surfaces()
    rows = build_rows()
    validate_rows(
        rows,
        held_out_terms=held_out,
        forbidden_surfaces=forbidden,
    )
    encoded_corpus = prior.corpus_bytes(rows)
    semantic_targets = [{"id": row["id"], "frames": row["frames"]} for row in rows]
    builder_path = Path(__file__)
    manifest = {
        "schema": "kev.reviewed-training-corpus-manifest.v1",
        "id": "kev-v6-reviewed-semantic-frame-corpus",
        "version": 6,
        "frozen": True,
        "created_at": REVIEWED_AT,
        "purpose": (
            "corrected first bounded experiment for frame-kind and cardinality proposal heads"
        ),
        "corpus": {
            "schema": "kev.artifact-ref.v1",
            "path": f"training/reviewed/{CORPUS_NAME}",
            "sha256": hashlib.sha256(encoded_corpus).hexdigest(),
            "size_bytes": len(encoded_corpus),
            "canonical_rows_sha256": canonical_json_sha256(rows),
            "semantic_targets_sha256": canonical_json_sha256(semantic_targets),
        },
        "builder": {
            "schema": "kev.artifact-ref.v1",
            "path": "scripts/build_reviewed_training_corpus_v6.py",
            "sha256": file_sha256(builder_path),
            "size_bytes": builder_path.stat().st_size,
        },
        "builder_dependencies": [
            reference
            for reference in predecessor_references
            if reference["role"] == "PRESERVED_CORPUS_BUILDER"
        ],
        "predecessor_rejection": {
            "schema": "kev.artifact-ref.v1",
            "path": f"training/reviewed/{REJECTION_NAME}",
            "sha256": hashlib.sha256(encoded_rejection).hexdigest(),
            "size_bytes": len(encoded_rejection),
            "canonical_json_sha256": canonical_json_sha256(rejection),
            "decision": "REJECTED_BEFORE_TRAINING",
        },
        "metrics": prior._metrics(rows),
        "provenance": {
            "authorship": "REPOSITORY_AUTHORED_SYNTHETIC",
            "authored_by": "openai-codex-agent:reviewed-corpus-task",
            "reviewed_by": REVIEWER,
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "authorization": "user-authorized corrected KEV v5 repository experiment",
            "permission": PERMISSION,
            "ordinary_chat_training_data": False,
            "third_party_text": False,
            "derived_from_rejected_corpus_sha256": V5_CORPUS_SHA256,
        },
        "separation": {
            "v5_promotion_suite_read_by_builder": False,
            "v5_promotion_suite_surface_used": False,
            "v5_promotion_suite_provenance_only": {
                "file_sha256": prior.V5_SUITE_FILE_SHA256_PROVENANCE_ONLY,
                "canonical_sha256": prior.V5_SUITE_CANONICAL_SHA256_PROVENANCE_ONLY,
                "content_read": False,
            },
            "v5_vocabulary_holdout_read_only": True,
            "v1_v4_exact_surface_exclusions": exclusions,
            "held_out_vocabularies": vocabulary_references,
            "final_training_boundary": (
                "train() must still verify this corpus against every frozen suite at execution time"
            ),
        },
        "coverage": {
            "frame_kinds": list(prior.FRAME_KINDS),
            "cardinalities": [1, 2, 3, 4],
            "required_phenomena": [
                "single-frame",
                "multi-frame",
                "negation",
                "order-swap",
                "number-change",
                "polite-buried-constraint",
                "conflicting-constraints",
                "repeated-kind",
            ],
            "semantic_target_group_policy": "EXACTLY_ONE_PARAPHRASE_GROUP_PER_TARGET",
        },
        "limitations": [
            "synthetic corpus; it does not establish natural-language breadth",
            "same-party agent review; it is not independent human or external review",
            "exact surface and vocabulary separation do not prove semantic independence",
            "training completion and lower loss cannot qualify a model",
        ],
        "mutation_policy": (
            "immutable after first write; create a new corpus and manifest version for any change"
        ),
    }
    encoded_manifest = _json_bytes(manifest)
    return encoded_corpus, encoded_manifest, encoded_rejection, manifest


def write_artifacts(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    corpus, manifest_bytes, rejection_bytes, manifest = build_artifacts(
        held_out_v2_path, held_out_v2_sha256
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    targets = {
        output_directory / CORPUS_NAME: corpus,
        output_directory / MANIFEST_NAME: manifest_bytes,
        output_directory / REJECTION_NAME: rejection_bytes,
    }
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(
            "v6 reviewed corpus evidence already exists; refusing to refresh: "
            + ", ".join(existing)
        )
    for path, content in targets.items():
        with path.open("xb") as handle:
            handle.write(content)
    return manifest


def check_artifacts(
    held_out_v2_path: Path,
    held_out_v2_sha256: str,
    output_directory: Path = OUTPUT_DIRECTORY,
) -> dict[str, Any]:
    corpus, manifest_bytes, rejection_bytes, manifest = build_artifacts(
        held_out_v2_path, held_out_v2_sha256
    )
    expected = {
        output_directory / CORPUS_NAME: corpus,
        output_directory / MANIFEST_NAME: manifest_bytes,
        output_directory / REJECTION_NAME: rejection_bytes,
    }
    for path, content in expected.items():
        if path.read_bytes() != content:
            raise ValueError(
                f"frozen corpus evidence does not replay byte-for-byte: {path}"
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build or verify the corrected immutable v6 training corpus."
    )
    parser.add_argument("--held-out-v2", type=Path, default=prior.PINNED_V2_VOCABULARY)
    parser.add_argument(
        "--held-out-v2-sha256", default=prior.PINNED_V2_VOCABULARY_SHA256
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    operation = check_artifacts if args.check else write_artifacts
    manifest = operation(
        args.held_out_v2,
        args.held_out_v2_sha256,
        args.output_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

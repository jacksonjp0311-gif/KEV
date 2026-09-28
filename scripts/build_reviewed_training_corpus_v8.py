"""Remove three independently flagged pretraining overlaps from preserved v7.

Only lesson identifiers are supplied by the separation audit. This builder does
not read v7 evaluation surfaces or change retained lesson text/semantic targets.
The rejected v7 corpus, its builder, manifest, and rejection receipt stay intact.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kev.artifacts import canonical_json_sha256

try:
    from scripts import build_reviewed_training_corpus_v7 as prior
except ImportError:  # pragma: no cover
    import build_reviewed_training_corpus_v7 as prior  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "training" / "reviewed"
CORPUS_NAME = "semantic-frame-paraphrases-v8-reviewed.jsonl"
MANIFEST_NAME = "semantic-frame-paraphrases-v8-manifest.json"
REVIEWED_AT = "2026-09-28T13:46:00Z"
REVIEWER = "openai-codex-agent:lessons-v8"
PERMISSION = (
    "repository-authored synthetic text; user-authorized KEV evolution; "
    "pretraining overlap removal; no chat or third-party text; "
    "same-party agent review, not human review"
)
SOURCE_CORPUS = OUTPUT_DIRECTORY / prior.CORPUS_NAME
SOURCE_CORPUS_SHA256 = (
    "2515fc2fd940c18d1bc0188d1ae5e108c966ddb7bf391ae66c1c248ad6879b29"
)
SOURCE_MANIFEST = OUTPUT_DIRECTORY / prior.MANIFEST_NAME
SOURCE_MANIFEST_SHA256 = (
    "f0e45c4f224c8dcfef0e1e7c0b5132bc33f45821458d36608001f4f900ce3511"
)
SOURCE_BUILDER_SHA256 = (
    "3b410a20d5009c59f6447851896c9b9a63843259e6147be9de7b4137b2e134e6"
)
REJECTION_RECEIPT = ROOT / "evals" / "evidence" / "v7-input-separation.json"
REJECTION_RECEIPT_SHA256 = (
    "d8e4374ee79e77b1a9279b238f97444023cb823e65266b109cf7e4461a37988c"
)
REMOVED_LESSON_IDS = (
    "v7-retained-receipt-rec-31-01",
    "v7-retained-receipt-rec-44-01",
    "v7-retained-receipt-rec-58-01",
)


def source_references() -> list[dict[str, Any]]:
    return [
        prior._reference(SOURCE_CORPUS, SOURCE_CORPUS_SHA256),
        prior._reference(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        prior._reference(Path(prior.__file__), SOURCE_BUILDER_SHA256),
    ]


def build_rows() -> list[dict[str, Any]]:
    source_references()
    prior._reference(REJECTION_RECEIPT, REJECTION_RECEIPT_SHA256)
    source = [
        json.loads(line)
        for line in SOURCE_CORPUS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    removed = [row["id"] for row in source if row["id"] in REMOVED_LESSON_IDS]
    if sorted(removed) != sorted(REMOVED_LESSON_IDS):
        raise ValueError("the exact three rejected lesson IDs must exist once")
    rows = []
    for row in source:
        if row["id"] in REMOVED_LESSON_IDS:
            continue
        predecessor_id = row["id"]
        predecessor = {
            "lesson_id": predecessor_id,
            "corpus_sha256": SOURCE_CORPUS_SHA256,
            "review": {
                key: row[key] for key in ("reviewed_by", "reviewed_at", "permission")
            },
            "prior_derived_from_lesson_id": row.get("derived_from_lesson_id"),
            "prior_source_corpus_sha256": row.get("source_corpus_sha256"),
            "prior_source_review": row.get("source_review"),
        }
        row["id"] = predecessor_id.replace("v7-", "v8-", 1)
        row["paraphrase_group"] = row["paraphrase_group"].replace("v7-", "v8-", 1)
        row["derived_from_lesson_id"] = predecessor_id
        row["source_corpus_sha256"] = SOURCE_CORPUS_SHA256
        row["source_review"] = predecessor["review"]
        row["predecessor_lineage"] = predecessor
        row["reviewed_by"] = REVIEWER
        row["reviewed_at"] = REVIEWED_AT
        row["permission"] = PERMISSION
        row["review"] = {
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "scope": (
                "only the three independently flagged IDs removed; retained surfaces "
                "and exact frame multisets unchanged; paraphrase groups remain valid"
            ),
        }
        row["training_tags"] = sorted(
            set(row["training_tags"]) | {"retained-v7", "pretraining-separation-repair"}
        )
        rows.append(row)
    return rows


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out_terms: Iterable[str],
    forbidden_surfaces: Iterable[str],
) -> None:
    for row in rows:
        if (
            row["reviewed_by"] != REVIEWER
            or row["reviewed_at"] != REVIEWED_AT
            or row["permission"] != PERMISSION
        ):
            raise ValueError("v8 review provenance changed")
        if row["derived_from_lesson_id"] in REMOVED_LESSON_IDS:
            raise ValueError("rejected source lesson must not enter v8")
    compatible = deepcopy(list(rows))
    for row in compatible:
        row["reviewed_by"] = prior.REVIEWER
        row["reviewed_at"] = prior.REVIEWED_AT
        row["permission"] = prior.PERMISSION
    prior.validate_rows(
        compatible,
        held_out_terms=held_out_terms,
        forbidden_surfaces=forbidden_surfaces,
    )
    groups = Counter(row["paraphrase_group"] for row in rows)
    if any(size < 2 for size in groups.values()):
        raise ValueError("a retained paraphrase group became a singleton")
    if len(rows) != 497:
        raise ValueError("v8 must retain exactly 497 source rows")


def build_artifacts() -> tuple[bytes, bytes, dict[str, Any]]:
    held_out, forbidden, vocabularies, exclusions = prior.exclusions()
    rows = build_rows()
    validate_rows(rows, held_out_terms=held_out, forbidden_surfaces=forbidden)
    corpus = prior.helpers.corpus_bytes(rows)
    metrics = prior.helpers._metrics(rows)
    manifest = {
        "schema": "kev.reviewed-training-corpus-manifest.v1",
        "id": "kev-v8-reviewed-semantic-frame-corpus",
        "version": 8,
        "frozen": True,
        "created_at": REVIEWED_AT,
        "purpose": "pretraining separation repair for the bounded v7 compositional experiment",
        "corpus": {
            "path": f"training/reviewed/{CORPUS_NAME}",
            "sha256": hashlib.sha256(corpus).hexdigest(),
            "size_bytes": len(corpus),
            "canonical_rows_sha256": canonical_json_sha256(rows),
            "semantic_targets_sha256": canonical_json_sha256(
                [{"id": row["id"], "frames": row["frames"]} for row in rows]
            ),
        },
        "builder": prior._reference(Path(__file__)),
        "builder_dependencies": [
            prior._reference(Path(prior.__file__), SOURCE_BUILDER_SHA256),
            prior._reference(Path(prior.helpers.__file__), prior.HELPERS_SHA256),
        ],
        "preserved_sources": source_references(),
        "predecessor_rejection": {
            **prior._reference(REJECTION_RECEIPT, REJECTION_RECEIPT_SHA256),
            "decision": "REJECTED_BEFORE_TRAINING",
            "reason": "three digit-normalized training/evaluation template overlaps",
            "exact_surface_overlap_count": 0,
            "digit_normalized_overlap_count": 3,
            "removed_lesson_ids": list(REMOVED_LESSON_IDS),
            "optimizer_steps": 0,
            "model_scores_used": False,
            "receipt_content_read_by_builder": False,
        },
        "metrics": metrics,
        "provenance": {
            "authorship": "REPOSITORY_AUTHORED_SYNTHETIC",
            "reviewed_by": REVIEWER,
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "permission": PERMISSION,
            "ordinary_chat_training_data": False,
            "third_party_text": False,
            "retained_v7_row_count": len(rows),
            "new_surface_count": 0,
            "removed_source_row_count": 3,
            "source_content_preserved": True,
        },
        "separation": {
            "v7_evaluation_prompts_read": False,
            "v7_evaluation_builder_read": False,
            "held_out_vocabularies": vocabularies,
            "preserved_surface_exclusions": exclusions,
            "distinct_forbidden_surface_count": len(forbidden),
            "repair_signal": "REMOVED_SOURCE_LESSON_IDS_ONLY",
            "final_training_boundary": "root must re-audit v8 against the unchanged v7 suite before optimizer steps",
        },
        "semantic_audit": {
            "target_identity": "CANONICAL_ORDER_INDEPENDENT_MULTISET_WITH_MULTIPLICITY",
            "group_policy": "EXACTLY_ONE_PARAPHRASE_GROUP_PER_TARGET",
            "minimum_group_size": min(
                Counter(row["paraphrase_group"] for row in rows).values()
            ),
            "surviving_texts_and_frames_unchanged": True,
            "labels_generated_from_runtime_parser": False,
        },
        "limitations": [
            "Synthetic bounded templates do not establish natural-language breadth.",
            "Same-party agent review is not independent human or external review.",
            "The v7 corpus rejection is preserved; three rows were removed before any training.",
            "No held-out evaluation item was rewritten and no model scores informed this repair.",
            "The final cross-input audit must succeed before training.",
        ],
        "mutation_policy": "exclusive writes; preserve all rejected and derived artifacts",
    }
    return corpus, prior._json_bytes(manifest), manifest


def write_artifacts(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Any]:
    corpus, manifest_bytes, manifest = build_artifacts()
    targets = {
        output_directory / CORPUS_NAME: corpus,
        output_directory / MANIFEST_NAME: manifest_bytes,
    }
    if any(path.exists() for path in targets):
        raise FileExistsError("v8 corpus artifacts exist; refusing to refresh")
    output_directory.mkdir(parents=True, exist_ok=True)
    for path, content in targets.items():
        with path.open("xb") as handle:
            handle.write(content)
    return manifest


def check_artifacts(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Any]:
    corpus, manifest_bytes, manifest = build_artifacts()
    for name, expected in ((CORPUS_NAME, corpus), (MANIFEST_NAME, manifest_bytes)):
        if (output_directory / name).read_bytes() != expected:
            raise ValueError(
                f"frozen v8 artifact does not replay byte-for-byte: {name}"
            )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIRECTORY)
    args = parser.parse_args()
    result = (check_artifacts if args.check else write_artifacts)(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

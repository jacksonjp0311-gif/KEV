"""Author the independently prepared, immutable v7 compositional lesson corpus.

Targets are authored directly as typed semantic frames, never inferred by the
parser under test. The author/builder does not read v7 evaluation prompts.
Composition combines independent clauses and preserves their exact frame
multiset under surface paraphrase and clause permutation. Retained v6 lessons
keep their source lineage. All review is same-party agent review, not human or
external validation; these are synthetic permission-clean lessons, not chat.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.uc51a2.semantic_breadth import _reviewed_rows_from_text, _target_hash

try:
    from scripts import build_reviewed_training_corpus_v5 as helpers
except ImportError:  # pragma: no cover
    import build_reviewed_training_corpus_v5 as helpers  # type: ignore[no-redef]


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIRECTORY = ROOT / "training" / "reviewed"
CORPUS_NAME = "semantic-frame-paraphrases-v7-reviewed.jsonl"
MANIFEST_NAME = "semantic-frame-paraphrases-v7-manifest.json"
REVIEWED_AT = "2026-09-28T13:43:00Z"
REVIEWER = "openai-codex-agent:lessons-v7"
PERMISSION = (
    "repository-authored synthetic text; user-authorized KEV v7 evolution; "
    "no third-party text or chat; same-party agent review, not human review"
)
SOURCE_CORPUS = OUTPUT_DIRECTORY / "semantic-frame-paraphrases-v6-reviewed.jsonl"
SOURCE_CORPUS_SHA256 = (
    "c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed"
)
SOURCE_MANIFEST = OUTPUT_DIRECTORY / "semantic-frame-paraphrases-v6-manifest.json"
SOURCE_MANIFEST_SHA256 = (
    "185ea3efcfe5a8083faa11c2a8dfc0a4df63d5844fe2fa640873a3730d850c4e"
)
HELPERS_SHA256 = "7b63bdb809becfde07641affc7ba4ee22387486ed5bb9d5ac9f51004aeea3a32"

# Explicit old-artifact inventory: no discovery pattern may read a v7 prompt.
OLD_SURFACE_ARTIFACTS = (
    *helpers.V1_V4_SURFACE_ARTIFACTS,
    (
        "evals/frozen/public-audit-v5-260.json",
        "e913f483bdce2e3ea1a435f0c139a846874a73660665c69b8723a9670348451c",
        "EVALUATION_SUITE",
    ),
    (
        "evals/frozen/public-audit-v6-260.json",
        "a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029",
        "EVALUATION_SUITE",
    ),
    (
        "evals/frozen/calibration-fit-v2.jsonl",
        "d0e7d4fcffef2c3ccd376acb2c77a1fb90289224caa046cddc88c9656ace101c",
        "CALIBRATION_FIT",
    ),
)


def _reference(path: Path, expected: str | None = None) -> dict[str, Any]:
    actual = file_sha256(path)
    if expected is not None and actual != expected:
        raise ValueError(f"preserved source hash mismatch: {path}")
    return {
        "schema": "kev.artifact-ref.v1",
        "path": path.resolve().relative_to(ROOT).as_posix(),
        "sha256": actual,
        "size_bytes": path.stat().st_size,
    }


def _provenance() -> dict[str, Any]:
    return {
        "status": "REVIEWED",
        "reviewed_by": REVIEWER,
        "reviewed_at": REVIEWED_AT,
        "permission": PERMISSION,
        "authorship": {
            "source_type": "REPOSITORY_AUTHORED_SYNTHETIC",
            "authored_by": REVIEWER,
            "authorization": "user-authorized KEV evolution",
            "ordinary_chat_source": False,
            "third_party_source": False,
        },
        "review": {
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "scope": (
                "independently authored semantic atoms, typed exact slots, clause "
                "composition and order swaps, canonical multiset group identity"
            ),
        },
    }


def semantic_atoms() -> list[dict[str, Any]]:
    """Two atoms per kind plus explicit conflicting policy counterparts."""
    h = helpers
    specifications = [
        (
            "goal-jitter",
            h._threshold("GOAL", "sensor_jitter", "LT", 36, "ms"),
            (
                "Bring sensor jitter below 36 ms",
                "We want sensor jitter to be less than 36 milliseconds",
                "The target is sensor jitter under 36 ms",
            ),
        ),
        (
            "constraint-archive",
            h._policy("erase", "archive", False),
            (
                "Do not erase the archive",
                "Please refrain from erasing the archive",
                "I would appreciate it if you never erased the archive",
            ),
        ),
        (
            "observation-jitter",
            h._measurement("sensor_jitter", 48, "ms"),
            (
                "Sensor jitter measured 48 ms",
                "The observed sensor jitter was 48 milliseconds",
                "A measurement put sensor jitter at 48 ms",
            ),
        ),
        (
            "prediction-jitter",
            h._forecast("sensor_jitter", "LT", 31, "ms"),
            (
                "I expect sensor jitter below 31 ms",
                "Sensor jitter will probably be less than 31 milliseconds",
                "We predict sensor jitter under 31 ms",
            ),
        ),
        (
            "copy-birch",
            h._base(
                "COPY_VALUE",
                source=("birch_value", "value"),
                destination=("reed_slot", "reference"),
            ),
            (
                "Copy the birch value into the reed slot",
                "Put a duplicate of the birch value in the reed slot",
                "The reed slot should receive a copy of the birch value",
            ),
        ),
        (
            "revision-juniper",
            h._base(
                "SUPERSEDES",
                old_value=("build_fir", "value"),
                current_value=("build_juniper", "value"),
            ),
            (
                "Build juniper replaces build fir",
                "Build fir is superseded by build juniper",
                "Replace build fir with build juniper",
            ),
        ),
        (
            "magnitude-47",
            h._base("MAGNITUDE", input=(-47, "number")),
            (
                "Find the absolute value of -47",
                "Give the magnitude of -47",
                "How far is -47 from zero",
            ),
        ),
        (
            "negate-26",
            h._base("NEGATE", input=(26, "number")),
            (
                "Reverse the sign of 26",
                "Give the opposite of 26",
                "Negate the number 26",
            ),
        ),
        (
            "reference-folio",
            h._base("REFERENCE", target=("folio-kestrel", "reference")),
            (
                "Refer to folio-kestrel",
                "Use folio-kestrel as the reference",
                "Point the reference to folio-kestrel",
            ),
        ),
        (
            "active-river",
            h._base(
                "ACTIVE_SELECTION",
                selection=("profile_river", "identifier"),
                active=(True, "boolean"),
            ),
            (
                "Profile river is active",
                "The active profile is river",
                "Profile river is the current selection",
            ),
        ),
        (
            "run-fern",
            h._base(
                "RUN_STATUS",
                run_id=("fern-probe", "identifier"),
                status=("FAILED", "status"),
            ),
            (
                "Run fern-probe failed",
                "The status of run fern-probe is failed",
                "Run fern-probe ended in failure",
            ),
        ),
        (
            "receipt-wren",
            h._base(
                "RECEIPT_VALUE",
                receipt_id=("wren-slip", "identifier"),
                value=(63, "number"),
            ),
            (
                "Receipt wren-slip returned 63",
                "The value recorded on receipt wren-slip is 63",
                "Receipt wren-slip carries the result 63",
            ),
        ),
        (
            "evidence-agrees",
            h._base("EVIDENCE_CONSISTENCY", status=("CONSISTENT", "status")),
            (
                "The separate pieces of evidence all agree",
                "The collected evidence is mutually consistent",
                "Agreement holds across the evidence records",
            ),
        ),
        (
            "goal-depth",
            h._threshold("GOAL", "buffer_depth", "LTE", 19, "items"),
            (
                "Keep buffer depth at most 19 items",
                "We want buffer depth no greater than 19 items",
                "Our target is buffer depth of 19 items or fewer",
            ),
        ),
        (
            "constraint-preserve",
            h._policy("preserve", "journal", True),
            (
                "You must preserve the journal",
                "Please ensure that the journal is preserved",
                "Preserving the journal is required",
            ),
        ),
        (
            "observation-depth",
            h._measurement("buffer_depth", 27, "items"),
            (
                "Buffer depth measured 27 items",
                "The observed buffer depth was 27 items",
                "A measurement put buffer depth at 27 items",
            ),
        ),
        (
            "prediction-depth",
            h._forecast("buffer_depth", "GT", 12, "items"),
            (
                "I predict buffer depth above 12 items",
                "Buffer depth will probably exceed 12 items",
                "We expect buffer depth greater than 12 items",
            ),
        ),
        (
            "copy-pebble",
            h._base(
                "COPY_VALUE",
                source=("pebble_value", "value"),
                destination=("clay_slot", "reference"),
            ),
            (
                "Copy the pebble value into the clay slot",
                "Put a duplicate of the pebble value in the clay slot",
                "The clay slot should receive a copy of the pebble value",
            ),
        ),
        (
            "revision-cedar",
            h._base(
                "SUPERSEDES",
                old_value=("edition_ash", "value"),
                current_value=("edition_cedar", "value"),
            ),
            (
                "Edition cedar replaces edition ash",
                "Edition ash is superseded by edition cedar",
                "Replace edition ash with edition cedar",
            ),
        ),
        (
            "magnitude-62",
            h._base("MAGNITUDE", input=(-62, "number")),
            (
                "Find the absolute value of -62",
                "Give the magnitude of -62",
                "How far is -62 from zero",
            ),
        ),
        (
            "negate-minus-38",
            h._base("NEGATE", input=(-38, "number")),
            (
                "Reverse the sign of -38",
                "Give the opposite of -38",
                "Negate the number -38",
            ),
        ),
        (
            "reference-spruce",
            h._base("REFERENCE", target=("folio-spruce", "reference")),
            (
                "Refer to folio-spruce",
                "Use folio-spruce as the reference",
                "Point the reference to folio-spruce",
            ),
        ),
        (
            "active-meadow",
            h._base(
                "ACTIVE_SELECTION",
                selection=("profile_meadow", "identifier"),
                active=(False, "boolean"),
            ),
            (
                "Profile meadow is not active",
                "The active selection is not profile meadow",
                "Profile meadow is inactive",
            ),
        ),
        (
            "run-willow",
            h._base(
                "RUN_STATUS",
                run_id=("willow-check", "identifier"),
                status=("COMPLETED", "status"),
            ),
            (
                "Run willow-check completed",
                "The status of run willow-check is completed",
                "Run willow-check finished successfully",
            ),
        ),
        (
            "receipt-grove",
            h._base(
                "RECEIPT_VALUE",
                receipt_id=("grove-slip", "identifier"),
                value=(81, "number"),
            ),
            (
                "Receipt grove-slip returned 81",
                "The value recorded on receipt grove-slip is 81",
                "Receipt grove-slip carries the result 81",
            ),
        ),
        (
            "evidence-disagrees",
            h._base("EVIDENCE_CONSISTENCY", status=("CONFLICT", "status")),
            (
                "The separate pieces of evidence disagree",
                "The collected evidence is contradictory",
                "Conflict remains across the evidence records",
            ),
        ),
        (
            "constraint-erase-required",
            h._policy("erase", "archive", True),
            (
                "You must erase the archive",
                "Erasing the archive is required",
                "Please ensure that the archive is erased",
            ),
        ),
        (
            "constraint-preserve-forbidden",
            h._policy("preserve", "journal", False),
            (
                "Do not preserve the journal",
                "Preserving the journal is forbidden",
                "Please refrain from preserving the journal",
            ),
        ),
    ]
    return [
        {"id": name, "frame": frame, "clauses": list(clauses)}
        for name, frame, clauses in specifications
    ]


def _composition_selections() -> list[tuple[int, ...]]:
    # Same-kind pairs include two independently bound slot values. The two
    # evidence status atoms are NOT combined: they lack a subject slot and
    # would collapse two unspecified evidence domains into one ambiguity.
    repeated = [(index, index + 13) for index in range(12)]
    adjacent = [(index, (index + 1) % 13) for index in range(13)]
    cross = [(index, index + 14) for index in range(11)]
    triples = [(index, (index + 3) % 26, (index + 8) % 26) for index in range(24)]
    quads = [
        (index, (index + 3) % 13, index + 13, (index + 3) % 13 + 13)
        for index in range(12)
    ]
    # Quads involving evidence use only one unscoped evidence status.
    quads = [selection for selection in quads if not {12, 25}.issubset(selection)]
    return repeated + adjacent + cross + triples + quads + [(1, 26), (14, 27)]


def _composition_text(atoms: Sequence[Mapping[str, Any]], variant: int) -> str:
    order = list(range(len(atoms)))
    if variant == 1:
        order.reverse()
    elif variant == 2:
        order = order[1:] + order[:1]
    clauses = [str(atoms[index]["clauses"][variant]) for index in order]
    separator = ("; ", ". ", ", and ")[variant]
    return separator.join(clauses) + "."


def build_rows() -> list[dict[str, Any]]:
    _reference(SOURCE_CORPUS, SOURCE_CORPUS_SHA256)
    _reference(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256)
    _reference(Path(helpers.__file__), HELPERS_SHA256)
    rows = [
        json.loads(line)
        for line in SOURCE_CORPUS.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for row in rows:
        source_id = row["id"]
        source_review = {
            key: row[key] for key in ("reviewed_by", "reviewed_at", "permission")
        }
        row["id"] = source_id.replace("v6-", "v7-retained-", 1)
        row["paraphrase_group"] = row["paraphrase_group"].replace(
            "v6-", "v7-retained-", 1
        )
        row["derived_from_lesson_id"] = source_id
        row["source_corpus_sha256"] = SOURCE_CORPUS_SHA256
        row["source_review"] = source_review
        row.update(_provenance())
        row["training_tags"] = sorted(set(row["training_tags"]) | {"retained-v6"})

    groups = {_target_hash(row["frames"]): row["paraphrase_group"] for row in rows}

    def add(
        name: str,
        texts: Sequence[str],
        frames: list[dict[str, Any]],
        tags: set[str],
        atom_ids: list[str],
    ) -> None:
        target = _target_hash(frames)
        group = groups.setdefault(target, f"v7-{name}")
        for variant, text in enumerate(texts):
            rows.append(
                {
                    "schema": "kev.reviewed-lesson.v1",
                    "id": f"v7-{name}-{variant + 1:02d}",
                    **_provenance(),
                    "text": text,
                    "frames": deepcopy(frames),
                    "frame_kinds": list(
                        dict.fromkeys(frame["kind"] for frame in frames)
                    ),
                    "frame_cardinality": len(frames),
                    "paraphrase_group": group,
                    "training_tags": sorted(tags),
                    "semantic_atom_ids": atom_ids,
                    "surface_variant": variant,
                    "target_authorship": "DIRECT_TYPED_SEMANTIC_AUTHORING_NOT_PARSER_OUTPUT",
                }
            )

    atoms = semantic_atoms()
    for atom in atoms:
        add(
            f"single-{atom['id']}",
            [clause + "." for clause in atom["clauses"]],
            [atom["frame"]],
            {"single-frame", "paraphrase", "new-v7"},
            [atom["id"]],
        )
    for index, selection in enumerate(_composition_selections()):
        members = [atoms[item] for item in selection]
        frames = [member["frame"] for member in members]
        tags = {
            "multi-frame",
            "paraphrase",
            "order-swap",
            "new-v7",
            f"cardinality-{len(frames)}",
        }
        kinds = [frame["kind"] for frame in frames]
        if len(set(kinds)) < len(kinds):
            tags.add("repeated-kind")
        if any(
            frame["slots"].get("polarity", {}).get("value") is False
            or frame["slots"].get("active", {}).get("value") is False
            for frame in frames
        ):
            tags.add("negation")
        if "CONSTRAINT" in kinds:
            tags.add("polite-buried-constraint")
        policies = [frame for frame in frames if frame["relation"] == "ACTION_POLICY"]
        if (
            len(policies) == 2
            and all(
                policies[0]["slots"][key] == policies[1]["slots"][key]
                for key in ("action", "object")
            )
            and policies[0]["slots"]["polarity"] != policies[1]["slots"]["polarity"]
        ):
            tags.add("conflicting-constraints")
        add(
            f"composition-{index + 1:03d}",
            [_composition_text(members, variant) for variant in range(3)],
            frames,
            tags,
            [member["id"] for member in members],
        )
    return rows


def exclusions() -> tuple[
    set[str], set[str], list[dict[str, Any]], list[dict[str, Any]]
]:
    terms: set[str] = set()
    vocabularies: list[dict[str, Any]] = []
    for path, expected in (
        (helpers.PINNED_V1_VOCABULARY, helpers.PINNED_V1_VOCABULARY_SHA256),
        (helpers.PINNED_V2_VOCABULARY, helpers.PINNED_V2_VOCABULARY_SHA256),
    ):
        found, _ = helpers._vocabulary(path, expected)
        terms.update(found)
        vocabularies.append(_reference(path, expected))
    surfaces: set[str] = set()
    references: list[dict[str, Any]] = []
    for relative, expected, role in OLD_SURFACE_ARTIFACTS:
        path = ROOT / relative
        references.append({**_reference(path, expected), "role": role})
        if role == "CALIBRATION_FIT":
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    surfaces.add(
                        str(row.get("text", row.get("input", ""))).strip().casefold()
                    )
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            surfaces.update(
                helpers._suite_surfaces(data)
                if role == "EVALUATION_SUITE"
                else helpers._known_failure_surfaces(data)
            )
    return terms, surfaces, vocabularies, references


def validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    held_out_terms: Iterable[str],
    forbidden_surfaces: Iterable[str],
) -> None:
    identifiers = [row["id"] for row in rows]
    surfaces = [str(row["text"]).strip().casefold() for row in rows]
    if len(set(identifiers)) != len(rows) or len(set(surfaces)) != len(rows):
        raise ValueError("duplicate lesson identifier or surface")
    for row in rows:
        if (
            row["reviewed_by"] != REVIEWER
            or row["reviewed_at"] != REVIEWED_AT
            or row["permission"] != PERMISSION
        ):
            raise ValueError("review provenance changed")
        if row["review"]["independent_human_review"] is not False:
            raise ValueError("same-party review must not claim human review")
        if any(
            row["authorship"][key] is not False
            for key in ("ordinary_chat_source", "third_party_source")
        ):
            raise ValueError("chat and third-party sources are forbidden")
        if row["frame_cardinality"] != len(row["frames"]):
            raise ValueError("frame multiplicity differs from cardinality")
        for frame in row["frames"]:
            operator = frame["slots"].get("operator")
            if operator is not None and operator["value"] not in {
                "LT",
                "LTE",
                "EQ",
                "GTE",
                "GT",
                "NE",
            }:
                raise ValueError("unknown canonical comparison operator")
    _reviewed_rows_from_text(
        helpers.corpus_bytes(rows).decode("utf-8"), held_out_terms, forbidden_surfaces
    )
    cardinalities = Counter(len(row["frames"]) for row in rows)
    if not all(cardinalities[value] >= 20 for value in (1, 2, 3, 4)):
        raise ValueError("insufficient cardinality coverage")
    if not 0.35 <= cardinalities[1] / len(rows) <= 0.50:
        raise ValueError("single-frame retention balance is outside the declared range")
    if {frame["kind"] for row in rows for frame in row["frames"]} != set(
        helpers.FRAME_KINDS
    ):
        raise ValueError("missing frame kind")


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n"
    ).encode("utf-8")


def build_artifacts() -> tuple[bytes, bytes, dict[str, Any]]:
    held_out, forbidden, vocabulary_references, exclusion_references = exclusions()
    rows = build_rows()
    validate_rows(rows, held_out_terms=held_out, forbidden_surfaces=forbidden)
    corpus = helpers.corpus_bytes(rows)
    manifest = {
        "schema": "kev.reviewed-training-corpus-manifest.v1",
        "id": "kev-v7-reviewed-semantic-frame-corpus",
        "version": 7,
        "frozen": True,
        "created_at": REVIEWED_AT,
        "purpose": "bounded compositional frame-kind and cardinality proposal training",
        "corpus": {
            "path": f"training/reviewed/{CORPUS_NAME}",
            "sha256": hashlib.sha256(corpus).hexdigest(),
            "size_bytes": len(corpus),
            "canonical_rows_sha256": canonical_json_sha256(rows),
            "semantic_targets_sha256": canonical_json_sha256(
                [{"id": row["id"], "frames": row["frames"]} for row in rows]
            ),
        },
        "builder": _reference(Path(__file__)),
        "builder_dependencies": [_reference(Path(helpers.__file__), HELPERS_SHA256)],
        "retained_sources": [
            _reference(SOURCE_CORPUS, SOURCE_CORPUS_SHA256),
            _reference(SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        ],
        "metrics": helpers._metrics(rows),
        "provenance": {
            "authorship": "REPOSITORY_AUTHORED_SYNTHETIC",
            "reviewed_by": REVIEWER,
            "review_type": "SAME_PARTY_AGENT_REVIEW",
            "independent_human_review": False,
            "permission": PERMISSION,
            "ordinary_chat_training_data": False,
            "third_party_text": False,
            "retained_row_count": 197,
            "new_row_count": len(rows) - 197,
            "target_authorship": "DIRECT_TYPED_SEMANTIC_AUTHORING_NOT_PARSER_OUTPUT",
        },
        "separation": {
            "v7_evaluation_prompts_read": False,
            "v7_evaluation_builder_read": False,
            "held_out_vocabularies": vocabulary_references,
            "preserved_surface_exclusions": exclusion_references,
            "distinct_forbidden_surface_count": len(forbidden),
            "final_training_boundary": "trainer and independent experiment owner must verify every frozen suite including v7 before optimization",
        },
        "semantic_audit": {
            "target_identity": "CANONICAL_ORDER_INDEPENDENT_MULTISET_WITH_MULTIPLICITY",
            "group_policy": "EXACTLY_ONE_PARAPHRASE_GROUP_PER_TARGET",
            "new_semantic_atom_count": len(semantic_atoms()),
            "new_composition_group_count": len(_composition_selections()),
            "new_composition_renderings_per_group": 3,
            "single_frame_share_bounds": [0.35, 0.50],
            "numeric_and_polarity_slots_authored_explicitly": True,
            "labels_generated_from_runtime_parser": False,
        },
        "limitations": [
            "Synthetic bounded templates do not establish natural-language breadth.",
            "Same-party agent review is not independent human or external review.",
            "Exact surface and held-out-token separation do not prove semantic independence.",
            "Existing v6 lessons are retained training data, never fresh evaluation data.",
            "Training completion and loss are not promotion evidence.",
        ],
        "mutation_policy": "exclusive writes; preserve artifacts and create a new version for any later change",
    }
    return corpus, _json_bytes(manifest), manifest


def write_artifacts(output_directory: Path = OUTPUT_DIRECTORY) -> dict[str, Any]:
    corpus, manifest_bytes, manifest = build_artifacts()
    targets = {
        output_directory / CORPUS_NAME: corpus,
        output_directory / MANIFEST_NAME: manifest_bytes,
    }
    if any(path.exists() for path in targets):
        raise FileExistsError("v7 corpus artifacts exist; refusing to refresh")
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
                f"frozen v7 artifact does not replay byte-for-byte: {name}"
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

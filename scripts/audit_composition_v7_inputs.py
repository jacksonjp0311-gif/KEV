"""Freeze the final train/evaluation separation audit before the v7 run."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from kev.artifacts import canonical_json_sha256, file_sha256
from kev.uc51a2.semantic_breadth import (
    _frozen_evaluation_exclusions,
    _paraphrase_group_manifest,
    _pinned_evaluation_manifest,
    _reviewed_rows_from_text,
    required_held_out_vocabulary,
)


ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "training/reviewed/semantic-frame-paraphrases-v7-reviewed.jsonl"
SUITE = ROOT / "evals/frozen/public-audit-v7-260.json"
OUTPUT = ROOT / "evals/evidence/v7-input-separation.json"


def _normal(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def _template(text: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", "<N>", _normal(text))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CORPUS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    corpus_path = args.corpus.resolve()
    output_path = args.output.resolve()
    if output_path.exists():
        raise SystemExit("input audit already exists; refusing to refresh")
    corpus_bytes = corpus_path.read_bytes()
    corpus_text = corpus_bytes.decode("utf-8")
    corpus_sha256 = hashlib.sha256(corpus_bytes).hexdigest()
    suite_bytes = SUITE.read_bytes()
    suite_sha256 = hashlib.sha256(suite_bytes).hexdigest()
    source_sha256 = file_sha256(Path(__file__))
    manifest, _ = _pinned_evaluation_manifest()
    promotion = manifest["artifacts"]["promotion_suite"]
    if (
        ROOT / promotion["path"]
    ).resolve() != SUITE.resolve() or suite_sha256 != promotion["sha256"]:
        raise ValueError("input audit suite differs from the pinned v7 manifest")
    rows = [json.loads(line) for line in corpus_text.splitlines() if line.strip()]
    suite = json.loads(suite_bytes)
    if canonical_json_sha256(suite) != promotion["canonical_sha256"]:
        raise ValueError("input audit canonical suite hash mismatch")
    items = [item for split in suite["splits"].values() for item in split]
    forbidden, exclusions = _frozen_evaluation_exclusions()
    terms, vocabulary_artifacts, _ = required_held_out_vocabulary()
    exact = {_normal(text) for text in forbidden}
    exact.update(_normal(item["input"]) for item in items)
    templates = {_template(item["input"]) for item in items}
    failures = {
        "exact_train_eval_overlap": [
            row["id"] for row in rows if _normal(row["text"]) in exact
        ],
        "digit_normalized_train_eval_overlap": [
            row["id"] for row in rows if _template(row["text"]) in templates
        ],
        "held_out_vocabulary_overlap": [
            row["id"]
            for row in rows
            if any(
                re.search(rf"\b{re.escape(term)}\b", _normal(row["text"]))
                for term in terms
            )
        ],
    }
    try:
        reviewed = _reviewed_rows_from_text(
            corpus_text,
            held_out_terms=terms,
            forbidden_texts=forbidden,
        )
        groups = _paraphrase_group_manifest(reviewed)
    except ValueError as error:
        failures["reviewed_input_validation"] = [str(error)]
        groups = []
    report = {
        "schema": "kev.input-separation-audit.v1",
        "decision": "PASS" if not any(failures.values()) else "REJECT_BEFORE_TRAINING",
        "corpus_path": str(corpus_path),
        "corpus_sha256": corpus_sha256,
        "suite_sha256": suite_sha256,
        "audit_source_sha256": source_sha256,
        "exact_check_scope": "ALL_PRESERVED_FROZEN_SURFACES_NFKC_CASEFOLD_WHITESPACE",
        "digit_template_check_scope": "CURRENT_V7_SUITE_ONLY",
        "training_rows": len(rows),
        "evaluation_rows": len(items),
        "training_cardinalities": dict(
            sorted(Counter(str(len(row["frames"])) for row in rows).items())
        ),
        "reviewed_group_count": len(groups),
        "mandatory_forbidden_surface_count": len(forbidden),
        "exclusions": exclusions,
        "held_out_vocabulary_artifacts": vocabulary_artifacts,
        "failures": failures,
        "limitation": "Exact and number-template separation does not prove semantic independence; same-party synthetic authoring.",
        "model_inference_executed": False,
        "training_executed": False,
    }
    report["report_sha256"] = canonical_json_sha256(report)
    inputs = {
        corpus_path: corpus_sha256,
        SUITE: suite_sha256,
        Path(__file__): source_sha256,
        **{Path(record["path"]): record["sha256"] for record in exclusions},
        **{Path(record["path"]): record["sha256"] for record in vocabulary_artifacts},
    }
    for path, digest in inputs.items():
        if not path.is_absolute():
            path = ROOT / path
        if file_sha256(path) != digest:
            raise ValueError(f"audit input changed during validation: {path}")
    with output_path.open("xb") as handle:
        handle.write((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "decision",
                    "training_rows",
                    "evaluation_rows",
                    "training_cardinalities",
                    "reviewed_group_count",
                    "failures",
                )
            },
            indent=2,
        )
    )
    if report["decision"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()

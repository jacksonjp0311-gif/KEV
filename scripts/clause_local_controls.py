"""Post-hoc attribution controls, NOT a new promotion experiment."""
from __future__ import annotations

import json
from pathlib import Path

from kev.evaluation import evaluate_suite
from kev.frames import FRAME_KINDS, _claim_ranges, extract_frames
from kev.model_runtime import CheckpointPredictor, predict_proposal, semantic_frame
from kev.uc51a3.alive import AliveStore
from scripts.clause_local_experiment import BOUNDARY, ENCODER, OUTPUT, ROOT, digest, write
from scripts.verify_clause_local_evidence import verify

CONTROLS = ROOT / "experiments/clause-local-v1-controls"


def run():
    verify()
    if CONTROLS.exists():
        raise FileExistsError("preserve existing controls")
    CONTROLS.mkdir()
    write(CONTROLS / "plan.json", {
        "status": "POST_HOC_AFTER_PRIMARY_RESULTS_NOT_PREREGISTERED",
        "activation": "NONE", "training": "NONE",
        "motivation": "Single-clause-only training may make global cardinality the bottleneck. A permissive parser control tests whether learned kind selection is necessary for these templates.",
        "source_sha256": digest(Path(__file__)),
        "original_protocol_sha256": digest(BOUNDARY / "protocol.json"),
        "original_card_sha256": digest(OUTPUT / "research-card.json"),
        "controls": ["global_kinds_with_summed_local_cardinality", "all_kinds_with_clause_count_budget_no_model"],
    })
    store = AliveStore(CONTROLS / "ledger")
    store.append_event("RESEARCH_CONTROLS_STARTED", {"plan_sha256": digest(CONTROLS / "plan.json"), "activation": "NONE"})
    predictor = CheckpointPredictor(OUTPUT / "candidate/semantic-breadth.pt", encoder_manifest=ENCODER)

    def predict(item, mode):
        text = item["input"]
        clauses = [text[a:b] for a, b in _claim_ranges(text) if text[a:b].strip()]
        if mode == "global_kinds_with_summed_local_cardinality":
            global_proposal = predict_proposal(predictor.model, text)
            kinds = [row["kind"] for row in global_proposal["proposals"]]
            cardinality = sum(predict_proposal(predictor.model, clause)["cardinality"] for clause in clauses)
        else:
            kinds, cardinality = list(FRAME_KINDS), len(clauses)
        frames = extract_frames(text, proposal_result={"kinds": kinds, "cardinality": cardinality}, timestamp="1970-01-01T00:00:00Z")
        return {"predicted": [semantic_frame(frame) for frame in frames], "score_status": "UNCALIBRATED"}

    summary = {"status": "POST_HOC_DIAGNOSTIC_ONLY", "activation": "NONE", "reports": {}}
    for mode in ["global_kinds_with_summed_local_cardinality", "all_kinds_with_clause_count_budget_no_model"]:
        report = evaluate_suite(BOUNDARY / "suite.json", predictor=lambda item: predict(item, mode))
        path = CONTROLS / f"{mode}.json"
        write(path, report)
        summary["reports"][mode] = {"sha256": digest(path), "scores": {name: {key: value[key] for key in ["correct", "total"]} for name, value in report["splits"].items()}}
    write(CONTROLS / "summary.json", summary)
    store.append_event("RESEARCH_CONTROLS_COMPLETED", {"summary_sha256": digest(CONTROLS / "summary.json"), "activation": "NONE"})
    write(CONTROLS / "inventory.json", {"files": {str(path.relative_to(CONTROLS)).replace('\\', '/'): digest(path) for path in sorted(CONTROLS.rglob("*")) if path.is_file() and path.name != ".state.lock"}})
    print(json.dumps(summary))


if __name__ == "__main__":
    run()

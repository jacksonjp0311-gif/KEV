"""Non-activating clause-local proposal experiment.

No slots are learned here. The existing parser remains the only slot authority;
this adapter supplies kind/cardinality hints located at existing claim boundaries.
It is intentionally not wired into CheckpointPredictor or incumbent activation.
"""
from __future__ import annotations

from typing import Any, Callable, Mapping

from kev.frames import ProposalHeadResult, ProposalSpan, _claim_ranges, extract_frames
from kev.model_runtime import semantic_frame


def locate_proposals(
    text: str, head: Callable[[str], Mapping[str, Any]]
) -> tuple[ProposalHeadResult, list[dict[str, Any]]]:
    spans: list[ProposalSpan] = []
    records: list[dict[str, Any]] = []
    cardinality = 0
    for start, end in _claim_ranges(text):
        if not text[start:end].strip():
            continue
        proposal = head(text[start:end])
        count = proposal["cardinality"]
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError("invalid clause cardinality")
        selected = proposal["proposals"]
        local_spans = [
            ProposalSpan(row["kind"], start, end, float(row["score"]))
            for row in selected
        ]
        # A zero-cardinality clause must not donate kinds to another clause.
        if count:
            spans.extend(local_spans)
        cardinality += count
        records.append({
            "text": text[start:end], "start": start, "end": end,
            "cardinality": count, "proposals": [dict(row) for row in selected],
            "score_status": "UNCALIBRATED",
        })
    return ProposalHeadResult(
        kinds=tuple(dict.fromkeys(span.kind for span in spans)),
        cardinality=cardinality, spans=tuple(spans),
    ), records


class ClauseLocalPredictor:
    """Research-only callback with inspectable, uncalibrated local proposals."""

    def __init__(self, head: Callable[[str], Mapping[str, Any]], model_sha256: str):
        self.head = head
        self.model_sha256 = model_sha256
        self.trace: list[dict[str, Any]] = []

    def __call__(self, item: Mapping[str, Any]) -> dict[str, Any]:
        text = str(item.get("input", item.get("text", "")))
        proposal, clauses = locate_proposals(text, self.head)
        frames = extract_frames(
            text, source_type="EVALUATION", source_id=str(item["id"]),
            actor="clause-local-research", model_hash=self.model_sha256,
            proposal_result=proposal, timestamp="1970-01-01T00:00:00Z",
        )
        self.trace.append({
            "item_id": item["id"], "clauses": clauses,
            "frames": [frame.to_dict() for frame in frames],
        })
        predicted: list[Any] = (
            sorted(set(proposal.kinds)) if item.get("projection") == "kinds"
            else [semantic_frame(frame) for frame in frames]
        )
        return {
            "predicted": predicted, "confidence": None,
            "score_status": "UNCALIBRATED", "temperature": None,
            "model_sha256": self.model_sha256,
        }

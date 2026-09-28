# Clause-local v1: attribution experiment, not a promoted model

Completed 2026-09-28. **Incumbent unchanged.** This is experimental research,
not evidence of AGI. The frozen primary hypothesis was **NOT_SUPPORTED**.
The checkpoint, reviewed synthetic lessons, raw failures, training receipt,
exact inference replay, and hash-chained research events are retained.

Local validation: 304 tests passed, plus lint, type checking, artifact doctor,
historical tie rejection, and exact primary experiment replay. Command outcomes
are retained in `experiments/clause-local-v1-validation.json`.

## What changed

`kev/clause_proposals.py` calls the existing neural proposal head separately
on the existing parser's claim boundaries. It attaches kind hints to exact
offsets and sums local cardinalities. It never supplies slot values, deletes
parser frames, writes cognitive state, or activates checkpoints. Empty clauses
are skipped. Scores are explicitly UNCALIBRATED: an utterance-level calibration
receipt would not justify calibrated clause-level or aggregate scores.

This adapter is research-only, not wired into `kev evolve` or the active
`CheckpointPredictor`. Activating weights evaluated under a different runtime
would violate the evidence boundary, so this experiment cannot qualify a model.

## Frozen primary experiment

The protocol and all inference/training source bytes were hashed before training.
One seed (52103), 400 steps, 128 reviewed single-clause synthetic lessons,
64 contrastive positive pairs. MiniLM stays frozen; only projection/frame/
cardinality heads train, from the unchanged genesis parent hash. The actual
objective remains relation loss + 0.20 cardinality loss + 0.35 contrastive loss.
Training loss is recorded, never used as a promotion feature.

The suite contains 16 single-frame, 16 reserved-vocabulary, 56 composition,
and the existing 40 retention cases. Composition includes repeated kinds,
conflicting constraints, order swaps, number changes, negation, paraphrases,
and polite buried constraints. Targets were authored, not copied from parser
outputs. Review is **same-party synthetic**, not independent human review.
Training and evaluation share templates: this is a mechanism probe, not a
test of broad linguistic generalization. Variants are correlated, and the
reserved words are held out of KEV head training, not necessarily MiniLM's
original pretraining.

| Runtime/control | Fresh /16 | Retention /40 | OOV /16 | Composition /56 |
|---|---:|---:|---:|---:|
| Existing parser only | 0 | 40 | 0 | 0 |
| Genesis, global head | 5 | 40 | 4 | 4 |
| Genesis, clause-local head | 5 | 40 | 4 | 2 |
| Trained checkpoint, global head | 16 | 40 | 16 | 2 |
| Same trained checkpoint, clause-local head | 16 | 40 | 16 | 55 |
| Global kinds + summed local cardinality (post-hoc) | 16 | 40 | 16 | 19 |
| All kinds + clause-count budget, **no model** (post-hoc) | 16 | 40 | 16 | 56 |

Direct local kind/cardinality exactness is 79/88 utterances for the trained
head versus 1/88 for genesis; this is separate from parser-backed frame quality.
Eight negated clauses still receive positive neural kinds, though the parser
blocks them. The remaining frame failure is the polite constraint in
`local-v1-composition-030`; its complete trace is preserved, not patched.

The preregistered support rule required strict fresh **and** composition gains
over both parser-only and the same-checkpoint global runtime. Fresh tied, so
the result remains NOT_SUPPORTED. That rule was poorly suited to isolating
clause-locality on single-clause inputs; it was not relaxed after the result.
The ordinary numerical comparison against genesis says QUALIFY **diagnostically**
inside the card, but is not a production qualification: this is not the trusted
promotion suite/runtime, has no held-out calibration, and cannot activate.
No MODEL_QUALIFIED event or registry update occurred.

## What the additional controls establish

Controls were designed **after** seeing the primary result and are explicitly
post-hoc. Cardinality alone explains part of the gain. More importantly, a
permissive no-model kind proposal lets the existing constrained slot parsers
solve all these templates. Therefore 55/56 does **not** establish that learned
kind localization is necessary or superior. The parser-only row above means
the existing deterministic entry path, not every possible symbolic control.
The stronger symbolic control is reported rather than hidden.

Do not scale training on this result alone. A next experiment needs ambiguous
but slot-groundable clauses on which the symbolic control abstains, a frozen
independently reviewed boundary if available, and distinct gates for runtime
attribution versus promotion. Keep permissive proposals research-only until
false-positive and adversarial ambiguity audits justify any runtime change.

## Evidence and replay

- Boundary: `experiments/clause-local-v1-boundary/` (protocol, suite, localization
  targets, reviewed lesson JSONL).
- Evidence: `experiments/clause-local-v1-evidence/` (both three-way ablations,
  full proposal/frame traces, immutable candidate and embeddings, training/card/
  replay receipts, ledger, inventory).
- Additional controls: `experiments/clause-local-v1-controls/` (post-hoc plan,
  both raw reports, summary, separate ledger and inventory).
- Artifact identities are documented in [MODEL_WEIGHTS.md](../MODEL_WEIGHTS.md).

Artifact validation needs no encoder download:

```powershell
python -m scripts.verify_clause_local_evidence
python -m pytest -q tests_public/test_clause_proposals.py tests_public/test_clause_local_evidence.py
```

With the existing pinned local MiniLM artifacts installed, rerun inference:

```powershell
python -m scripts.verify_clause_local_evidence --inference
```

Use `--receipt PATH` for a new exclusively created verification receipt. The
original frozen driver's `freeze`, `run`, and `replay` commands intentionally
refuse to overwrite existing experiment artifacts; do not delete them to rerun.
Replay currently checks exact predictions and scores, so an unrecognized
numerical backend difference fails instead of silently widening tolerance.

This research suite is outside the production v7 manifest; the research runner
explicitly excludes its surfaces and reserved vocabulary before training.
It is **not automatically covered by the older production trainer's manifest**.
Before any future training campaign, carry this exposed suite and its vocabulary
into that campaign's exclusions. Never relabel it fresh or train on its failures.

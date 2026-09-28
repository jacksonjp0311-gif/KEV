# KEV composition experiment v7

This protocol is written before v7 checkpoint inference or challenger training.
The preceding implementation and rejected v6 experiment are preserved at Git
commit `1f5d9f322876d3e9ad7500b543521a64a3de5be1`.

## Hypothesis and intervention

V6 improved fresh and aggregate OOV scores but tied composition in all three
seeds. Development inspection found that global kind proposals were assigned
to available clauses by ordinal position, and a kind could be used only once.
V7 replaces that assignment with constrained matching across clauses, permits
repeated kinds, and abstains on incompatible or underdetermined assignments.
Exact values still come from source spans. The deterministic parser retains
authority, and model scores cannot resolve contradictory slot interpretations.

The reviewed corpus retains prior lessons and adds synthetic compositions,
repeated kinds, clause-order variants, and polite constraints. Both authoring
tasks are performed by agents in the same project. The evaluation author does
not read the new corpus or runtime changes; the corpus/runtime authors do not
read the new evaluation prompts. This separation is not independent human or
external evaluation. Ordinary chat and third-party text are excluded.

## Fixed decision protocol

- Freeze v7 evaluation and reviewed corpus before model inference or training.
- Preserve v1–v6 suites, reports, and every rejected checkpoint unchanged.
- Use all prior frozen surfaces and both held-out vocabulary lists as mandatory
  training exclusions. Check new train/eval exact and digit-normalized overlap.
- Capture and recheck runtime/source/input hashes around baseline measurement;
  close those identities and the resulting public evidence registry in the
  generated plan before challenger training.
- Measure the unchanged genesis checkpoint with the same v7 runtime used for
  every challenger. Attribute runtime changes separately from weight changes.
- Run seeds 52031, 52047, and 52069, each for 800 optimizer steps. Seed 52031 is
  the sole primary; the others are robustness checks and cannot replace it.
- Freeze the local encoder. Train only the projection and typed heads using
  `L_relation + 0.20 L_cardinality + 0.35 L_contrastive`.
- Fit temperature on the separate calibration-fit-v2 slice only.
- Require strict fresh and composition improvement, retention epsilon zero,
  no aggregate OOV regression, and no audit-family regression. A tie rejects.
- Record training loss as diagnostic evidence only.
- Retain all candidates, calibration receipts, eval cards, raw failures, and
  terminal ledger events. Independently replay checkpoint inference in the
  runner before its aggregate result is accepted.
- Do not tune sources, lessons, thresholds, seeds, or evaluation items after
  seeing v7 scores. Any later experiment needs a newly declared boundary.

Each seed uses an isolated state directory. Qualification updates that state's
checkpoint pointer through the existing gated evolution transaction. The
aggregate itself has no activation authority. A rejected experiment is still
published with its literal results; it is never described as weight improvement.

Git updates publish code and evidence. They do not override model qualification.

## Pre-training input rejection

The first cross-input audit rejected the 500-row v7 corpus before any model
inference or optimizer step. Three retained receipt lessons shared a
digit-normalized template with new evaluation inputs; exact overlap was zero.
The literal rejection is preserved at `evals/evidence/v7-input-separation.json`,
SHA-256 `d8e4374ee79e77b1a9279b238f97444023cb823e65266b109cf7e4461a37988c`.
V7 corpus bytes and evaluation bytes stay frozen. A separately versioned v8
corpus removes only those three lesson IDs and retains the remaining reviewed
paraphrases. The final plan may use it only after a new separation audit passes.
This change is based on input contamination, with no model scores available.

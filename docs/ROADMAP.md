# KEV Research Roadmap

KEV is experimental research software, not AGI or superintelligence. This roadmap separates capabilities present in v0.52 from hypotheses and product work that remain planned.

## Current — v0.52 research alpha

The following exists in the repository now.

### Compositional state

- Parser-first extraction can emit zero or more `GOAL`, `CONSTRAINT`, `OBSERVATION`, `PREDICTION`, or base-relation frames from one utterance.
- Frames preserve typed slots, exact claim boundaries, derivation, status, and provenance.
- The neural head proposes only frame kinds and cardinality. It cannot invent slot values, and an unparseable proposal is not persisted as state.
- Language-derived observations are `REPORTED`; only successful tool results with durable receipts become `VERIFIED`.

### Persistent evidence

- KEV Alive v2 stores facts and revisions, typed state groups, sessions, lesson records, prediction scores, active-model metadata, and model history.
- Atomic state writes, thread/process locking, a recoverable pending state/event journal, a sequence-bearing hash chain, and a separate co-located ledger-head anchor are implemented. Interrupted state/event pairs are completed exactly once on reopen; the anchor is still local and unsigned.
- Lesson entry creates `DRAFT` records. Explicit review adds reviewer, time, and permission provenance; only `REVIEWED` lessons are exported for training.
- The basic local desktop supports chat, session reset, lesson drafting, and explicit lesson review; it does not expose tools, evaluation, or model activation.

### Reproducible public baseline

- The repository includes an untrained, deterministic, `UNCALIBRATED` 285,092-parameter genesis checkpoint under `models/public/`.
- `models/registry.json`, `models/public/incumbent-manifest.json`, and `models/public/incumbent-evidence-v10.json` pin its identity and current evidence; the v10 evidence file SHA-256 is `69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21`.
- `evals/frozen/manifest-v6.json` pins the current 260-item promotion pack with fresh, retention, held-out-vocabulary, composition, and calibration splits. Its SHA-256 is `07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`; the suite file and canonical hashes are `a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029` and `3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`.
- The manifest retains the ten-term held-out list and separate label-clean, balanced 52-row calibration-fit slice. Both are forbidden weight-training inputs.
- `evals/evidence/v6-genesis-baseline.json` preserves the untouched genesis result, all 28 audit-family gates, and all 119 raw failures: fresh 48/80, retention 40/40, OOV 3/40, composition 50/80, and calibration 0/20. Its file SHA-256 is `881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4`.
- V1–v5 remain preserved development evidence. V6 replaced six v5 fresh items after a template-overlap audit, before any challenger training. The v5 audit finding and original suite remain immutable and non-current. Six parser probes that influenced repairs are also frozen separately with their literal pre-fix outputs and `promotion_eligible: false`.
- The incomplete historical 190/260 versus 190/260 aggregate is preserved and replayed as a rejection without fabricating missing item-level evidence or checkpoint hashes.

### Closed measured evolution

- Reviewed lessons train a new immutable challenger with a training receipt.
- Calibration fits a separate temperature artifact and writes a distinct calibrated checkpoint; it never alters the uncalibrated challenger.
- Training data is checked against the frozen promotion suite and held-out vocabulary.
- Incumbent and challenger are evaluated on the same hash-pinned suite, with raw failures and content hashes preserved.
- Promotion requires strict fresh improvement, retention within the declared epsilon, no held-out-vocabulary regression, strict composition improvement when applicable, and no regression in any audit family reported by either evaluation. Missing or size-mismatched audit-family metrics reject.
- Strict-gate ties reject. Training loss is excluded from promotion.
- Rejected artifacts remain on disk. Qualification commits `MODEL_QUALIFIED` and the active-model state through the recoverable state/event journal, then projects the derived local registry; the prior incumbent is preserved. A post-qualification recording failure is never mislabeled as rejection.
- Qualification is generation compare-and-swap protected: the evaluated incumbent path, checkpoint hash, and monotonic generation must still match at commit time. A stale run is rejected and preserved.

### Completed bounded encoder experiment — negative result

- The preregistered frozen-encoder plan is `experiments/frozen-encoder-v6-20260928-plan.json`, SHA-256 `c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`.
- Its 197-row corpus, SHA-256 `c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed`, is permission-clean, repository-authored synthetic data with same-party agent review. It has no independent human review, ordinary chat, or third-party text.
- Seed 52031 scored fresh 60/80, retention 40/40, OOV 16/40, composition 50/80, calibration 4/20. Seed 52047 scored 58/80, 40/40, 9/40, 50/80, 4/20. Seed 52069 scored 62/80, 40/40, 18/40, 50/80, 3/20.
- All three challengers were rejected for `COMPOSITION_TIE` and regression on `held-out-vocabulary:ameliorate`: the incumbent scored 3/4 and the challengers scored 0/4, 1/4, and 0/4.
- The immutable aggregate evidence has outcome `PRIMARY_REJECTED_NO_RECOMMENDATION`, file SHA-256 `c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`, and canonical integrity SHA-256 `82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.
- All seed runs completed, all rejected checkpoints, receipts, evaluation cards, raw failures, ledgers, and final inventories remain preserved, and the public incumbent remains SHA-256 `8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.
- The actual objective was relation loss plus `0.20 ×` cardinality loss plus `0.35 ×` contrastive loss. Loss histories are evidence, never promotion features.

### Bounded action research API

- A programmatic allowlist, schema boundary, state-derived action plan, canonical receipt, verified tool observation, prediction scorer, and grounded narrator are implemented.
- The default registry is empty, production mutation is rejected, and the Alive chat surface never executes tools.
- No general autonomous tool loop or bundled real-world adapter set is claimed.

## Next — harden the v0.52 product surface

1. Exercise clean-clone install, `Doctor`, tests, server startup, and CLI smoke flows continuously on supported Windows and Linux Python versions.
2. Extend the local desktop with frame inspection, exact claim/provenance display, and detailed ledger/model status.
3. Add session navigation, state search, correction history, export/backup, and candidate/evaluation comparison without hiding the underlying artifacts.
4. Make generated evolution-run evidence bundles path-portable and add an optional external or signature-backed ledger anchor without weakening local-first operation.
5. Expand migration and adversarial corruption tests; crash recovery, concurrent ledger writes, server boundaries, and end-to-end rejected evolution are already exercised publicly.
6. Define signed release packaging and recovery procedures while retaining local-first defaults.

Release gate: a new user can install a clean clone, verify all shipped hashes, reproduce the public baseline, inspect one multi-frame state update, review/export a lesson, and run a preserved challenger rejection without undocumented inputs.

## Then — improve semantic breadth without weakening evidence

1. Expand parser coverage and adversarial paraphrase tests while preserving exact-value binding and negation/order sensitivity.
2. Treat the first frozen local sentence-encoder benchmark as complete and negative. Do not promote any of its challengers or reinterpret fresh/OOV gains as success.
3. Preregister any next experiment against a newly frozen, untouched suite and a newly versioned corpus boundary. V6 items and failures have influenced development and must never be patched, relabeled, or reused as fresh promotion evidence.
4. Target the observed deficiencies directly: multi-frame/composition quality and robust handling of `ameliorate`, while keeping parser-grounded slots and generated text outside state authority.
5. Train and calibrate future public challengers only from provenance-bearing reviewed data, and measure multi-frame exact match, slot accuracy, calibration, held-out vocabulary, composition, retention, and false state writes.
6. Version frozen packs instead of editing an audit after it influences development.

Research gate: demonstrate improvement on a newly frozen fresh/composition evaluation with no disallowed retention or held-out-vocabulary regression. Lower loss, more parameters, or a higher version number do not satisfy this gate.

## Then — state-conditioned reasoning and bounded action

1. Add explicit conflict, dependency, and temporal reasoning over goals, constraints, observations, predictions, facts, and evidence.
2. Expose action planning and execution through deliberate CLI/UI workflows with clear user authority boundaries.
3. Ship a small set of read-only or sandboxed reference adapters, each with declared schemas and deterministic receipt behavior.
4. Connect later verified observations to prediction resolution and longitudinal calibration reports.
5. Keep arbitrary generated prose non-authoritative unless supported by explicit state and receipts.

Research gate: complete bounded tasks from structured state while obeying constraints, producing a durable receipt for every attempted action, and never converting an unreceipted result into a verified observation.

## Then — a usable continual-learning workbench

1. Build a state/evidence timeline that makes revisions, conflicts, sources, and verification status inspectable.
2. Add a lesson review queue with provenance and permission checks.
3. Add immutable experiment and candidate comparison views for parent lineage, training data hashes, calibration, split metrics, raw failures, and promotion decisions.
4. Support explicit rollback of the active pointer without deleting any qualified, rejected, or historical artifact.
5. Add encrypted local backups and schema-aware import/export before considering optional synchronization.

Product gate: a skeptical reviewer can answer what changed, why it changed, what evidence supported it, what failed, and which exact checkpoint is active without reading source code.

## Longer-term research

- Broader and multilingual semantics using permission-compatible data and documented teacher distillation.
- Transfer across task families without catastrophic forgetting or audit contamination.
- A stable SDK for typed state and bounded adapters.
- Reproducible benchmark packs maintained independently from training data.
- External replication of capability, robustness, calibration, and retention results.
- Optional multi-device or team workflows only after local provenance and recovery guarantees are mature.

These are research directions, not current capabilities or delivery promises.

## Evidence standard

Every stage preserves the same invariants:

- never overwrite an incumbent checkpoint;
- training completion is not promotion;
- ordinary conversation is not automatically truth or training data;
- reviewed lessons are explicit and provenance-bearing;
- once an audit influences a change, preserve it as development evidence;
- separate memory improvement from weight improvement;
- never claim a tool ran without a durable receipt; and
- preserve raw failures and rejected candidates.

Claims of increasingly general intelligence require breadth, transfer, retention, calibration, robustness, reproducibility, and independent evaluation. Version numbers, fluent output, stored memories, tool access, and lower training loss are not substitutes.

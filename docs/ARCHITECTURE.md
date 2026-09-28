# KEV v0.52 Architecture

KEV v0.52 is an experimental, local-first cognitive-runtime research system. It
implements typed compositional frames, explicit persistent state, bounded
receipt-bearing actions, immutable model candidates, and evidence-backed
promotion gates. It is not AGI or superintelligence, and none of its version
numbers, losses, scores, or state records should be described as such.

## Architectural invariants

1. **Language is an interface, not cognitive state.** Raw prose can be retained
   as a claim boundary, but only validated typed frames and explicit fact
   operations enter state.
2. **Neural output has no state authority.** The semantic model may propose
   frame kinds and cardinality. It cannot invent slot values, delete parser
   results, execute tools, mutate state directly, or activate a checkpoint.
3. **Exact values remain grounded in source spans.** Slot parsers copy and
   normalize values from bounded claims while preserving the original spelling,
   offsets, utterance hash, actor, timestamp, and model hash when present.
4. **Language reports are not verified observations.** An observation extracted
   from language is `REPORTED`. In the integrated action path, a successful
   allowlisted tool action can create a `VERIFIED` observation only after its
   receipt is accepted by the ledger.
5. **Actions start from explicit state.** The planner reads only facts, goals,
   constraints, observations, predictions, and reviewed lessons. Free-form chat
   is never parsed into a hidden command language.
6. **Tools are deny-by-default and receipt-bearing.** The registry starts empty,
   inputs and outputs are schema-checked, production-mutation tools are rejected,
   and every attempted action constructs a success, rejection, or failure
   receipt that must be accepted before its result can become verified state.
7. **Checkpoints are immutable.** Training and calibration create new artifacts;
   they do not overwrite or silently activate their parents.
8. **Training completion is not promotion.** Qualification requires the same
   frozen suite, strict fresh improvement, strict composition improvement when
   that split is present, bounded retention, no OOV regression, and no
   per-audit-family regression. Missing or size-mismatched audit-family metrics
   reject. Training loss is not a promotion feature.

## Implemented topology

```text
language / CLI / local web surface
                 |
                 v
       deterministic fact routes
                 or
       compositional frame extraction
          /                     \
 parser-first claim/slot logic   neural kind + cardinality proposals
          \                     /
           validated typed frames
                 |
                 v
      explicit state + session log
                 |
                 +------------------------+
                 |                        |
                 v                        v
        grounded narration        structured GOAL planner
                                          |
                                  allowlisted ToolSpec
                                          |
                                  schema-checked execution
                                          |
                             receipt persisted to event ledger
                                          |
                                VERIFIED observation
                                          |
                                  prediction scoring

reviewed lessons -> immutable challenger -> optional calibration
                                      -> frozen same-suite evaluation
                                      -> reject and preserve
                                         or qualify and repoint registry
```

The executable boundaries live in
[`kev/frames.py`](../kev/frames.py),
[`kev/actions.py`](../kev/actions.py),
[`kev/uc51a3/alive.py`](../kev/uc51a3/alive.py), and
[`kev/evolution.py`](../kev/evolution.py). The precise model and gate equations
are documented in [Mathematics](MATHEMATICS.md).

## 1. Language and compositional frames

`extract_frames()` is parser-first and set-valued: one utterance can yield zero,
one, or several frames. v0.52 supports the nine inherited relation kinds plus
four cognitive kinds:

```text
COPY_VALUE · SUPERSEDES · MAGNITUDE · NEGATE · REFERENCE
ACTIVE_SELECTION · RUN_STATUS · RECEIPT_VALUE · EVIDENCE_CONSISTENCY
GOAL · CONSTRAINT · OBSERVATION · PREDICTION
```

Each `kev.frame.v1` record keeps these concerns separate:

- `kind` and `relation` describe meaning;
- typed `slots` hold canonical values and their raw source spelling;
- `claim` identifies the exact half-open character range that supports it;
- `provenance` records source, actor, time, utterance SHA-256, and optional
  model or receipt hashes;
- `status` distinguishes reported language from verified tool evidence;
- `derivation` distinguishes deterministic parsing from a neural proposal that
  was completed by a constrained slot parser.

Deterministic parser results always survive. A neural proposal is considered
only when it asks for more frames than the parsers found, and it becomes a frame
only if a bounded slot parser can recover the required values from the proposed
claim. A high proposal score by itself never becomes state.

Example:

```text
Reduce latency below 50 ms, do not modify production,
and the last measured latency was 73 ms.
```

produces three independent frames: a `GOAL` threshold, a negative
`CONSTRAINT` on production modification, and a `REPORTED` `OBSERVATION`. The
original claim spans remain attached, so conflicts are preserved rather than
merged away.

The active public semantic front-end is a small hashed-feature neural model. It
proposes multiple frame kinds and a cardinality; it does not supply slots. The
optional frozen-encoder interface in [`kev/encoders.py`](../kev/encoders.py)
preserves the same authority boundary but is not the active public incumbent.
Its first allowlisted artifact is the Apache-2.0
`sentence-transformers/all-MiniLM-L6-v2` encoder at commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its local-only manifest
validator requires approved permission provenance, a license identifier, an
embedding dimension, and hashes for every contained model file before loading
the exact local safetensors and tokenizer bytes. The runtime performs no
network fetch or dynamic remote-code load. Acquisition is a separate explicit,
receipted command documented in [Frozen Encoder](FROZEN_ENCODER.md).
Current hashed-model challenger training freezes its feature encoder and
updates only its final projection and the two proposal heads. The frozen
sentence-encoder path precomputes verified local features and likewise trains
only a deterministic 384-to-80 projection and the two proposal heads. Its
small checkpoint binds the external manifest hash and parent lineage; encoder
weights remain external and frozen. A deterministic source manifest also binds
the sentence trainer, encoder runtime, and shared semantic target/loss source;
the closed evolution gate independently re-hashes those files.

That interface has now been exercised in one bounded, preregistered v6
experiment. Three independently seeded challengers improved fresh and aggregate
OOV exact match, but all three tied the incumbent on composition and regressed
the `held-out-vocabulary:ameliorate` audit family. All were rejected. This is a
negative experimental result, not evidence that the encoder has broader state
authority; the hashed-feature genesis checkpoint remains the public incumbent.

## 2. Explicit persistent state

`AliveStore` persists `kev.alive-state.v2` with separate compartments for:

- current facts and their revision metadata;
- all typed frames plus indexed goals, constraints, observations, predictions,
  and inherited relations;
- draft and reviewed lessons;
- prediction scores;
- active model metadata and model history.

The local state directory also contains per-session JSONL transcripts, a
sequence-numbered hash-linked event ledger, and a cached ledger-head anchor.
State replacement is atomic at the file level. In-process re-entrant locks and
a cross-process file mutex serialize mutations. A durable
`pending-state-event.json` journal binds a projected state hash to its intended
event; if a process stops between the two filesystem writes, the next store
open completes the state/event pair exactly once. This supplies crash recovery,
not a claim of database-level cross-file atomicity.

Earlier state is migrated to the v2 envelope, but old prose-only records are
not silently promoted into structured frames. Fact corrections create a new
revision and retain the superseded value and revision in evidence.

The ledger detects broken sequences, altered event hashes, and disagreement
with its local head anchor. It is a local integrity chain, not an externally
signed or independently anchored audit service.

## 3. Bounded actions and receipts

The action layer is independent of the language model and storage
implementation:

- `ToolRegistry` is an explicit allowlist and is empty by default.
- `ToolSpec` declares input/output schemas and whether a tool requests
  production mutation.
- `StatePlanner` accepts only structured tool requests already present in a
  `GOAL`; it ignores free-form goal text and non-explicit state.
- `ActionPlan` binds a tool request to a canonical hash of the explicit state.
- `ActionExecutor` validates input, rejects production mutation, invokes the
  handler, validates output, and hashes a receipt for every attempt.

The ledger sink receives the receipt before a successful result can be exposed
as a `VERIFIED` observation. If receipt persistence fails, the executor raises
with the raw receipt and emits no verified observation. Rejected and failed
attempts remain evidence but do not produce observations. A successful receipt
may authorize only one verified observation; replaying even the exact result is
rejected.

At the lower-level frame API, `extract_frames()` treats `source_type="tool"`
plus a syntactically valid SHA-256 receipt reference as caller-supplied
provenance. Callers that need the stronger durability guarantee must use the
integrated executor/store path above.

After a verified observation is stored, structured predictions can be matched
by prediction identifier or exact subject and scored only against a later
observation. Results are `CONFIRMED`, `REFUTED`, or `UNRESOLVED` and retain the
observation and receipt reference.

The shipped Alive chat surface does not register or execute tools; its turn
records explicitly report `tool_execution: NONE`. Tool execution is available
as a bounded application interface for callers that deliberately construct a
registry.

## 4. Language output

`StateNarrator` can deterministically serialize an explicit state snapshot and
receipts as `GROUNDED` output. Arbitrary supplied prose is labeled `UNGROUNDED`
and returns no fact updates. Fluency alone is therefore never evidence and has
no write authority.

## 5. Reviewed learning and model evolution

Teaching has an explicit lifecycle:

```text
DRAFT lesson
  -> human review with reviewer, time, and permission provenance
  -> REVIEWED export with SHA-256
  -> immutable challenger + training receipt
  -> optional immutable temperature-calibrated child + receipt
  -> incumbent and challenger evaluated on the same frozen suite
  -> REJECT or QUALIFY
```

Training rejects unreviewed rows, rows without permission provenance, and rows
containing the repository's pinned held-out vocabulary on every training entry
point; callers may add exclusions but cannot disable the pinned list. The
trainer also loads every preserved frozen public suite, calibration-fit
artifact, and development-influenced known-failure audit and rejects exact
surface overlap, so direct calls cannot bypass the
retention/promotion exclusion. The evolution pipeline independently repeats the
current-suite separation check. Every candidate run receives
a new directory; checkpoint and receipt creation use exclusive writes.

Contrastive positives are never inferred from a shared broad frame kind. They
must belong to an explicit reviewed `paraphrase_group` whose rows have an
identical canonical multiset of exact kinds, relations, and typed slot values. Invalid
or singleton groups reject before checkpoint creation. Repeated kinds count as
separate frames for the cardinality target. Ungrouped reviewed lessons may
supervise the typed heads but cannot become positive contrastive pairs.

The training receipt records parent, lesson, held-out-vocabulary, and trainer
source hashes; paraphrase-group and semantic-target hashes; positive-pair
counts; Python and Torch versions; seed; step count; objective; full loss
history; challenger hash; `activation: NONE`; and an empty list of promotion
features. Calibration changes no model weights and creates a separate
checkpoint and receipt from a calibration-fit slice whose identifiers and
surface forms are disjoint from the promotion suite. The receipt's calibrated
output hash must match the evaluated checkpoint, and its promotion-suite file
or canonical hash must match the evaluated frozen suite. Checkpoint temperature
metadata without this bound receipt does not upgrade evidence from
`UNCALIBRATED`; a generic scalar is insufficient. Predictor scores that already
used the bound temperature are not scaled again.

Incumbent and challenger reports retain per-item predictions, raw failures,
checkpoint hashes, suite hashes, calibration status, split and audit-family
metrics, and a canonical report hash. A challenger qualifies only if every
applicable gate passes. On rejection, all candidate bytes and evidence remain
on disk. On qualification, the old incumbent remains in history and the local
state and registry are repointed to the verified challenger hash.

The preregistered runner adds a second verification layer around this lifecycle.
Its immutable plan closes the input and executable-source inventories before
training. Each training receipt binds the exact source manifest for the frozen
encoder trainer, encoder runtime, and shared semantic loss implementation.
After each run, the runner reloads the incumbent and calibrated challenger,
independently reproduces the complete evaluation card, validates the
calibration receipt against its source checkpoint, calibrated checkpoint,
calibration-fit bytes, and promotion-suite hashes, and inventories every final
evidence file. It then revalidates all seed inventories and common non-seed
training inputs before writing aggregate evidence.

Qualification also uses a generation-bearing compare-and-swap. The expected
incumbent is the exact `(path, SHA-256, generation)` snapshot evaluated by the
run. A qualification commits only if all three still match the current state
and the checkpoint bytes still match the hash; success increments the
generation. A stale run is rejected and preserved rather than overwriting a
newer incumbent.

## 6. Replayable public artifacts

The public genesis model is a deterministic, untrained bootstrap artifact. Its
presence makes a clean checkout replayable; it is not evidence of learning or
general intelligence, and its scores are explicitly uncalibrated.

| Role | Replayable artifact |
|---|---|
| Active model registry | [`models/registry.json`](../models/registry.json) |
| Incumbent manifest | [`models/public/incumbent-manifest.json`](../models/public/incumbent-manifest.json) |
| Incumbent evidence manifest | [`models/public/incumbent-evidence-v10.json`](../models/public/incumbent-evidence-v10.json) |
| Frozen v6 artifact manifest | [`evals/frozen/manifest-v6.json`](../evals/frozen/manifest-v6.json) |
| Current 260-item promotion suite | [`evals/frozen/public-audit-v6-260.json`](../evals/frozen/public-audit-v6-260.json) |
| Separate calibration-fit slice | [`evals/frozen/calibration-fit-v2.jsonl`](../evals/frozen/calibration-fit-v2.jsonl) |
| Held-out vocabulary | [`evals/frozen/held-out-vocabulary-v2.txt`](../evals/frozen/held-out-vocabulary-v2.txt) |
| Development-only parser failures | [`evals/frozen/frame-parser-known-failures-v1.json`](../evals/frozen/frame-parser-known-failures-v1.json) |
| Public genesis baseline | [`evals/evidence/v6-genesis-baseline.json`](../evals/evidence/v6-genesis-baseline.json) |
| Frozen-encoder experiment plan | [`experiments/frozen-encoder-v6-20260928-plan.json`](../experiments/frozen-encoder-v6-20260928-plan.json) |
| Frozen-encoder aggregate evidence | [`experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`](../experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json) |
| Historical 190/260 tie replay | [`evals/frozen/historical-190-260-replay.json`](../evals/frozen/historical-190-260-replay.json) |

The registry pins the current public genesis checkpoint to SHA-256
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.
The current `incumbent-evidence-v10.json` has SHA-256
`69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21`.
The current v6 manifest has SHA-256
`07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`.
It pins promotion-suite file SHA-256
`a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029`
and canonical SHA-256
`3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`.
V6 replaced six v5 fresh items whose templates repeated earlier generations;
the v5 freshness finding, suite, baseline, and manifest remain preserved as
development evidence. V1–v5, their baseline reports, raw failures, and the
development-influenced parser probes are never silently refreshed.

The completed frozen-encoder experiment plan has SHA-256
`c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`.
Its aggregate evidence has file SHA-256
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`
and canonical integrity SHA-256
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.
The outcome is `PRIMARY_REJECTED_NO_RECOMMENDATION`: all three seed runs are
complete, all three ledger decisions are `MODEL_REJECTED`, every candidate and
failure remains inventoried, and no active-model pointer changed.

## Current claim boundary

v0.52 demonstrates a bounded architecture for compositional parsing, explicit
state, receipt-backed local tools, prediction scoring, and gated model
evolution. Its parser coverage is finite, the public genesis neural head is
untrained and uncalibrated, the default tool registry is empty, and the
file-backed store is not a distributed transactional database. These mechanisms
are research infrastructure, not evidence of AGI or superintelligence.

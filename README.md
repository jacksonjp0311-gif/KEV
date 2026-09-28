# KEV

**A local-first research system for explicit cognitive state, reviewed teaching, bounded action, and evidence-gated model evolution.**

> **Status: v0.52 research alpha.** KEV is experimental research software. It is not AGI or superintelligence, and the current public measurements are not evidence of either. Successful and failed experiments are preserved so every capability claim can remain tied to inspectable evidence.

## Why KEV exists

Most assistants treat a conversation as text to continue. KEV explores a different architecture: **language is an interface to cognitive state, not the state itself**.

KEV separates:

- language from typed state;
- a neural proposal from parser-validated values;
- a reported statement from a verified observation;
- persistent memory from changed model weights;
- training completion from model qualification; and
- a claimed action from a receipted tool result.

The research question is whether those separations can support persistent, correctable, and measurable learning without mistaking fluent output, stored text, tool access, or lower training loss for genuine capability improvement.

## Architecture

```text
language
   ↓
parser-first compositional frames
 kinds · typed slots · exact claim spans · provenance
   ↓
explicit persistent state
 facts · revisions · goals · constraints · observations · predictions
   ↓
state-conditioned planning → allowlisted tool → durable receipt
                                              ↓
                                      verified observation
                                              ↓
reviewed lessons → immutable challenger → calibration → frozen evaluation
                                                   ↙              ↘
                                                reject           qualify
```

The model may propose frame kinds and cardinality. It cannot supply slot values. Slot values must be bound from exact source spans by constrained parsers before they can enter state. Language-derived observations are `REPORTED`; only a successful tool result backed by a persisted receipt becomes `VERIFIED`.

## Implemented in v0.52

### Compositional cognitive frames

`kev.frames.extract_frames` can emit zero or more typed frames from one utterance. Each frame keeps its kind, relation, typed slots, exact claim boundary, derivation, status, and provenance separate.

For example:

> Reduce latency below 50 ms, do not modify production, and the last measured latency was 73 ms.

is parsed into a `GOAL`, a `CONSTRAINT`, and an `OBSERVATION`, with the values `50`, `production`, and `73` copied from their supporting claims. The current parser covers bounded documented patterns; it is not a general language-understanding system.

The v0.52 challenger objective is:

```text
L = L_relation + 0.20 L_cardinality + 0.35 L_contrastive
```

The neural heads are proposal mechanisms only. Parser results take precedence, and a proposal without parser-backed slots does not become state. Challenger training freezes the feature encoder and updates only its final projection plus the frame-kind and cardinality heads. Contrastive positive pairs require an explicit reviewed paraphrase group with identical exact frame relations and slot values; sharing a broad kind is insufficient, and repeated kinds still count separately for cardinality. `kev.encoders` provides a verified local-only manifest boundary for a permission-compatible frozen sentence encoder: license and review provenance plus every declared file hash are checked before trusted caller code may load it, and only the projection and typed heads remain trainable. The first selected feature source is the Apache-2.0 `sentence-transformers/all-MiniLM-L6-v2` artifact at an exact upstream commit; its large bytes remain local and its [acquisition and offline verification contract](docs/FROZEN_ENCODER.md) is hash-pinned. No external model is downloaded implicitly or at runtime. A frozen encoder is a feature interface, not cognition. The public incumbent does not include this encoder, so selection, integration, or reduced training loss are not broader-language capability claims.

### Persistent state and reviewed lessons

KEV Alive stores correction-aware facts, structured frames, sessions, lesson records, prediction scores, model history, and a hash-linked event ledger locally. State mutations use atomic writes, process/thread locking, and a durable pending state/event journal; after an interrupted cross-file commit, the next store open completes the same state/event pair exactly once. This is recoverability, not a claim of database-level atomicity. The co-located ledger-head anchor detects altered, reordered, or removed events and byte-size changes, but not same-size formatting-only rewrites; it is not an external, signature-backed adversarial anchor.

Teaching is now an explicit lifecycle:

```text
DRAFT → REVIEWED → export → train challenger
```

A lesson is not trainable merely because it was entered. Review requires reviewer and permission provenance, and exports contain only `REVIEWED` lessons.

### Public, hash-pinned evidence

A clean clone includes a deterministic, untrained public genesis checkpoint:

```text
models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt
```

Its SHA-256 is pinned in `models/registry.json`. The checkpoint has 285,092 parameters, no training data or training steps, and an `UNCALIBRATED` score status. It is a reproducible bootstrap artifact, not evidence of learning.

The current public evidence chain is `models/public/incumbent-evidence-v10.json` (file SHA-256 `69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21`), `evals/frozen/manifest-v6.json` (file SHA-256 `07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`), and `evals/frozen/public-audit-v6-260.json` (file SHA-256 `a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029`; canonical JSON SHA-256 `3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`). The v6 pack was authored and hash-frozen before the preregistered frozen-encoder challenger training, without reading its lesson corpus. Its fresh, retention, held-out-vocabulary, composition, and calibration splits contain distinct natural-language surfaces, plus single-frame, multi-frame, buried and polite constraints, paraphrase equivalence, conflicting constraints, negation, number-change, and order-swap audits. A separate label-clean 52-row temperature-fit slice and ten-term held-out vocabulary are also frozen. V1–v5 and every earlier result remain unchanged development evidence. The untouched public genesis baseline, including all 28 audit-family gates and all 119 raw failures, is preserved in `evals/evidence/v6-genesis-baseline.json` (file SHA-256 `881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4`):

| Split | Exact match |
|---|---:|
| Fresh | 48/80 |
| Retention | 40/40 |
| Held-out vocabulary | 3/40 |
| Composition | 50/80 |
| Calibration | 0/20 |

These literal baseline measurements expose substantial limitations, especially on unfamiliar vocabulary. They are not general-intelligence claims.

Earlier v1 through v5 suites and baseline cards remain unchanged. The v3 lineage documents the discovered label leakage instead of being silently corrected; v4 preserves the label-clean repair and its audit-gated baseline; and v5 preserves the freshness-audit finding that preceded v6. Six parser failures that influenced the accompanying parser repair are separately frozen in `evals/frozen/frame-parser-known-failures-v1.json`, explicitly marked `promotion_eligible: false`, and never counted in the v6 score.

### Measured model evolution

The v0.52 evolution path:

1. accepts only explicitly reviewed lessons with provenance;
2. rejects overlap with every preserved frozen promotion/calibration surface and development-influenced known-failure audit, and always enforces the pinned held-out vocabulary (callers may only add exclusions);
3. trains a new challenger without overwriting its parent;
4. writes a training receipt and preserves the uncalibrated challenger;
5. optionally writes a distinct temperature-calibrated checkpoint and receipt;
6. evaluates incumbent and challenger on the same frozen suite;
7. preserves raw failures and the evaluation card; and
8. records `MODEL_REJECTED` or `MODEL_QUALIFIED` in the ledger.

A challenger qualifies only if every applicable gate passes:

- strict improvement on fresh items;
- retention regression no greater than the declared epsilon (`0` by default);
- no held-out-vocabulary regression;
- strict composition improvement when composition items are present; and
- no regression in any audit family reported by either evaluation, with missing or
  size-mismatched family metrics rejecting.

A tie on a strict gate rejects. Training loss is recorded but is never a promotion feature. Qualification updates only the local active-model pointer; it does not overwrite the old checkpoint.

#### Completed preregistered frozen-encoder experiment

The frozen-encoder v6 experiment was preregistered in `experiments/frozen-encoder-v6-20260928-plan.json` (file SHA-256 `c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`). Its retained aggregate is `experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json` (file SHA-256 `c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`; canonical integrity SHA-256 `82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`). All three seed bundles are `COMPLETE`, and all three decisions are `REJECT`:

| Role / seed | Fresh | Retention | Held-out vocabulary | Composition | Calibration | Decision |
|---|---:|---:|---:|---:|---:|---|
| Frozen genesis baseline | 48/80 | 40/40 | 3/40 | 50/80 | 0/20 | Reference |
| Primary / 52031 | 60/80 | 40/40 | 16/40 | 50/80 | 4/20 | `REJECT` |
| Robustness / 52047 | 58/80 | 40/40 | 9/40 | 50/80 | 4/20 | `REJECT` |
| Robustness / 52069 | 62/80 | 40/40 | 18/40 | 50/80 | 3/20 | `REJECT` |

Each challenger tied the baseline on composition and regressed on the held-out-vocabulary `ameliorate` audit family, yielding `COMPOSITION_TIE` and `AUDIT_HELD_OUT_VOCABULARY_AMELIORATE_REGRESSED`. The aggregate outcome is `PRIMARY_REJECTED_NO_RECOMMENDATION`: no challenger qualified, and no promotion was recommended. Training loss declined, but the preregistered protocol excluded it from promotion input. The 197-row training corpus is synthetic, same-party reviewed material, not independent human review. The public registry did not change; its incumbent remains the genesis checkpoint with SHA-256 `8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`. Challenger checkpoints, evaluation cards, and raw failures remain preserved under the experiment evidence directory. This is a bounded rejected experiment, not an AGI claim.

The historical v0.51 result—parent 190/260 and challenger 190/260—is preserved as aggregate-only evidence at `evals/frozen/historical-190-260-replay.json`. It replays as a rejection, while explicitly recording that the original item-level data and checkpoint hashes were not published.

### Bounded action primitives

`kev.actions` implements an explicit tool allowlist, input/output schema validation, plans derived only from structured goals, canonical receipts, receipt-backed observations, prediction scoring, and grounded state narration.

This is a programmatic research API, not an autonomous agent product. The default tool registry is empty, production-mutating tools are rejected, no bundled real-world adapters ship in v0.52, and ordinary Alive chat never executes a tool (`tool_execution: NONE`).

## Quick start

KEV requires Python 3.11–3.13. The repository provides a PowerShell entry point and creates `.kev-venv` locally.

```powershell
.\KEV.ps1 -Mode Install
.\KEV.ps1 -Mode Doctor
.\KEV.ps1 -Mode AliveStatus
.\KEV.ps1 -Mode AliveVerifyLedger
```

`Doctor` starts at the public registry, verifies its pinned incumbent-evidence
bytes, and follows that evidence to the checkpoint, frozen manifest, suite
file and canonical hashes, eval-card file and report hashes, excluded known
failures, predecessor evidence, and every manifest artifact.

Extract compositional frames without mutating state:

```powershell
.\.kev-venv\Scripts\python.exe -m kev.cli frames --message "Reduce latency below 50 ms, do not modify production, and the last measured latency was 73 ms."
```

Use the persistent runtime or open the basic local browser surface:

```powershell
.\KEV.ps1 -Mode AliveChat -Message "Remember that project codename is Lumen"
.\KEV.ps1 -Mode AliveDesktop
```

The desktop is a small local chat/teaching surface with explicit lesson drafting and review. Evaluation and evolution remain CLI-first, and the desktop cannot execute tools or activate checkpoints.

### Review and evolve a lesson

Create a draft lesson:

```powershell
.\KEV.ps1 -Mode AliveTeach -StateDir .\experiment-state -Intent CONSTRAINT -Text "Never overwrite an incumbent checkpoint in place."
```

Review the returned lesson ID with explicit provenance:

```powershell
.\KEV.ps1 -Mode AliveReview -StateDir .\experiment-state -LessonId "lesson-..." -ReviewedBy "reviewer-name" -Permission "author-approved-for-training"
```

Export only reviewed lessons, then run the complete challenger gate:

```powershell
.\.kev-venv\Scripts\python.exe -m kev.uc51a3.cli --state-dir .\experiment-state export-lessons --output .\reviewed-lessons.jsonl
.\KEV.ps1 -Mode Evolve -StateDir .\experiment-state -Lessons .\reviewed-lessons.jsonl
```

Use the same state directory for all four commands (`--state-dir` on the direct
Python command and `-StateDir` on `KEV.ps1`) to keep an experiment isolated.
`Evolve` is the preferred end-to-end path. `AliveTrain` is a lower-level
command that creates an unactivated challenger but does not evaluate or qualify
it.

Evaluate two existing checkpoints without activation:

```powershell
.\KEV.ps1 -Mode Eval `
  -Incumbent .\path\to\incumbent.pt `
  -Challenger .\path\to\challenger.pt `
  -Suite .\evals\frozen\public-audit-v6-260.json
```

Replay the historical aggregate tie:

```powershell
.\.kev-venv\Scripts\python.exe -m kev.cli historical-replay
```

Run the public validation suite:

```powershell
.\.kev-venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.kev-venv\Scripts\python.exe -m pytest -q -p no:cacheprovider tests_public
```

## Artifact locations

The public source-of-truth artifacts are:

- model registry: `models/registry.json`;
- incumbent manifest: `models/public/incumbent-manifest.json`;
- incumbent evidence: `models/public/incumbent-evidence-v10.json`;
- frozen evaluation manifest: `evals/frozen/manifest-v6.json`;
- current promotion suite: `evals/frozen/public-audit-v6-260.json`;
- separate calibration-fit data: `evals/frozen/calibration-fit-v2.jsonl`;
- held-out vocabulary: `evals/frozen/held-out-vocabulary-v2.txt`;
- development-only parser failures: `evals/frozen/frame-parser-known-failures-v1.json`; and
- public baseline report: `evals/evidence/v6-genesis-baseline.json`;
- preregistered frozen-encoder plan: `experiments/frozen-encoder-v6-20260928-plan.json`; and
- completed rejected-run evidence: `experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`.

Runtime state defaults to `%LOCALAPPDATA%\KEV\alive` on Windows and `~/.kev/alive` elsewhere. A default evolution run writes immutable artifacts beneath:

```text
<state-dir>/evolution/candidates/<run-id>/
├── trained/
│   ├── semantic-breadth.pt
│   └── training-receipt.json
├── calibrated/
│   ├── semantic-breadth.calibrated.pt
│   └── calibration-receipt.json
├── eval-card.json
└── evolution-result.json
```

Pipeline failures write `raw-failure.json`. Rejected candidates remain in their run directory. A qualified model is referenced by `<state-dir>/model-registry.json` and the active-model fields in `state.json`; the former incumbent remains preserved.

See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md) for hashes, manifests, evidence boundaries, and promotion policy, and [the reproducibility guide](docs/REPRODUCIBILITY.md) for an exact clean-clone replay.

## Implemented versus planned

Implemented now:

- parser-first multi-frame extraction for bounded patterns;
- typed state with exact claim spans and provenance;
- local persistence, fact revisions, migrations, locking, and ledger verification;
- draft/review/export lesson provenance;
- immutable challenger training, separate calibration, frozen evaluation, and strict reject/qualify gates;
- a reproducible public genesis artifact and public evaluation pack; and
- library-level bounded action receipts, verified observations, and prediction scoring.

Not yet implemented as a complete product:

- robust open-domain or unfamiliar-language understanding;
- a bundled, calibrated, broadly trained public semantic incumbent;
- default real-world tool adapters or autonomous action execution;
- general state-conditioned reasoning across arbitrary tasks;
- a full state, evidence, lesson-review, and candidate-comparison desktop workspace;
- an external or signature-backed ledger anchor and path-portable generated evolution-run evidence bundles;
- independent external evaluation; or
- AGI or superintelligence.

See the [research roadmap](docs/ROADMAP.md) for the staged plan.

## Invariants for contributors and AI agents

- Never overwrite an incumbent checkpoint.
- Training completion is not promotion.
- Ordinary chat is not automatically truth or training data.
- Reviewed lessons must be explicit and provenance-bearing.
- Once an audit influences a change, preserve it as development evidence; do not call it fresh again.
- Separate memory improvement from weight improvement.
- Never claim a tool ran without a durable execution receipt.
- Preserve raw failures and rejected candidates.

Start with [AGENT_START_HERE.md](AGENT_START_HERE.md), [AI_AGENT_GUIDE.md](AI_AGENT_GUIDE.md), [Architecture](docs/ARCHITECTURE.md), [Mathematics](docs/MATHEMATICS.md), and the [Roadmap](docs/ROADMAP.md).

## Repository policy

Small public reference and evaluation artifacts are committed so the current research path is replayable. The bounded v6 experiment also retains its small rejected candidate heads and evidence bundle in Git. Large encoder weights, unrelated generated candidates, historical/private checkpoints, private runtime state, conversations, caches, and archives remain outside ordinary Git history. Never silently substitute different weights or publish private lesson/session data.

## Project status

**Research alpha.** KEV is a persistent, teachable, evidence-grounded cognitive-runtime experiment. The ambition is large; the measurements stay literal.

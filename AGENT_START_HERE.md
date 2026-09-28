# Start Here — KEV

KEV is experimental research software for explicit cognitive state and gated
model evolution. It is not AGI or superintelligence. Language is an interface
to explicit cognitive state, not the state itself; a frozen encoder is likewise
a feature interface, not cognition.

Read, in order:

1. `README.md`
2. `AI_AGENT_GUIDE.md`
3. `docs/ARCHITECTURE.md`
4. `docs/MATHEMATICS.md`
5. `docs/ROADMAP.md`
6. `MODEL_WEIGHTS.md`

## Non-negotiable invariants

Latest research-only boundary: `experiments/clause-local-v1-boundary/`.
Read `docs/CLAUSE_LOCAL_V1.md` before further model work. This exposed suite and
its reserved vocabulary must be excluded from future training; the production
v7 manifest does not yet automatically include this separate research pack.
Do not promote its candidate or mistake the diagnostic numeric gate for authority.

- Never overwrite an incumbent checkpoint. Training and calibration write new,
  content-addressed artifacts.
- Training completion is not promotion. Strict-gate ties reject.
- Ordinary chat is neither truth nor training data. Only explicitly reviewed,
  permission-bearing lesson exports may reach `train()`.
- Keep memory improvement separate from weight improvement.
- Preserve provenance, raw failures, superseded facts, frozen suites, eval
  cards, and rejected candidates.
- Never claim a tool ran without its execution receipt. Tool observations are
  `VERIFIED` only after the receipt is accepted by the ledger.
- Do not edit an evaluation pack after it influences development. Create a new
  version and preserve the old bytes and result.
- Generated text has no state authority. Unsupported narration is
  `UNGROUNDED` and cannot become a fact.
- A completed run or declining training loss is not promotion evidence. Never
  describe a rejected challenger as qualified.

## Current sources of truth

- Active checkpoint pointer and hash: `models/registry.json`
- Checkpoint provenance: `models/public/incumbent-manifest.json`
- Current incumbent evidence: `models/public/incumbent-evidence-v11.json`
- Frozen artifact manifest: `evals/frozen/manifest-v7.json`
- Current promotion suite: `evals/frozen/public-audit-v7-260.json`
- Current calibration-fit slice: `evals/frozen/calibration-fit-v2.jsonl`
- Current held-out vocabulary: `evals/frozen/held-out-vocabulary-v2.txt`
- Current raw baseline: `evals/evidence/v7-genesis-baseline.json`
- Development-only parser failures: `evals/frozen/frame-parser-known-failures-v1.json`
- Historical aggregate replay: `evals/frozen/historical-190-260-replay.json`
- Preregistered frozen-encoder plan: `experiments/frozen-encoder-v6-20260928-plan.json`
- Completed frozen-encoder evidence: `experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`
- Composition protocol: `experiments/composition-v7-20260928-protocol.md`
- Composition experiment: `experiments/composition-v7-20260928-plan.json`
- Composition result and replay: `docs/COMPOSITION_V7.md`

V1 through v6 evaluation artifacts are preserved development evidence, not
current fresh suites. Do not regenerate or patch them. The parser-failure file
is development-influenced and explicitly ineligible for promotion scoring.

## Current v7 result

The [v7 report](docs/COMPOSITION_V7.md) records the completed alpha.2 experiment.
All three candidates were rejected for a composition tie at 75/80. Primary seed
52031 improved fresh from 26/80 to 39/80 but also regressed a held-out vocabulary
audit. The incumbent weights remain unchanged. The controlled runtime repair
improved eight developer-proposal probes from 1/8 to 8/8; that is not evidence
of learned composition improvement. V7 results have now influenced interpretation;
do not tune against them and present a rerun as fresh evidence.

## Preserved v6 preregistered result

The plan SHA-256 is
`c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`.
The aggregate evidence file SHA-256 is
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`,
with canonical integrity SHA-256
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.
Its outcome is `PRIMARY_REJECTED_NO_RECOMMENDATION`.

The frozen genesis baseline scored 48/80 fresh, 40/40 retention, 3/40 OOV,
50/80 composition, and 0/20 calibration, with 119 raw failures. Primary seed
52031 scored 60/80, 40/40, 16/40, 50/80, and 4/20. Robustness seeds 52047 and
52069 scored 58/80, 40/40, 9/40, 50/80, 4/20 and 62/80, 40/40, 18/40,
50/80, 3/20, respectively. All three evidence bundles are `COMPLETE`; all
three decisions are `REJECT` for `COMPOSITION_TIE` and
`AUDIT_HELD_OUT_VOCABULARY_AMELIORATE_REGRESSED`.

Training loss declined but was not promotion input. The 197-row corpus was
synthetic and same-party reviewed, not independent human review. Candidates
and raw failures are retained. No challenger qualified, the public registry
did not change, and the incumbent SHA-256 remains
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.
Do not convert this bounded rejected experiment into a capability or AGI claim.

## Before and after a change

Run:

```text
python -m kev.cli doctor
python -m pytest -q -p no:cacheprovider tests_public
```

For a model change, use `kev evolve --lessons FILE`; do not swap checkpoint
files manually. A credible result includes the parent, lesson, held-out-data,
suite, candidate, and eval-card hashes plus the ledger decision. If a skeptic
cannot replay the claim from retained artifacts, the work is incomplete.

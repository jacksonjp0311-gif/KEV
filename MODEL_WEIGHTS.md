# Model and Evaluation Artifacts

KEV checkpoints are immutable, content-addressed artifacts. A model is an
incumbent only when a registry points to its exact path and SHA-256. Training
creates a challenger; it never overwrites or activates the parent.

## Active public reference incumbent

Latest evaluation-only pack: [language challenge v1](docs/LANGUAGE_CHALLENGE_V1.md).
No checkpoint was created, modified, or promoted. Its 64-case suite is
`experiments/language-challenge-v1-boundary/suite.json`, SHA-256
`0fc1a03e07cfdaeb93d265d19ebe97592a36fdef67560ff7ce3346ea4f2ff52e`.
Its protocol SHA-256 is
`8c777a15aaefefab25a416746ce43c96d3c139e4419761437f9b1f3a19a9878e`;
evidence inventory SHA-256 is
`bc4060dd8ef283f77700754b75bf3cd3429fd6d6e6855b5758c3524d73c0b138`.
The protocol pins the unchanged genesis and retained clause-local-v1 candidate,
runtime sources, registry, encoder manifest and retained lesson corpora.

Latest non-activating experiment: [clause-local v1](docs/CLAUSE_LOCAL_V1.md).
Its checkpoint is **not an incumbent**; the primary hypothesis was NOT_SUPPORTED,
and the post-hoc no-model control outperformed the learned local head.

| Research artifact | SHA-256 |
|---|---|
| `experiments/clause-local-v1-boundary/protocol.json` | `f776c41a65bba23beeff41c027049f54965373e5f3177ca9073c48d49cf0d41f` |
| `experiments/clause-local-v1-boundary/suite.json` | `64f71b7714f488ae62caf6ba1f5e25e719b3721761c396ab70effa3c2493c4a4` |
| `experiments/clause-local-v1-boundary/reviewed-lessons.jsonl` | `d4b4eb48ef35866e077b69a8aa00e3be6bdd7112e6fd8917e48ea700274329cb` |
| `experiments/clause-local-v1-evidence/candidate/semantic-breadth.pt` | `74d3463755d692c4e8f6fd6be62eba74f76ff9c880194a670b422aaabd99917c` |
| Parent checkpoint | `8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6` |
| `experiments/clause-local-v1-evidence/research-card.json` | `6acc060d669398dfb77ffcb225a9904605b6444c878f7ddf524ec26dcdb698f5` |
| `experiments/clause-local-v1-evidence/inventory.json` | `61f201bfacaa998e32a8f8d1092b1791bcdd6b8efb1505fd2b4c1bed842b8b91` |

The inventory pins raw reports, checkpoint, embeddings, receipts and ledger.
Scores are UNCALIBRATED; no weights or production registry were replaced.

The repository includes a small public genesis checkpoint so a clean clone is
replayable without silently substituting historical private weights.

| Field | Value |
|---|---|
| Name | `v0.52 public genesis reference` |
| Path | `models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt` |
| SHA-256 | `8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6` |
| Size | 1,144,120 bytes |
| Parameters | 285,092 |
| Parent | none; deterministic genesis initialization |
| Training data / steps | none / 0 |
| Score status | `UNCALIBRATED` |
| Claim boundary | bootstrap artifact, not evidence of learning or general intelligence |

Registry: `models/registry.json`

Artifact manifest: `models/public/incumbent-manifest.json`

Evidence manifest: `models/public/incumbent-evidence-v11.json`
(`146bd0ed36728df46566d33002b9b253b4fb7d06d6f41a031c2dfba5dd2de770`)

Its current frozen v7 baseline is preserved at
`evals/evidence/v7-genesis-baseline.json`
(`b124cae0447847225f4c7edc54e38f97517e24ae13ce809708ad0f1dd84ca4a0`).
Its canonical report hash is
`b0b9a10f727a9c5c7849a6b2c570b0d7064c10f8ce03f4212c67e490ead3decf`.
Measured exact-match scores are fresh 26/80, retention 40/40, OOV 2/40,
composition 75/80, and calibration 1/20, with all 116 raw failures retained.
These measure unchanged genesis weights with the v7 runtime. Runtime changes,
different evaluation suites, and weight improvement are separate claims.

A hosted replay differed in one rounded confidence by `0.000001`, with
identical predictions, failures, accuracy counts, and reject decision. Its complete
report has SHA-256
`8a50f0fa1b92697681e0688c509b3a30c3dd2bdc03bac2ea91101504879cccc6`.
The [numerical replay manifest](experiments/composition-v7-portability/replay-variants-v1.json)
preserves that derivative and its failure provenance without replacing the
reference baseline or advancing the registry. Only these exact declared
report bytes are accepted by the replay test; unseen variation fails.

## Frozen promotion evidence

Current promotion suite:

- Path: `evals/frozen/public-audit-v7-260.json`
- File SHA-256: `e22bdbea5f07c8b44cf5a1684a2b8433be6b252d74ddf86ef0252b736dbe9c76`
- Canonical JSON SHA-256: `d3590d6f9eda20a3510df94f7473aeed45c5d616aadfe4ee7fb5f452f0036f28`
- Manifest: `evals/frozen/manifest-v7.json`
- Manifest SHA-256: `f8784d55629f86377ca80aadf52c838e40b20c497c28653b8605d5e505f6a036`
- Splits: fresh 80, retention 40, OOV 40, composition 80, calibration 20
- Robustness audits: single and multiple frames, buried and polite constraints,
  paraphrase equivalence, conflicting constraints, negation, number change, and
  clause-order swap
- Independence boundary: v7 retains 40 frozen retention items and adds 220
  newly authored surfaces, excluding exact and digit-normalized historical
  overlaps. All composition targets require exact semantic frames. The suite
  author did not read the new training corpus; review remains same-party
  synthetic review, not independent external evaluation.

Separate calibration-fit slice:

- Path: `evals/frozen/calibration-fit-v2.jsonl`
- SHA-256: `d0e7d4fcffef2c3ccd376acb2c77a1fb90289224caa046cddc88c9656ace101c`
- Scope: 52 label-clean rows, balanced across all 13 frame kinds

Held-out vocabulary:

- Path: `evals/frozen/held-out-vocabulary-v2.txt`
- SHA-256: `087da62e8129dd259c1b2662f0439eb7d6387714528af14444a4575c66e76798`
- Terms: `ameliorate`, `sacrosanct`, `attested`, `presage`, `dormant`,
  `displaces`, `norm`, `antipode`, `citation`, and `concordant`

These artifacts must never enter `train()`. After an evaluation influences a
change, its file is preserved. New items require a new version and new hashes.

Development-influenced parser probes are preserved separately at
`evals/frozen/frame-parser-known-failures-v1.json`
(`1aa0d4b519bfdd96e1874f25a0f814149a3b48f51dc8e38274f3847a46a6a901`).
The artifact records six cases, including the two-item order control, with
seven literal pre-fix observations. It is explicitly marked
`promotion_eligible: false`, is pinned by the v4 manifest, and is never scored
as fresh promotion evidence.

### Preserved development evidence

The first frozen public pack saturated the deterministic fresh parser slice.
It was not edited after that result. It remains at
`evals/frozen/public-audit-v1-260.json`
(`1c5889428acb0920786f813331f5188b63393d63ec64986c123350218c8b58f0`),
with raw self-evaluation at `evals/evidence/v1-baseline-self-eval.json`
(`d8338e14763c091ea67ae556e8e2d306db6487abaf9faa92653cdb5300de644f`).
V2 added measurable neural-head headroom while preserving v1 unchanged. Its
suite remains at `evals/frozen/public-audit-v2-260.json`
(`ef9ff61967f8716e541ccdef3adea887da99dc988799098a9faf47e2ea3dd3c4`)
and its raw baseline remains at `evals/evidence/v2-genesis-baseline.json`
(`f8282ddd540c0903f1219996f085ad39574fa4d0ef912ab20507328c6d7ce3ac`).
V3 added the explicit clause-order-swap audit without changing either earlier
pack. It is preserved as development evidence after an audit found that 120
kind-projection prompts exposed their expected kind names or synthetic labels;
it is not the current promotion suite. The v3 baseline preserves 113 raw
failures. The first v3 card with an
informational absolute build path also remains unchanged at
`evals/evidence/v3-genesis-baseline.json`
(`5f27127f44044cb36690cda5df768d4366af79f7436ce8ed48d6ab192c92903a`).
The intermediate CRLF-serialized card changed only path serialization to
repository-relative form, and its evidence manifest is also preserved;
portable-v2 fixed newline serialization. Portable-v3 additionally published
audit-family metrics and audit tags on every raw failure. Portable-v4 quantizes
evidence scores to six decimals for stable replay. Its canonical report hash is
`93688ff01a7e28f1b47a7ab33e8453561ad94f7db41e02acbc607bf401d60f10`.

The first v4 self-evaluation predates per-audit-family promotion gates and is
preserved unchanged at `evals/evidence/v4-genesis-baseline.json`
(`3e2ac77d9dd210cad542b079563a29575d030d35971c8dca3e6ecbdbd3f2f838`)
under `models/public/incumbent-evidence-v7.json`
(`a6c19aafc595f291821109e1a9fc26fd1d5b22d4bc2748081db0a4dc7e752bf3`).
The v8 evidence manifest links this intermediate evidence to the current
audit-gated card; neither the v4 suite nor its manifest was modified for that
later policy change.

V4's audit-gated card and v8 evidence remain immutable at
`evals/evidence/v4-genesis-baseline-audit-gated.json` and
`models/public/incumbent-evidence-v8.json`. The v5 manifest directly pins both
that card and the v4 manifest. V5's suite, baseline, freshness-audit finding,
and v9 incumbent evidence also remain immutable at
`evals/frozen/public-audit-v5-260.json`,
`evals/evidence/v5-genesis-baseline.json`,
`evals/evidence/v5-fresh-template-audit.json`, and
`models/public/incumbent-evidence-v9.json`. The audit found six number-only
fresh-template repetitions before any v6 challenger training. V6 replaced
only those surfaces and points back to v5/v9; it did not rewrite them.

## Historical v0.51 record

The former documentation named `models/v051a2/semantic-breadth.pt` as a
284,525-parameter incumbent, but neither its bytes, exact SHA-256, nor the
original 260 item texts were present in the public Git history. KEV does not
fabricate them.

The published aggregate parent 190/260 versus challenger 190/260 result is
preserved at `evals/frozen/historical-190-260-replay.json`
(`4c7109721436b6d2f854c2a89449667d98a4b0f74161e1f26a869b75311890cb`).
The replay rejects the tie and explicitly records that item-level failures and
checkpoint hashes are unavailable. It is historical aggregate evidence, not a
current full-policy evaluation.

## Promotion policy

A challenger qualifies only when all applicable gates pass:

1. strict fresh improvement;
2. retention regression no greater than the declared epsilon;
3. no OOV regression;
4. strict composition improvement when composition items are present; and
5. no regression in every audit family present in either report, with a
   missing family or different family total rejecting.

A tie on a strict gate rejects. Training loss is logged but is never a
promotion feature. Rejected checkpoint bytes, raw failures, training receipts,
and eval cards stay on disk.

## Current composition experiment

The `composition-v7-20260928` protocol and authoritative result are recorded in
[Composition v7](docs/COMPOSITION_V7.md). Its final training input is
[`training/reviewed/semantic-frame-paraphrases-v8-reviewed.jsonl`](training/reviewed/semantic-frame-paraphrases-v8-reviewed.jsonl),
497 rows, SHA-256
`c78eeb1b91d22eea88177f003971a884e5f519212acf140a6343f7b912f15a09`.
Its manifest is
[`training/reviewed/semantic-frame-paraphrases-v8-manifest.json`](training/reviewed/semantic-frame-paraphrases-v8-manifest.json),
SHA-256 `bc6698f7d64ea61ae9ddf0522ec94c9cb813809366fdaa0ab32b57aa36615242`.
The preserved 500-row v7 predecessor was rejected before training for three
digit-normalized template overlaps. V8 removes only those three lesson IDs;
the unchanged evaluation suite and surviving lesson texts remain frozen.
The corpus and its review are synthetic and same-party, not independent human
review. Training completion does not qualify a checkpoint.

The immutable [plan](experiments/composition-v7-20260928-plan.json) has SHA-256
`7e51fe4ab1928e6805dda6443c2c8a8fb0d4ca0e796f2b6fca9ef0c1ad688d0c`.
The [aggregate evidence](experiments/composition-v7-20260928-evidence/aggregate-evidence.json)
has file SHA-256
`50cff3716b62d0c06e9b0aec2ffa3d6e49e0dee3a7dd27b92620d0ea1b625f0f`
and canonical integrity SHA-256
`5fff30454c7aca1dd1a0e32e6bdaa7c92658f802fb93ce620fc48e416bccd798`.
Outcome: `PRIMARY_REJECTED_NO_RECOMMENDATION`. Every seed has `COMPLETE`
integrity and a `MODEL_REJECTED` terminal event. Scores below are exact-match
counts in fresh / retention / OOV / composition / calibration order.

| Seed | Scores | Calibrated checkpoint SHA-256 | Eval-card file SHA-256 | Terminal event SHA-256 |
|---:|---|---|---|---|
| [52031](experiments/composition-v7-20260928-evidence/seed-results/seed-52031/) | 39/80; 40/40; 5/40; 75/80; 10/20 | `8a7587f08f5e27c1dca589af7402f5798bce3b663253e6d0c2d54eb861307cf8` | `0806f35c8e3801873ec0cfe1e80fda0e3de061e8872176166ef9ddf068596afb` | `70ff7146384a0cde725cc0e615d9b18de4c11a15375c4ecbecc27d8df0615cab` |
| [52047](experiments/composition-v7-20260928-evidence/seed-results/seed-52047/) | 41/80; 40/40; 10/40; 75/80; 8/20 | `30a8257eb5b061544d04e15e36974af5bde776478056f78c92f3b63ccf8e544f` | `a80c5ca0fc6926ef97bb223b3c64914a0553057f6432026decb0f4ad7f074bb5` | `5a12195598b1d05618316583b67f022e300e3f3573c92d06d5926722812e2d22` |
| [52069](experiments/composition-v7-20260928-evidence/seed-results/seed-52069/) | 41/80; 40/40; 8/40; 75/80; 11/20 | `efdfe8944fa0a926ae33f9f8cd31a3ee352c162b65869e74ea8f27c43f9e56a8` | `6cf6c9c65d4d9853c57a8f623fb1000b1f735c17c44375bd5b1ae0db49e91697` | `2036d7ce5b4c0903424cad6d55a92aba53bd6ee5884e49614eab15ce3ef1f94b` |

The unchanged incumbent scored 26/80; 40/40; 2/40; 75/80; 1/20 under the
same runtime. All challengers reject for `COMPOSITION_TIE`; 52031 and 52047
also reject for `AUDIT_HELD_OUT_VOCABULARY_AMELIORATE_REGRESSED` (2/4 to 1/4).
Seed 52069 has no audit regression but still fails the strict composition gate.
Raw failure counts are 91, 86, and 85. Final input, cross-seed, ledger, and
independent replay checks passed. The evidence pointer was updated to v11
before training; neither the registry nor the incumbent changed during the run.
Developer-proposal probes improved from 1/8 to 8/8, but those development cases
are promotion-ineligible and do not establish learned composition improvement.

## Previous completed v6 frozen-encoder experiment

The bounded v6 experiment is complete. Its immutable plan is
[`experiments/frozen-encoder-v6-20260928-plan.json`](experiments/frozen-encoder-v6-20260928-plan.json)
(`c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`).
The preserved aggregate is
[`experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`](experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json):
file SHA-256
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`
and canonical evidence SHA-256
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.

The reviewed lesson input is the 197-row repository-authored synthetic corpus
[`training/reviewed/semantic-frame-paraphrases-v6-reviewed.jsonl`](training/reviewed/semantic-frame-paraphrases-v6-reviewed.jsonl)
(`c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed`).
It received same-party agent review, not independent human or external review;
ordinary chat was not used as training data. Its limitations and provenance
are pinned by
[`training/reviewed/semantic-frame-paraphrases-v6-manifest.json`](training/reviewed/semantic-frame-paraphrases-v6-manifest.json).

Scores below are exact-match counts in the fixed order fresh / retention / OOV
/ composition / calibration. The raw checkpoint is the uncalibrated trained
projection and typed heads; the calibrated checkpoint is the distinct artifact
that entered evaluation.

| Seed and role | Raw checkpoint SHA-256 | Calibrated checkpoint SHA-256 | Temperature | Exact-match scores | Eval-card file SHA-256 | Terminal ledger-event SHA-256 | Decision |
|---|---|---|---:|---|---|---|---|
| [52031 (primary)](experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52031/) | `f9403cfe957e88adf94011bd90244651c5efa43aec44803d3026ef6c1238d8bb` | `d7ca702ff24333b952dbc4928e8aa1fd0bc2a4ab8b7a4ecf1c8f62254e570e84` | 0.6370194554 | 60/80; 40/40; 16/40; 50/80; 4/20 | `363c6a076203a00a5868e439a0bcfdbd67855a4fc27d43b70537b90a425d2a1a` | `c980feb4ecf56f55418b3bdee66feefeb3c1de47653b3b8e0c7ef2d15701819f` | `REJECT` |
| [52047 (robustness)](experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52047/) | `83d00e2cc15bb14a721f0098613feefdc2b300659da13481fe28d51465e203b4` | `081527c9cb50fb78a313e2fff2ef56e4231358e099f4a8fd72798d7bd030768f` | 0.6506620646 | 58/80; 40/40; 9/40; 50/80; 4/20 | `a114be7dcd0565bc9901e2249e8513faae60420826b64d6274fbeab5a065db0e` | `4eac00b851b36a5b793bb11d786dee16a824a7a76203b1b5218981b2e62ae6aa` | `REJECT` |
| [52069 (robustness)](experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52069/) | `830a85aa07c9c98eb25a7a6076e81b628c1ec4a7a556fbf76870ffc79a6f2a4f` | `a1caae479b34bee670be566747c3491539ad3172d128f37629338f2b26f046e6` | 0.6312441230 | 62/80; 40/40; 18/40; 50/80; 3/20 | `ba9c8d94080ddc657917acedfdb60c023356d614cd39df2e16d67eb94c2471ba` | `3ce6d4516e58d3c1c3015fcaf84eee968bf36a1e1d2aad88dc1c4fdbaa518e21` | `REJECT` |

Every seed was rejected for both `COMPOSITION_TIE` and
`AUDIT_HELD_OUT_VOCABULARY_AMELIORATE_REGRESSED`. Aggregate input validation,
cross-seed non-seed input equality, per-seed evidence integrity, ledger
verification, and final revalidation all passed. Training loss was recorded
but never entered the gate. All three raw and calibrated candidates, receipts,
eval cards, raw failures, and terminal events remain under
[`experiments/frozen-encoder-v6-20260928-evidence/`](experiments/frozen-encoder-v6-20260928-evidence/).
The aggregate outcome is `PRIMARY_REJECTED_NO_RECOMMENDATION`; the public
registry remains on the genesis checkpoint.

This is a negative bounded experiment. It does not establish broad language
understanding, general intelligence, AGI, or superintelligence.

## Selected external frozen encoder (not an incumbent)

KEV used `sentence-transformers/all-MiniLM-L6-v2` as the local frozen feature
encoder in the completed preregistered experiment above. The encoder and all
three rejected challengers are not incumbents; their use does not establish a
model improvement.

| Field | Value |
|---|---|
| Upstream revision | `1110a243fdf4706b3f48f1d95db1a4f5529b4d41` |
| Declared license | `Apache-2.0` |
| Weight SHA-256 | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| Acquisition record | `models/encoder-sources/all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json` |
| Acquisition-record SHA-256 | `a46078dd1c13d9114bff07a891a1a903c1281681cf6dd3002e8d9e5ea8da03e9` |
| Expected local-manifest SHA-256 | `cfd8f8bc3413f31ab927ad7d78eb380c3359fdb2893061075d2369dfec81ea57` |
| Runtime policy | `LOCAL_ONLY`; safetensors, fixed BERT loader, no dynamic remote code |

Large encoder bytes remain ignored under `models/local/encoders/`. A
challenger stores only its projection and typed heads while binding the local
manifest hash. Exact acquisition, offline verification, loader, training, and
claim-boundary details are in
[`docs/FROZEN_ENCODER.md`](docs/FROZEN_ENCODER.md).

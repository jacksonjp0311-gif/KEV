# Model and Evaluation Artifacts

KEV checkpoints are immutable, content-addressed artifacts. A model is an
incumbent only when a registry points to its exact path and SHA-256. Training
creates a challenger; it never overwrites or activates the parent.

## Active public reference incumbent

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

Evidence manifest: `models/public/incumbent-evidence-v10.json`
(`69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21`)

Its current frozen v6 baseline is preserved at
`evals/evidence/v6-genesis-baseline.json`
(`881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4`).
Its canonical report hash is
`e6c768c496048afbdcf56554700ac2cc19e9c09d013139e2c7c53cdb3a9f658e`.
Measured exact-match scores are fresh 0.60, retention 1.00, OOV 0.075,
composition 0.625, and calibration split 0.00, with all 119 raw failures
retained. These are literal baseline measurements, not capability claims.

## Frozen promotion evidence

Current promotion suite:

- Path: `evals/frozen/public-audit-v6-260.json`
- File SHA-256: `a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029`
- Canonical JSON SHA-256: `3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`
- Manifest: `evals/frozen/manifest-v6.json`
- Manifest SHA-256: `07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`
- Splits: fresh 80, retention 40, OOV 40, composition 80, calibration 20
- Robustness audits: single and multiple frames, buried and polite constraints,
  paraphrase equivalence, conflicting constraints, negation, number change, and
  clause-order swap
- Independence boundary: v5 was preserved after an independent audit found
  six number-only fresh-template repetitions. Only those six surfaces were
  replaced before challenger training, without reading a lesson corpus. V6
  rejects exact and digit-normalized template overlap with v1-v5 and the
  calibration fit, plus complete expected-label names.

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

## Completed preregistered frozen-encoder experiment

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

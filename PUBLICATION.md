# Public Repository Snapshot

This repository is the replayable public KEV v0.52 research snapshot. It
includes source, tests, documentation, frozen evaluation packs, raw public
baseline failures, manifests, one deterministic untrained genesis checkpoint,
and the complete evidence from one bounded preregistered frozen-encoder
experiment. KEV is experimental research software, not AGI or
superintelligence.

## Current public evidence boundary

The active public incumbent remains the untrained genesis checkpoint
[`models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt`](models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt),
SHA-256
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.
The registry was not changed by the experiment.

The current frozen evaluation chain is:

- v6 manifest
  [`evals/frozen/manifest-v6.json`](evals/frozen/manifest-v6.json), SHA-256
  `07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`;
- v6 promotion suite
  [`evals/frozen/public-audit-v6-260.json`](evals/frozen/public-audit-v6-260.json),
  file SHA-256
  `a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029`
  and canonical SHA-256
  `3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`;
- genesis v6 baseline
  [`evals/evidence/v6-genesis-baseline.json`](evals/evidence/v6-genesis-baseline.json),
  file SHA-256
  `881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4`;
  and
- incumbent evidence v10
  [`models/public/incumbent-evidence-v10.json`](models/public/incumbent-evidence-v10.json),
  SHA-256
  `69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21`.

V1-v5 suites, their baseline evidence, and incumbent evidence v1-v9 remain in
the repository as immutable historical lineage. V6/v10 supersede those files
as the current references; they do not overwrite them. The historical 190/260
tie remains aggregate-only because its item texts and checkpoint hashes are
unavailable and are not fabricated.

## Completed preregistered experiment

The immutable experiment plan is
[`experiments/frozen-encoder-v6-20260928-plan.json`](experiments/frozen-encoder-v6-20260928-plan.json),
SHA-256
`c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`.
The aggregate evidence is
[`experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`](experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json),
file SHA-256
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`
and canonical SHA-256
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.

The only reviewed lesson corpus supplied to weight training was the 197-row
repository-authored synthetic corpus
[`training/reviewed/semantic-frame-paraphrases-v6-reviewed.jsonl`](training/reviewed/semantic-frame-paraphrases-v6-reviewed.jsonl),
SHA-256
`c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed`.
It had same-party agent review and no independent human or external review.
Ordinary chat was not used as training data.

Scores are exact-match counts in fresh / retention / OOV / composition /
calibration order.

| Seed | Raw checkpoint SHA-256 | Calibrated checkpoint SHA-256 | Temperature | Scores | Eval-card file SHA-256 | Terminal event SHA-256 | Result |
|---:|---|---|---:|---|---|---|---|
| 52031 | `f9403cfe957e88adf94011bd90244651c5efa43aec44803d3026ef6c1238d8bb` | `d7ca702ff24333b952dbc4928e8aa1fd0bc2a4ab8b7a4ecf1c8f62254e570e84` | 0.6370194554 | 60/80; 40/40; 16/40; 50/80; 4/20 | `363c6a076203a00a5868e439a0bcfdbd67855a4fc27d43b70537b90a425d2a1a` | `c980feb4ecf56f55418b3bdee66feefeb3c1de47653b3b8e0c7ef2d15701819f` | `REJECT` |
| 52047 | `83d00e2cc15bb14a721f0098613feefdc2b300659da13481fe28d51465e203b4` | `081527c9cb50fb78a313e2fff2ef56e4231358e099f4a8fd72798d7bd030768f` | 0.6506620646 | 58/80; 40/40; 9/40; 50/80; 4/20 | `a114be7dcd0565bc9901e2249e8513faae60420826b64d6274fbeab5a065db0e` | `4eac00b851b36a5b793bb11d786dee16a824a7a76203b1b5218981b2e62ae6aa` | `REJECT` |
| 52069 | `830a85aa07c9c98eb25a7a6076e81b628c1ec4a7a556fbf76870ffc79a6f2a4f` | `a1caae479b34bee670be566747c3491539ad3172d128f37629338f2b26f046e6` | 0.6312441230 | 62/80; 40/40; 18/40; 50/80; 3/20 | `ba9c8d94080ddc657917acedfdb60c023356d614cd39df2e16d67eb94c2471ba` | `3ce6d4516e58d3c1c3015fcaf84eee968bf36a1e1d2aad88dc1c4fdbaa518e21` | `REJECT` |

Every seed failed the same two gates: `COMPOSITION_TIE` and
`AUDIT_HELD_OUT_VOCABULARY_AMELIORATE_REGRESSED`. Training loss was logged but
was never a promotion feature. Plan/input validation, cross-seed non-seed input
equality, each evidence inventory, every ledger, and final seed revalidation
all passed. The raw and calibrated rejected candidates, receipts, eval cards,
raw failures, and terminal events are preserved under
[`experiments/frozen-encoder-v6-20260928-evidence/`](experiments/frozen-encoder-v6-20260928-evidence/).

## Publication and claim policy

The experiment is a negative result: `PRIMARY_REJECTED_NO_RECOMMENDATION`.
Higher fresh or aggregate OOV counts do not override a strict composition tie
or an audit-family regression. No candidate qualified, no checkpoint was
promoted, and the public registry still names the genesis incumbent.

These measurements do not establish broad language understanding, semantic
generalization, a new cognitive state, AGI, or superintelligence. The frozen
sentence encoder is a feature front-end; authority remains with typed state,
receipts, and the gated decision. The synthetic, same-party-reviewed corpus is
a material limitation and must accompany any report of the result.

The repository intentionally excludes unreported private/local runtime state,
session transcripts, private reviewed lessons, caches, archives, and
unverifiable historical checkpoint binaries. Large future checkpoints should
use a content-addressed GitHub Release or Git LFS reference. Never silently
substitute weights, regenerate a frozen suite in place, remove rejected
candidates from a reported experiment, publish private state, or report
training completion as promotion.

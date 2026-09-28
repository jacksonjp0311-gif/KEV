# Frozen Sentence Encoder Acquisition and Runtime

KEV's selected broader-language feature source is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned to upstream commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its model card declares
`Apache-2.0`. This is a frozen language-feature component, not cognitive state,
an authority source, or a model-promotion decision.

The selected encoder is not bundled in Git and is not the public incumbent.
Selection and integration do not establish an improvement. The previous v6
three-seed experiment rejected every challenger. The current composition
experiment is documented in [Composition v7](COMPOSITION_V7.md); only a
challenger that passes every frozen evaluation gate may become an incumbent.

## Pinned evidence

| Artifact | SHA-256 |
|---|---|
| Committed acquisition source record | `a46078dd1c13d9114bff07a891a1a903c1281681cf6dd3002e8d9e5ea8da03e9` |
| Upstream `model.safetensors` | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| Upstream model card | `dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0` |
| Apache-2.0 license text | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| Reproducible local encoder manifest | `cfd8f8bc3413f31ab927ad7d78eb380c3359fdb2893061075d2369dfec81ea57` |
| Current composition v7 plan | `7e51fe4ab1928e6805dda6443c2c8a8fb0d4ca0e796f2b6fca9ef0c1ad688d0c` |
| Current reviewed v8 corpus | `c78eeb1b91d22eea88177f003971a884e5f519212acf140a6343f7b912f15a09` |
| Previous v6 experiment plan | `c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca` |
| Previous v6 reviewed corpus | `c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed` |
| Previous v6 aggregate evidence | `c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77` |

The committed source record is
[`models/encoder-sources/all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json`](../models/encoder-sources/all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json).
It pins all 12 acquired files, their sizes and hashes, the loader recipe, the
license evidence, the expected generated manifest hash, and the exact review
inputs. The acquisition execution writes a machine-readable local
`acquisition-receipt.json`; that receipt is host-specific and remains beside
the ignored model bytes.

The license review approves local frozen feature extraction under the model
card's declared Apache-2.0 terms. It does not relicense upstream training data,
authorize automatic training-data collection, or permit generated text to
become state.

## Reproducible acquisition

Use Python 3.12 and install the published CPU research versions:

```text
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-research.txt -e ".[dev,semantic-encoder]"
```

The [constraints](../constraints-research.txt) pin package versions rather than
wheel hashes. Exact numerical replay also requires the runtime backend and
source versions in the experiment plan; hardware-independent floating-point
identity is not claimed.

From a clean repository root, run the exact command pinned in the source record:

```text
python scripts/acquire_frozen_encoder.py --source-manifest models/encoder-sources/all-MiniLM-L6-v2-1110a243fdf4706b3f48f1d95db1a4f5529b4d41.json --reviewed-by "KEV repository owner authorization recorded in task" --reviewed-at 2026-09-28T11:21:02Z --accept-license Apache-2.0
```

The only network-bearing step is this explicit acquisition command. It fetches
the named files from the exact upstream revision and the Apache license URL,
then rejects any byte or size mismatch. It writes without overwriting an
existing local manifest or receipt.

The resulting ignored local bundle is:

```text
models/local/encoders/all-MiniLM-L6-v2/
└── 1110a243fdf4706b3f48f1d95db1a4f5529b4d41/
    ├── encoder-manifest.json
    ├── acquisition-receipt.json
    ├── model.safetensors
    ├── tokenizer.json
    ├── config.json
    ├── modules.json
    ├── 1_Pooling/config.json
    ├── README.md
    ├── LICENSE-2.0.txt
    └── other hash-pinned tokenizer/config files
```

After copying that complete directory to an offline machine, verify it without
network access:

```text
python scripts/acquire_frozen_encoder.py --verify-only
```

Verification checks the committed source record, expected local manifest hash,
every encoder/config/tokenizer/license file, upstream revision, and acquisition
receipt bindings.

## Current composition experiment

The immutable plan is
[`experiments/composition-v7-20260928-plan.json`](../experiments/composition-v7-20260928-plan.json),
SHA-256 `7e51fe4ab1928e6805dda6443c2c8a8fb0d4ca0e796f2b6fca9ef0c1ad688d0c`.
It uses the same frozen external encoder and the unchanged genesis incumbent,
with `evals/frozen/public-audit-v7-260.json` and the 497-row reviewed v8 corpus.
The 500-row v7 corpus remains preserved and rejected before training; only the
three lesson IDs flagged for digit-normalized template overlap were removed.
The final [input-separation report](../evals/evidence/v7-input-separation-v8-corpus.json)
passed and has SHA-256
`f00563c656abe15e9e05cb02a93dc04f764f5e510cd556fe338d1119b67192e7`.
No ordinary chat, held-out vocabulary, or third-party text enters weight
training. The synthetic corpus received same-party agent review, not
independent human review. [Composition v7](COMPOSITION_V7.md) provides the
authoritative outcome and preserved candidate evidence.

The completed run rejected all three seeds, each with `COMPLETE` evidence.
The primary seed scored 39/80 fresh, 40/40 retention, 5/40 OOV, 75/80
composition, and 10/20 calibration. Robustness seeds scored 41/80, 40/40,
10/40, 75/80, 8/20 and 41/80, 40/40, 8/40, 75/80, 11/20. Every challenger
tied the incumbent's 75/80 composition result. Seeds 52031 and 52047 also
regressed the held-out `ameliorate` audit from 2/4 to 1/4. The result is
`PRIMARY_REJECTED_NO_RECOMMENDATION`; unchanged incumbent weights remain active.

The [aggregate evidence](../experiments/composition-v7-20260928-evidence/aggregate-evidence.json)
file SHA-256 is
`50cff3716b62d0c06e9b0aec2ffa3d6e49e0dee3a7dd27b92620d0ea1b625f0f`;
canonical integrity SHA-256 is
`5fff30454c7aca1dd1a0e32e6bdaa7c92658f802fb93ce620fc48e416bccd798`.
The separate development probes improved from 1/8 to 8/8 with supplied
proposals. This is evidence for the runtime repair, while the frozen evaluation
shows no learned composition improvement. V7 artifacts and failures are now
development-influenced and remain immutable.

## Previous completed v6 experiment

The immutable plan is
[`experiments/frozen-encoder-v6-20260928-plan.json`](../experiments/frozen-encoder-v6-20260928-plan.json).
It pins the encoder manifest, incumbent, then-current v6 suite, calibration fit,
held-out vocabulary, reviewed corpus, trainer-discovered inputs, executable
sources, environment, three seeds, and output root before training. The corpus
contains 197 permission-clean, repository-authored synthetic reviewed rows. It
uses same-party agent review, has no independent human review, and contains no
ordinary chat or third-party text.

The result is preserved at
[`experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`](../experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json).
Its file SHA-256 is
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`
and its canonical integrity SHA-256 is
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.
The outcome is `PRIMARY_REJECTED_NO_RECOMMENDATION`.

| Model | Fresh | Retention | OOV | Composition | Calibration |
|---|---:|---:|---:|---:|---:|
| Public incumbent | 48/80 | 40/40 | 3/40 | 50/80 | 0/20 |
| Seed 52031 challenger | 60/80 | 40/40 | 16/40 | 50/80 | 4/20 |
| Seed 52047 challenger | 58/80 | 40/40 | 9/40 | 50/80 | 4/20 |
| Seed 52069 challenger | 62/80 | 40/40 | 18/40 | 50/80 | 3/20 |

All three runs completed and all three decisions were `REJECT`. Each challenger
tied composition at 50/80, where strict improvement was required. Each also
regressed `held-out-vocabulary:ameliorate`: the incumbent scored 3/4 and the
three challengers scored 0/4, 1/4, and 0/4. Fresh and aggregate OOV gains do not
override either failed gate. The public incumbent remains the hashed-feature
genesis checkpoint with SHA-256
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.

The v6 suite, its failures, and the corpus have now influenced development.
They must not be patched or reused as fresh evidence. A next experiment must
preregister a newly versioned untouched suite and corpus boundary and should
target multi-frame/composition behavior and the `ameliorate` regression without
granting generated text any state authority.

## Runtime security boundary

`kev.encoders.load_supported_local_encoder()`:

- accepts only the allowlisted `hf-bert-mean-pooling-normalize-v1` recipe;
- reads `BertModel` configuration, tokenizer JSON, pooling recipe, module list,
  and safetensors from verified local bytes;
- does not call `from_pretrained`, Hub APIs, arbitrary dynamic model code, or a
  network loader;
- hashes the exact bytes consumed by the loader, closing the verify/load gap;
- reproduces the documented mean-pooling plus L2-normalization recipe; and
- freezes every encoder parameter and keeps the encoder in evaluation mode.

The encoder produces features only. Parser-bound slots still decide what can
enter explicit state.

## Challenger, calibration, and evaluation

Use the local manifest only through the closed evolution path:

```text
kev evolve --lessons reviewed-lessons.jsonl --encoder-manifest models/local/encoders/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/encoder-manifest.json
```

The trainer computes the frozen sentence embeddings once and preserves a
hash-bound embedding cache. It initializes the new 384-to-80 projection
deterministically, copies the incumbent's shape-compatible frame and
cardinality heads, and trains only that projection and those typed heads. The
small challenger checkpoint contains no encoder weights; it binds the external
manifest path and hash, parent checkpoint hash, reviewed lessons, exclusions,
embedding values, architecture transition, and complete loss history. Its
training-source manifest independently binds the sentence trainer, encoder
runtime, and the shared `semantic_breadth.py` target/loss implementation in
`training_inputs`, the checkpoint, and the receipt. Evolution re-hashes all
three source files before accepting that evidence.

For the previous v6 experiment, the canonical training-source-manifest SHA-256
was `429d7674d614dddef1a88ccaea6f53385551a50fa41d479a2a225641a2b95359`.
The actual objective was
`L_relation + 0.20 L_cardinality + 0.35 L_contrastive`. Every loss value was
logged in the receipt and excluded from promotion.

`CheckpointPredictor`, calibration, `kev eval`, and `AliveRuntime` recognize
this checkpoint schema. `kev eval` ordinarily follows the checkpoint's pinned
manifest; `--incumbent-encoder-manifest` and
`--challenger-encoder-manifest` permit an explicit relocated copy only when its
hash matches the checkpoint.

Calibration still writes a distinct checkpoint and bound receipt. Training
loss remains evidence only and is not a promotion feature. A rejected
challenger and its cache, receipts, raw failures, and evaluation card remain on
disk. The preregistered runner independently reloads both checkpoints,
reproduces each complete evaluation card, verifies the calibration receipt's
source/output checkpoint and fit/promotion-suite bindings, and records a final
hash inventory for every seed before it writes aggregate evidence.

No selected encoder or completed training run can activate itself.
Qualification uses an incumbent compare-and-swap over the evaluated path,
SHA-256, and monotonic model generation; stale results reject. The previous v6
experiment wrote one `MODEL_REJECTED` receipt per seed, preserved all three
candidates, and left the active model and generation unchanged. See
[Reproducibility](REPRODUCIBILITY.md) for the immutable plan verification,
one-shot run warning, exact result paths, and independent checkpoint replay.

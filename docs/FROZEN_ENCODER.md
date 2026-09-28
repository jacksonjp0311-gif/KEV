# Frozen Sentence Encoder Acquisition and Runtime

KEV's selected broader-language feature source is
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
pinned to upstream commit
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`. Its model card declares
`Apache-2.0`. This is a frozen language-feature component, not cognitive state,
an authority source, or a model-promotion decision.

The selected encoder is not bundled in Git and is not the public incumbent.
Selection and integration do not establish an improvement. The first bounded
three-seed experiment has now run, and every challenger was rejected. Only a
future challenger that passes every frozen evaluation gate may become an
incumbent.

## Pinned evidence

| Artifact | SHA-256 |
|---|---|
| Committed acquisition source record | `a46078dd1c13d9114bff07a891a1a903c1281681cf6dd3002e8d9e5ea8da03e9` |
| Upstream `model.safetensors` | `53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db` |
| Upstream model card | `dcd602d2fd35c203a247304a06fec6654a12f7941b739f9221a064fe8dc3b7f0` |
| Apache-2.0 license text | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| Reproducible local encoder manifest | `cfd8f8bc3413f31ab927ad7d78eb380c3359fdb2893061075d2369dfec81ea57` |
| Preregistered experiment plan | `c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca` |
| Reviewed corpus | `c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed` |
| Aggregate experiment evidence | `c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77` |

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

Install the explicit optional dependencies:

```text
python -m pip install -e ".[semantic-encoder]"
```

From a clean repository root, run the exact command pinned in the source
record:

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

## Completed bounded experiment

The immutable plan is
[`experiments/frozen-encoder-v6-20260928-plan.json`](../experiments/frozen-encoder-v6-20260928-plan.json).
It pins the encoder manifest, incumbent, current v6 suite, calibration fit,
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

For the completed experiment, the canonical training-source-manifest SHA-256
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
SHA-256, and monotonic model generation; stale results reject. The completed
experiment wrote one `MODEL_REJECTED` receipt per seed, preserved all three
candidates, and left the active model and generation unchanged. See
[Reproducibility](REPRODUCIBILITY.md) for the immutable plan verification,
one-shot run warning, exact result paths, and independent checkpoint replay.

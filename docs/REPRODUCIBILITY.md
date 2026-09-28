# Reproducing KEV v0.52 Evidence

Use Python 3.12 for the published CPU research environment. The package also
supports Python 3.11–3.13, but that compatibility is not a claim of identical
numerical results. Verification and evaluation do not train or activate a
model. A new isolated evolution run trains a challenger and may activate it
only inside its named state directory.

## 1. Verify shipped bytes

```text
python -m pip install torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-research.txt -e ".[dev,semantic-encoder]"
python -m kev.cli doctor
python -m pytest -q -p no:cacheprovider tests_public
```

`doctor` uses `models/registry.json` as its local root of trust. It verifies the
pinned v11 evidence bytes before trusting that evidence's manifest path or
contents, then checks the checkpoint, manifest hash, suite byte and canonical
hashes, eval-card byte and canonical report hashes, development-only known
failures, predecessor links, calibration slice, held-out vocabulary, and every
preserved manifest artifact.

The repository's `.gitattributes` marks `evals/`, `models/public/`, and the
public model registry as byte-preserved (`-text`). This prevents Git line-ending
conversion from changing content-addressed evidence on Windows or Linux.

The [research constraints](../constraints-research.txt) pin package versions,
not wheel hashes or every hardware property. Byte-for-byte numerical replay
requires the recorded experiment runtime, CPU backend, source versions, and
artifact bindings. Other builds or hardware may produce different floating-
point values; record those differences rather than claiming universal byte
equality.

## 2. Replay the current public baseline

Choose a new output path; eval reports are immutable and refuse replacement.

```text
python -m kev.cli eval --incumbent models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt --challenger models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt --suite evals/frozen/public-audit-v7-260.json --output replay-v7.json
```

Expected evidence:

- output file SHA-256:
  `b124cae0447847225f4c7edc54e38f97517e24ae13ce809708ad0f1dd84ca4a0`;
- canonical `report_sha256`:
  `b0b9a10f727a9c5c7849a6b2c570b0d7064c10f8ce03f4212c67e490ead3decf`;
- decision: `REJECT`, because fresh and composition tie;
- audit policy: all 25 reported families are checked independently for
  no regression; missing or size-mismatched family metrics reject;
- exact match: fresh 26/80, retention 40/40, OOV 2/40,
  composition 75/80, calibration 1/20; and
- raw incumbent failures: 116.

The reference copy is
[`evals/evidence/v7-genesis-baseline.json`](../evals/evidence/v7-genesis-baseline.json).
It is bound by
[`models/public/incumbent-evidence-v11.json`](../models/public/incumbent-evidence-v11.json).
The v11 evidence file SHA-256 is
`146bd0ed36728df46566d33002b9b253b4fb7d06d6f41a031c2dfba5dd2de770`.
The frozen manifest SHA-256 is
`f8784d55629f86377ca80aadf52c838e40b20c497c28653b8605d5e505f6a036`;
the suite file and canonical hashes are
`e22bdbea5f07c8b44cf5a1684a2b8433be6b252d74ddf86ef0252b736dbe9c76`
and `d3590d6f9eda20a3510df94f7473aeed45c5d616aadfe4ee7fb5f452f0036f28`.
Repository artifact paths
and newline serialization are stable across supported platforms.

One observed hosted numerical replay instead has file SHA-256
`8a50f0fa1b92697681e0688c509b3a30c3dd2bdc03bac2ea91101504879cccc6`:
one confidence differs by one millionth, with changed derived calibration
summaries and hashes but identical typed predictions, failures, accuracy
counts, and reject decision. Its full report and provenance are retained in
the [explicit replay-variant manifest](../experiments/composition-v7-portability/replay-variants-v1.json).
The test accepts only exact declared report bytes, independently checks their
semantic identity and integrity, and rejects any unknown numerical variant.
The registry's original baseline is not replaced. Do not interpret this finite
record of observed replays as universal bitwise reproducibility.

## 3. Replay the historical claim without inventing evidence

```text
python -m kev.cli historical-replay
```

This reproduces only what was published: parent 190/260, challenger 190/260,
and `REJECT` on a strict fresh tie. The output explicitly marks the original
item texts, raw failures, and checkpoint hashes unavailable.

## 4. Current composition experiment

[Composition v7](COMPOSITION_V7.md) is the authoritative protocol and result
record for `composition-v7-20260928`. It identifies the frozen plan, candidate
paths, receipts, score cards, decisions, and replay instructions. The input is
the immutable 497-row v8 corpus; the rejected 500-row v7 corpus remains
preserved with its pretraining separation audit. No evaluation prompt was
edited to resolve the three training-template overlaps.

The developer-proposal runtime probes at
[`experiments/composition-v7-development/probes.json`](../experiments/composition-v7-development/probes.json)
improved from 1/8 to 8/8. They are development evidence, not promotion evidence
or a trained-model measurement. The current baseline uses unchanged genesis
weights with the new runtime, so its scores must not be compared directly with
v6 as evidence of weight improvement.

The completed [plan](../experiments/composition-v7-20260928-plan.json) has
SHA-256 `7e51fe4ab1928e6805dda6443c2c8a8fb0d4ca0e796f2b6fca9ef0c1ad688d0c`.
Its [aggregate evidence](../experiments/composition-v7-20260928-evidence/aggregate-evidence.json)
has file SHA-256
`50cff3716b62d0c06e9b0aec2ffa3d6e49e0dee3a7dd27b92620d0ea1b625f0f`
and canonical integrity SHA-256
`5fff30454c7aca1dd1a0e32e6bdaa7c92658f802fb93ce620fc48e416bccd798`.
All three seed bundles are `COMPLETE` and rejected for `COMPOSITION_TIE`.
The primary seed scores fresh 39/80, retention 40/40, OOV 5/40, composition
75/80, calibration 10/20; its 91 failures are preserved. The outcome is
`PRIMARY_REJECTED_NO_RECOMMENDATION`. No weights qualified.

The one-shot plan's output root already exists; `verify` and `run` deliberately
refuse to overwrite it. Preserve those bytes. To replay the primary checkpoint
in the original recorded runtime to a new report, use:

```text
python -m kev.cli eval --incumbent models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt --challenger experiments/composition-v7-20260928-evidence/seed-results/seed-52031/candidates/20260928-135231-c78eeb1b91-bf84a45f/calibrated/semantic-breadth.calibrated.pt --suite evals/frozen/public-audit-v7-260.json --challenger-temperature experiments/composition-v7-20260928-evidence/seed-results/seed-52031/candidates/20260928-135231-c78eeb1b91-bf84a45f/calibrated/calibration-receipt.json --challenger-encoder-manifest models/local/encoders/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/encoder-manifest.json --output replay-v7-seed-52031.json
```

The original primary eval-card file SHA-256 is
`0806f35c8e3801873ec0cfe1e80fda0e3de061e8872176166ef9ddf068596afb`.
Absolute provenance paths in generated calibration/run receipts currently
limit relocation. A different workspace or runtime must validate its bindings
and report derivative hashes rather than asserting original outer byte equality.
All three stored checkpoint replays passed during the recorded experiment.

## 5. Verify the previous v6 frozen-encoder experiment

The following scores are preserved historical results. Exact replay needs the
v6 source/runtime recorded in its plan (the prior source commit is
`1f5d9f322876d3e9ad7500b543521a64a3de5be1`), in a separate checkout, plus its
recorded CPU backend. Running old checkpoints through the current v7 runtime
is a new measurement and is not expected to reproduce the old card bytes.

Acquire and verify the external encoder bytes first by following
[Frozen Encoder](FROZEN_ENCODER.md). Before execution, the immutable plan was
checked without training with:

```text
python scripts/run_preregistered_experiment.py verify --plan experiments/frozen-encoder-v6-20260928-plan.json
```

That verifier deliberately requires the plan's output root not to exist. It now
refuses because the preserved result directory exists; this is expected
one-shot behavior, not an instruction to remove the evidence. The current plan
bytes can be checked non-destructively with:

```text
python -c "import hashlib, pathlib; p=pathlib.Path('experiments/frozen-encoder-v6-20260928-plan.json'); print(hashlib.sha256(p.read_bytes()).hexdigest())"
```

The expected plan SHA-256 is
`c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`.
The one-shot command that produced the preserved experiment was:

```text
python scripts/run_preregistered_experiment.py run --plan experiments/frozen-encoder-v6-20260928-plan.json
```

Do **not** run that command again expecting it to reuse the committed output
root. The plan names
`experiments/frozen-encoder-v6-20260928-evidence/`, which now exists, and the
runner deliberately refuses overwrite or append. Do not delete that evidence
to make the command pass. A new execution requires a newly preregistered plan,
a new output root, and—because v6 has now influenced development—a newly frozen
untouched suite and corpus boundary.

The completed result paths are:

| Evidence | Path |
|---|---|
| Aggregate | [`experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`](../experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json) |
| Preserved plan copy | [`experiments/frozen-encoder-v6-20260928-evidence/preregistered-plan.json`](../experiments/frozen-encoder-v6-20260928-evidence/preregistered-plan.json) |
| Seed 52031 run | [`experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52031/candidates/20260928-130706-c626da3350-68bddca7/`](../experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52031/candidates/20260928-130706-c626da3350-68bddca7/) |
| Seed 52047 run | [`experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52047/candidates/20260928-130734-c626da3350-6a4efdbc/`](../experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52047/candidates/20260928-130734-c626da3350-6a4efdbc/) |
| Seed 52069 run | [`experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52069/candidates/20260928-130750-c626da3350-9f68c272/`](../experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52069/candidates/20260928-130750-c626da3350-9f68c272/) |

The aggregate file SHA-256 is
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`;
its canonical integrity SHA-256 is
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.
Expected outcome: `PRIMARY_REJECTED_NO_RECOMMENDATION`.

| Model | Fresh | Retention | OOV | Composition | Calibration |
|---|---:|---:|---:|---:|---:|
| Incumbent | 48/80 | 40/40 | 3/40 | 50/80 | 0/20 |
| Seed 52031 | 60/80 | 40/40 | 16/40 | 50/80 | 4/20 |
| Seed 52047 | 58/80 | 40/40 | 9/40 | 50/80 | 4/20 |
| Seed 52069 | 62/80 | 40/40 | 18/40 | 50/80 | 3/20 |

Every seed is complete and rejected for `COMPOSITION_TIE` plus
`held-out-vocabulary:ameliorate` regression. The incumbent scored 3/4 on that
family; seeds 52031, 52047, and 52069 scored 0/4, 1/4, and 0/4. The public
incumbent therefore remains SHA-256
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.

### Independent checkpoint replay

The experiment runner did not trust stored score summaries. For each seed it
reloaded the exact incumbent and calibrated challenger, verified the
calibration receipt's source/output checkpoint hashes and its calibration-fit
and v6 promotion-suite bindings, recomputed every prediction, and required the
entire regenerated evaluation card to equal the stored card. It then re-hashed
the final per-seed file inventory and revalidated all seeds before writing the
aggregate.

A skeptic using the preserved workspace and acquired encoder bundle can replay
a seed to a new output file. For seed 52031, for example:

```text
python -m kev.cli eval --incumbent models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt --challenger experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52031/candidates/20260928-130706-c626da3350-68bddca7/calibrated/semantic-breadth.calibrated.pt --suite evals/frozen/public-audit-v6-260.json --challenger-temperature experiments/frozen-encoder-v6-20260928-evidence/seed-results/seed-52031/candidates/20260928-130706-c626da3350-68bddca7/calibrated/calibration-receipt.json --challenger-encoder-manifest models/local/encoders/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41/encoder-manifest.json --output replay-seed-52031.json
```

Use a different new output path for each replay; evaluation reports are
immutable. The other exact run directories are listed above. Generated run and
calibration receipts currently retain absolute provenance paths. A relocated
clone must not claim the original outer byte hashes unless those bindings still
validate; a derivative replay should instead compare checkpoint and suite
hashes, split and audit metrics, reason codes, and raw predictions and identify
its newly written receipt/report hashes.

The three training receipts share source-manifest canonical SHA-256
`429d7674d614dddef1a88ccaea6f53385551a50fa41d479a2a225641a2b95359`,
binding the sentence trainer, encoder runtime, and shared semantic loss code.
Their actual objective is relation loss plus 0.20 cardinality loss plus 0.35
contrastive loss. Loss is logged but never used for promotion. Final evidence
inventories include rejected checkpoints, frozen-embedding caches, training
and calibration receipts, evaluation cards, evolution results, state, and
ledger files. Each seed ledger contains one `MODEL_REJECTED`; no model
generation or active pointer changed.

## 6. Exercise a new closed evolution run

For a separate, non-public exercise, create lessons through the `DRAFT` →
`REVIEWED` flow, export them, and choose new state and output directories:

```text
python -m kev.cli evolve --lessons reviewed-lessons.jsonl --state-dir replay-state-new --output-root replay-candidates-new
```

The run writes a unique candidate directory containing the uncalibrated
challenger, optional calibrated child, training and calibration receipts, raw
eval failures, eval card, and evolution result. Its ledger receives exactly one
`MODEL_REJECTED` or `MODEL_QUALIFIED` terminal decision. Training loss is
present only in the training receipt and is not a gate input. Qualification is
compare-and-swap guarded by the incumbent path, SHA-256, and monotonic model
generation; a stale result rejects. On qualification the old incumbent remains
on disk.

## Frozen-pack rule

The scripts under `scripts/` document how each public generation was made and
refuse to overwrite existing artifacts. Do not rerun them into the committed
paths. If an audit item has influenced development, preserve that pack and its
failures, then create a new version with new hashes.

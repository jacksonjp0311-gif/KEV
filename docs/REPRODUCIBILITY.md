# Reproducing KEV v0.52 Evidence

Run these commands from a clean repository root with Python 3.11–3.13. The
verification and replay steps in sections 1–3 do not train or activate a
model; the isolated evolution exercise in section 4 does train a challenger
and may activate it only inside the named replay state directory.

## 1. Verify shipped bytes

```text
python -m pip install -e ".[dev]"
python -m kev.cli doctor
python -m pytest -q -p no:cacheprovider tests_public
```

`doctor` uses `models/registry.json` as its local root of trust. It verifies the
pinned v10 evidence bytes before trusting that evidence's manifest path or
contents, then checks the checkpoint, manifest hash, suite byte and canonical
hashes, eval-card byte and canonical report hashes, development-only known
failures, predecessor links, calibration slice, held-out vocabulary, and every
preserved manifest artifact.

The repository's `.gitattributes` marks `evals/`, `models/public/`, and the
public model registry as byte-preserved (`-text`). This prevents Git line-ending
conversion from changing content-addressed evidence on Windows or Linux.

## 2. Replay the current public baseline

Choose a new output path; eval reports are immutable and refuse replacement.

```text
python -m kev.cli eval --incumbent models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt --challenger models/public/semantic-breadth-genesis-sha256-8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6.pt --suite evals/frozen/public-audit-v6-260.json --output replay-v6.json
```

Expected evidence:

- output file SHA-256:
  `881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4`;
- canonical `report_sha256`:
  `e6c768c496048afbdcf56554700ac2cc19e9c09d013139e2c7c53cdb3a9f658e`;
- decision: `REJECT`, because fresh and composition tie;
- audit policy: all 28 reported families are checked independently for
  no regression; missing or size-mismatched family metrics reject;
- exact match: fresh 48/80, retention 40/40, OOV 3/40,
  composition 50/80, calibration 0/20; and
- raw incumbent failures: 119.

The reference copy is
[`evals/evidence/v6-genesis-baseline.json`](../evals/evidence/v6-genesis-baseline.json).
It is bound by
[`models/public/incumbent-evidence-v10.json`](../models/public/incumbent-evidence-v10.json).
The v10 evidence file SHA-256 is
`69dc2c3c9bffac32c4c8d78929ccb25e70b51c47a2760d3b0f44783e545acd21`.
The frozen manifest SHA-256 is
`07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`;
the suite file and canonical hashes are
`a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029`
and `3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`.
Repository artifact paths
and newline serialization are stable across supported platforms.

## 3. Replay the historical claim without inventing evidence

```text
python -m kev.cli historical-replay
```

This reproduces only what was published: parent 190/260, challenger 190/260,
and `REJECT` on a strict fresh tie. The output explicitly marks the original
item texts, raw failures, and checkpoint hashes unavailable.

## 4. Verify the completed frozen-encoder experiment

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

## 5. Exercise a new closed evolution run

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

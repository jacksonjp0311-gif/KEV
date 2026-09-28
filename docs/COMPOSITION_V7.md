# KEV alpha.2 composition experiment

KEV remains experimental research software. This update separates a measured
runtime repair from the decision about new model weights. The previous runtime
and v6 experiment are preserved in Git commit
`1f5d9f322876d3e9ad7500b543521a64a3de5be1`.

## Runtime change

Global frame-kind proposals used to be assigned to clauses by position and
could bind a kind only once. The new fallback tries each proposed kind against
every available clause. It supports repeated kinds, keeps exact source slots,
and abstains when different interpretations overlap or cardinality cannot
select a unique result. Narrow span hints cannot hide enclosing negation or
modality. Deterministic parser results remain authoritative.

The [development evidence](../experiments/composition-v7-development/probes.json)
preserves both source snapshots and eight raw before/after probes: expected
outcomes improved from 1/8 to 8/8, with the parser control unchanged. These
probes use developer-supplied proposals, not a learned model, and are explicitly
ineligible for promotion. Replay them without Git history or model downloads:

```text
python scripts/measure_composition_binding_v7.py --check
```

## Frozen experiment boundary

The [protocol](../experiments/composition-v7-20260928-protocol.md) and
[machine-readable plan](../experiments/composition-v7-20260928-plan.json) fix
three seeds, 800 steps each, retention epsilon zero, and primary seed 52031.
Seeds 52047 and 52069 are robustness checks and cannot substitute for it.
The plan SHA-256 is
`7e51fe4ab1928e6805dda6443c2c8a8fb0d4ca0e796f2b6fca9ef0c1ad688d0c`.

The [v7 evaluation pack](../evals/frozen/public-audit-v7-260.json) has 80 fresh,
40 retention, 40 OOV, 80 composition, and 20 calibration items, with 25 audit
families. Every composition target is an exact semantic frame multiset.
The evaluation author did not read the new corpus or runtime changes, and the
corpus/runtime authors did not read its prompts. It remains same-party synthetic
evaluation, without independent human or external review.

The first 500-row corpus failed a pre-training number-template overlap check
on three retained receipt lessons. Its bytes and
[rejection](../evals/evidence/v7-input-separation.json) are preserved. The
[v8 corpus](../training/reviewed/semantic-frame-paraphrases-v8-reviewed.jsonl)
removes only those three rows: 497 lessons, 157 exact semantic paraphrase groups,
570 positive pairs. The remaining texts and targets were unchanged. The
[final separation audit](../evals/evidence/v7-input-separation-v8-corpus.json)
passed before training. No ordinary chat or third-party text entered training.

The encoder stays frozen. Only the projection and typed heads train, with
`L_relation + 0.20 L_cardinality + 0.35 L_contrastive`. Temperature uses the
separate calibration-fit-v2 slice. Loss cannot affect qualification.

The unchanged genesis checkpoint, measured with the same runtime as all
challengers, scored 26/80 fresh, 40/40 retention, 2/40 OOV, 75/80 composition,
and 1/20 calibration, with 116 raw failures. The previous v6 numbers use a
different suite and cannot be directly compared as progress percentages.

## Measured result

All three candidates were rejected. The
[aggregate evidence](../experiments/composition-v7-20260928-evidence/aggregate-evidence.json)
has outcome `PRIMARY_REJECTED_NO_RECOMMENDATION`, file SHA-256
`50cff3716b62d0c06e9b0aec2ffa3d6e49e0dee3a7dd27b92620d0ea1b625f0f`,
and canonical integrity SHA-256
`5fff30454c7aca1dd1a0e32e6bdaa7c92658f802fb93ce620fc48e416bccd798`.

| Model | Fresh | Retention | OOV | Composition | Calibration | Raw failures |
|---|---:|---:|---:|---:|---:|---:|
| Unchanged incumbent | 26/80 | 40/40 | 2/40 | 75/80 | 1/20 | 116 |
| 52031, primary | 39/80 | 40/40 | 5/40 | 75/80 | 10/20 | 91 |
| 52047, robustness | 41/80 | 40/40 | 10/40 | 75/80 | 8/20 | 86 |
| 52069, robustness | 41/80 | 40/40 | 8/40 | 75/80 | 11/20 | 85 |

Every candidate failed `COMPOSITION_TIE`. Seeds 52031 and 52047 also failed
`AUDIT_HELD_OUT_VOCABULARY_AMELIORATE_REGRESSED`: 2/4 incumbent versus 1/4
challenger. Seed 52069 avoided audit regression but still tied composition.
No seed can qualify on these results. Training loss was never a decision input.

All three evidence bundles are `COMPLETE`; independent checkpoint replay,
final inventories, common non-seed inputs, and ledger verification passed.
Every ledger has a `MODEL_REJECTED` event. All checkpoint bytes, caches,
calibration receipts, evaluation cards, and raw failures remain preserved.
The public model is still genesis SHA-256
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.
Only its evidence references advanced to v7/v11 before the experiment.

The runtime repair is supported by the development probes. The larger lesson
corpus did not improve measured composition. These are separate findings;
further work should test clause-local proposal learning under a new boundary,
not treat the controlled probes or better fresh scores as model qualification.

Local verification completed with 280 tests passing, plus lint, type checks,
compilation, immutable-artifact checks, and development-evidence replay.

Publication CI exposed a static-analysis portability issue in the existing
file mutex: mypy does not narrow `os.name` branches, and its Linux view omits
the Windows-only `msvcrt` declarations. The
[original failed check](https://github.com/jacksonjp0311-gif/KEV/actions/runs/36432931490)
is retained. CI now explicitly type-checks the Windows API view on both hosts,
without suppressing errors or changing frozen runtime sources. Runtime tests
still run separately on Linux and Windows; the dynamically imported POSIX
`fcntl` backend is runtime-tested, not statically typed.

### Observed numerical replay variation

Hosted CI also exposed a numerical portability limit, preserved in
[run 36434282077](https://github.com/jacksonjp0311-gif/KEV/actions/runs/36434282077).
One rounded, uncalibrated confidence changed from `0.517286` to `0.517287`;
its derived calibration summaries and integrity hashes changed accordingly.
Every typed prediction, exact slot, raw failure, accuracy count, and promotion
decision stayed identical. The captured runner reports AVX512 and Torch
`2.5.1+cpu`; the original local run reports AVX2 and `2.5.1+cu121` on CPU.
Those environment differences are recorded, not isolated as a proven cause.

The [numerical replay manifest](../experiments/composition-v7-portability/replay-variants-v1.json)
retains the full actual report, failure diagnostic, provenance, and exact leaf
differences. The original baseline remains unchanged. Tests require exact
bytes from one of the two explicitly recorded reports; an unknown hash still
fails and produces diagnostics. This is not a tolerance, automatic snapshot
refresh, evaluation-item change, or promotion-policy exception. Neither the
registry nor any experiment candidate or decision was changed. These two
observed replays do not establish universal bitwise portability.

The final local suite, including ten additional replay-contract tests, passed
all 290 tests. The 58 original experiment-plan artifact hashes remain unchanged.

## Artifacts and replay

| Artifact | SHA-256 |
|---|---|
| [v7 suite](../evals/frozen/public-audit-v7-260.json) | `e22bdbea5f07c8b44cf5a1684a2b8433be6b252d74ddf86ef0252b736dbe9c76` |
| [v7 manifest](../evals/frozen/manifest-v7.json) | `f8784d55629f86377ca80aadf52c838e40b20c497c28653b8605d5e505f6a036` |
| [v7 baseline](../evals/evidence/v7-genesis-baseline.json) | `b124cae0447847225f4c7edc54e38f97517e24ae13ce809708ad0f1dd84ca4a0` |
| [incumbent evidence v11](../models/public/incumbent-evidence-v11.json) | `146bd0ed36728df46566d33002b9b253b4fb7d06d6f41a031c2dfba5dd2de770` |
| [reviewed v8 corpus](../training/reviewed/semantic-frame-paraphrases-v8-reviewed.jsonl) | `c78eeb1b91d22eea88177f003971a884e5f519212acf140a6343f7b912f15a09` |
| [final input audit](../evals/evidence/v7-input-separation-v8-corpus.json) | `f00563c656abe15e9e05cb02a93dc04f764f5e510cd556fe338d1119b67192e7` |

Use Python 3.12 and the published constraints for the tested research dependency
versions. The experiment plan records the actual runtime/package versions;
the constraints are not a complete platform wheel lock.

```text
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c constraints-research.txt -e ".[dev,semantic-encoder]"
python -m kev.cli doctor
python -m pytest -q -p no:cacheprovider tests_public
```

The one-shot run command is preserved for provenance:

```text
python scripts/run_preregistered_experiment.py run --plan experiments/composition-v7-20260928-plan.json
```

It refuses an existing output directory; never delete evidence to rerun it.
Each stored evaluation was independently reproduced from both checkpoint
payloads before the aggregate was accepted. Candidate and calibration receipts
retain absolute provenance paths, so relocating a clone requires rebinding a
derivative replay rather than claiming the original outer report byte hash.
The exact encoder acquisition is described in [Frozen Encoder](FROZEN_ENCODER.md).

Once these results influence another change, keep v7 immutable and declare a
new evaluation boundary. Publishing a code update does not qualify its weights.

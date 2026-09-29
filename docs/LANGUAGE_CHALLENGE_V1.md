# Language challenge v1

Evaluation-only: no training, runtime edits, model activation, or promotion.
This pack tests the unchanged parser, the permissive all-kind symbolic control,
the genesis global head, and the retained clause-local-v1 checkpoint with both
global and clause-local inference. All five are specified before measurement.

The frozen boundary has 24 authored cases: four each for unfamiliar phrasing,
resolvable references, ambiguous references, interacting constraints, minimal
order/negation/number pairs, and explicit controls. Forty unchanged v7 retention
cases are reported separately. Targets preserve exact entities, values and
comparators. New positive targets have literal entity/number support spans.
Unresolved references must not become concretely bound frames; this measures
abstention, not an implemented clarification interface.

Review is same-party synthetic, not independent human review. The sample is
small and pairs are correlated. It is not a general intelligence benchmark,
a representative real-world sample, or the production promotion suite. No OOV
or calibrated-confidence claim is made. Novel full surfaces are checked against
all retained reviewed corpora, but constituent phrases may have been seen.

Protocol: `experiments/language-challenge-v1-boundary/protocol.json`, SHA-256
`8c777a15aaefefab25a416746ce43c96d3c139e4419761437f9b1f3a19a9878e`.
The protocol and source are committed before the first measurement. Do not
edit cases after results, train on their failures, or relabel reruns as fresh.

Commands from the repository root:

```text
python -m scripts.language_challenge_v1 run
python -m scripts.language_challenge_v1 verify
python -m scripts.language_challenge_v1 replay --receipt NEW_PATH.json
```

`run` creates evidence exclusively and refuses to replace a previous run.
`verify` checks input and output hashes and the ledger without downloading an
encoder. `replay` requires the pinned local MiniLM artifacts and recomputes every
mode, requiring exact raw-report equality. Receipt paths must be new.

Future training must exclude this pack and the earlier clause-local-v1 pack.
Both are separate research packs outside the older production v7 trainer's
automatic exclusion manifest; do not assume that older guard covers them.

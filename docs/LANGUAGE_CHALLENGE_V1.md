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
The protocol and source were pushed in commit `71bfd4ab829897cf641edc99ec4bb30bd689235a`
before the first measurement. Do not
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

## Completed result

All five approaches scored **13/24** on new cases and **40/40** on retention.
Excluding the four explicit controls, the challenge-only score is **9/20**.
There is no demonstrated learned advantage on this boundary.

| Family | Exact frames (every approach) |
|---|---:|
| Unfamiliar phrasing | 0/4 |
| Resolvable references | 0/4 |
| Ambiguous references, conservative abstention | 3/4 |
| Interacting constraints | 2/4 |
| Order / negation / number pairs | 4/4 |
| Explicit controls | 4/4 |
| Preserved retention | 40/40 |

The parser and all-kind control each emitted five unsupported frame candidates;
the three neural modes each emitted six. "Unsupported" here means outside the
authored target multiset, not proof that a persisted fact or tool action occurred.
Nothing was written into operational cognitive state. All approaches missed
twelve target frames and failed the same eleven complete-case judgments.

Examples of diagnosed boundaries:

- `Keep it below ...` can emit a subject literally named `it`, without an
  antecedent binding. Even an explicitly unspecified referent is not always
  rejected. A claim span alone is not proof of resolved entity identity.
- Broader phrasing can misbind subjects as `render_delay_down` or
  `for_checkout_latency`, or produce no frame at all.
- Clear order/number controls and explicit conflicting constraints can work
  while differently worded constraints are missed.

These observations identify reference resolution, entity binding, and parser
coverage as concrete bottlenecks. They do not establish that more head training
would solve them. Next implementation work should define a typed unresolved
reference/clarification boundary, then test any change on new, independently
reviewed cases where feasible. Preserve these cases as exposed diagnostic
evidence; do not patch them or claim a tuned rerun is fresh improvement.

Complete predictions, missing/unsupported frames, per-family scores, local
proposal traces, and a hash-chained evaluation ledger are in
`experiments/language-challenge-v1-evidence/`. Fresh inference reproduced every
raw report exactly; receipt: `experiments/language-challenge-v1-replay.json`.
The summary SHA-256 is
`7c5dce9be57d0d88bb997a9f3c37af0d0bb6494a271823b4aefb2de896e4b07e`;
the sealed inventory SHA-256 is
`bc4060dd8ef283f77700754b75bf3cd3429fd6d6e6855b5758c3524d73c0b138`.

Local validation passed 313 tests, lint, compilation, type checking, artifact
verification and exact inference replay. The command-result summary is retained
at `experiments/language-challenge-v1-validation.json`.

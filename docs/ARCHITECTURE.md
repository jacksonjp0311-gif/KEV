# KEV Architecture

KEV separates language, semantic meaning, cognitive state, memory, execution, observation, and learning.

1. **Language surface** — human input/output.
2. **Semantic layer** — phrasing → internal meaning.
3. **Grounded cognitive state** — exact values, provenance, revisions, goals, constraints, observations, predictions.
4. **Persistent state** — survives restarts and preserves superseded facts.
5. **Reasoning / bounded capabilities** — operates over state and declared tools.
6. **Observation** — records what actually occurred.
7. **Learning** — compiles reviewed evidence into challenger training.
8. **Evaluation / promotion** — rejects or qualifies challengers.

Core invariant: **language is an interface to cognition; it is not the cognitive state itself.**

```text
input → semantics → state → reasoning/action → observation
  ↑                                      ↓
  └──── language ← response ← learning/evaluation
```

Current bottleneck: semantic breadth and compositionality under unfamiliar language.

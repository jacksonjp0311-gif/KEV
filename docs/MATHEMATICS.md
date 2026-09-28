# Mathematics and the KEV Bridge

Given input (x), an encoder produces (h=f_θ(x)), normalized as (z=h/||h||₂).

A relation head estimates:

```text
P(r | x) = softmax(Wᵣ z + bᵣ)
```

The documented Semantic Core objective is:

```text
L = L_relation + 0.35 L_contrastive
```

For paraphrases of the same relation, training encourages cosine similarity toward 1 while separating different relations.

KEV separates semantic relation from exact payload:

```text
"red is obsolete; blue applies now"
→ RELATION = SUPERSEDES
→ OLD_VALUE = red
→ CURRENT_VALUE = blue
```

A simplified persistent state is:

```text
S_t = {facts, goals, constraints, observations, predictions, memory, evidence}
```

The next target is a set of compositional frames rather than one relation:

```text
F(x) = {f₁, f₂, …, f_k}
```

This is the bridge from semantic classification toward machine-operable cognitive state.

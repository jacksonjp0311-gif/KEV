# KEV

**A local-first research system for persistent, teachable AI: semantic state, grounded evidence, reviewed learning, candidate training, and measurable model evolution.**

> **Status:** experimental research software. KEV is not demonstrated AGI or superintelligence. Positive results and failed experiments are both preserved so capability claims remain tied to evidence.

## Why KEV exists

Most assistants treat a conversation as text to continue. KEV explores a different architecture: **language is an interface to persistent cognitive state**. Facts can be corrected, goals and constraints can survive restarts, observations and predictions are typed separately, lessons are reviewed before becoming training evidence, and new checkpoints remain challengers until evaluation supports promotion.

The research question is whether these pieces can support continual learning without confusing memory, fluent generation, tool execution, or lower loss with genuine capability improvement.

## Architecture

```text
human language
      ↓
semantic interpretation
      ↓
persistent cognitive state
 facts · revisions · goals · constraints
 observations · predictions · reviewed lessons
      ↓
memory / reasoning / bounded tools
      ↓
verified observations
      ↓
learning → challenger weights → evaluation
                               ↙          ↘
                            reject      qualify
```

The current research stack combines:

- **Grounded Cognitive State** — explicit evidence/state rather than treating prose as truth.
- **Semantic Core** — maps varied language toward stable meaning relations.
- **Semantic Breadth** — adds `GOAL`, `CONSTRAINT`, `OBSERVATION`, and `PREDICTION`.
- **KEV Alive** — persistent sessions, correction-aware facts, typed cognitive memory, reviewed lessons, a hash-chained experience ledger, and a local chat surface.

## The mathematical bridge

KEV's core principle is:

```text
Language ≠ cognitive state
```

Given language (x), a semantic encoder produces

```text
h = fθ(x)
z = h / ||h||₂
```

and a learned relation head estimates

```text
P(r | x) = softmax(Wᵣ z + bᵣ)
```

The documented Semantic Core objective combines classification and contrastive pressure:

```text
L = L_relation + 0.35 L_contrastive
```

Paraphrases of the same relation are pushed toward nearby regions of representation space. Exact payload values—names, numbers, identifiers—are separated from semantic meaning where possible, so the system can learn **the relation** instead of memorizing the payload.

The target bridge is:

```text
Language → Meaning → Grounded State → Reasoning → Action
         → Observation → Learning → Language
```

See [Architecture](docs/ARCHITECTURE.md) and [Mathematics](docs/MATHEMATICS.md).

## What has actually been demonstrated

KEV's research history includes both success and failure:

- reviewed teaching has produced measurable weight changes in bounded tasks;
- persistent facts survive restart, with corrections creating revisions instead of silently erasing history;
- semantic models can map multiple known phrasings toward common internal relations;
- challengers can be trained, evaluated, and rejected when they fail to improve;
- a reviewed-lesson challenger was **rejected after tying its incumbent at 190/260**;
- out-of-vocabulary audits exposed semantic brittleness and remain part of the research record.

These are **not** claims of general intelligence.

## Running KEV

KEV uses one PowerShell entry point:

```powershell
.\KEV.ps1 -Mode Install
.\KEV.ps1 -Mode AliveStatus
.\KEV.ps1 -Mode AliveVerifyLedger
.\KEV.ps1 -Mode AliveDesktop
```

Native Windows acceptance remains an explicit open validation item for the current consolidated branch.

## Teaching and model evolution

Ordinary conversation is not automatically training truth. Reviewed lessons are explicit:

```json
{"id":"lesson-001","intent":"CONSTRAINT","text":"Never overwrite the incumbent checkpoint in place."}
```

The intended loop is:

```text
teach → review → train challenger → fresh + retention evaluation → reject or qualify
```

**Training completed ≠ model improved.**

## Current research direction: compositional cognition

The next milestone is to extract multiple simultaneous cognitive frames from one utterance.

> Reduce latency below 50 ms, do not modify production, and the last measured latency was 73 ms.

Target:

```yaml
GOAL:
  latency_ms: "< 50"
CONSTRAINT:
  production_mutation: forbidden
OBSERVATION:
  latency_ms: 73
  source: previous_measurement
```

## For AI agents

If you are an AI/coding agent entering KEV, read:

1. [AGENT_START_HERE.md](AGENT_START_HERE.md)
2. [AI_AGENT_GUIDE.md](AI_AGENT_GUIDE.md)
3. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
4. [docs/MATHEMATICS.md](docs/MATHEMATICS.md)
5. [docs/ROADMAP.md](docs/ROADMAP.md)

Preserve provenance. Distinguish memory from weights. Preserve rejected candidates. Never overwrite an incumbent checkpoint. Never reinterpret a development audit as fresh evidence after it influenced a change. Never claim a tool ran without a receipt.

## Repository policy

Large model checkpoints, runtime state, private conversations, caches, and generated archives are intentionally excluded from ordinary Git history. See [MODEL_WEIGHTS.md](MODEL_WEIGHTS.md), [SECURITY.md](SECURITY.md), and [CONTRIBUTING.md](CONTRIBUTING.md).

## Project status

**Research alpha.** KEV is a persistent, teachable local AI research architecture—not currently a demonstrated AGI or superintelligent system.

**The ambition is large; the measurements stay literal.**

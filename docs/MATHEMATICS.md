# Mathematics and Evidence Contracts in KEV v0.52

This document describes the equations implemented by v0.52. They define a
bounded semantic proposal model, parser-grounded frame construction,
receipt-backed observations, and strict checkpoint comparison. They do not
measure or imply AGI or superintelligence.

## 1. Language is not state

Let an utterance be \(x\). KEV does not insert \(x\) directly into cognitive
state. Instead it constructs a set of validated frames:

```text
x -> exact claim spans -> typed slots -> validated frames -> explicit state
```

Formally, let \(P(x)\) be the deterministic parser output and
\(Q_\theta(x)\) the optional neural kind/cardinality proposal. The stored frame
set is

\[
F(x)=\operatorname{dedupe}\!\left(P(x)\cup V(x,Q_\theta(x))\right),
\]

where \(V\) is a constrained slot-binding function. \(V\) returns a frame only
when it can bind required values from an exact source span. Neural proposals
cannot supply slot values, and \(P(x)\) is never reduced by a proposal.

A frame is represented as

\[
f=(k,r,s,b,p,q,d),
\]

where \(k\) is the kind, \(r\) the relation, \(s\) the typed slot map, \(b\)
the claim boundary, \(p\) the provenance, \(q\) the evidence status, and \(d\)
the derivation. Its identifier is the SHA-256 of canonical semantic content,
claim offsets, and source identity. Surface text and provenance are preserved,
while `semantic_key()` compares only kind, relation, and typed canonical slots.

At the parser API boundary, the frame status rule is

\[
q=\texttt{VERIFIED}
\iff
k=\texttt{OBSERVATION}\land
\operatorname{source}(f)=\texttt{TOOL}\land
\operatorname{receiptHash}(f)\text{ has valid SHA-256 syntax}.
\]

All observations extracted from language remain `REPORTED`. This low-level rule
trusts caller-supplied source metadata; the integrated action path adds the
stronger requirement that the ledger sink accepted the receipt before the
executor returns a `VERIFIED` observation.

## 2. Active semantic proposal model

The public v0.52 incumbent uses a stable hashing front-end. Token
canonicalization produces unigrams and adjacent bigrams. Each feature string
is mapped with the first 32 bits of SHA-256 into \(D=1024\) buckets. If \(n_j\)
is the count in bucket \(j\), the input is

\[
\phi_j(x)=\log(1+n_j),\qquad \phi(x)\in\mathbb{R}^{1024}.
\]

The trainable encoder is

\[
u=\operatorname{LayerNorm}\!\left(\operatorname{GELU}(W_1\phi+b_1)\right),
\]

\[
h=W_2u+b_2,\qquad z=\frac{h}{\lVert h\rVert_2},\qquad z\in\mathbb{R}^{80}.
\]

Independent frame-kind logits and cardinality logits are

\[
a=W_fz+b_f\in\mathbb{R}^{13},
\qquad
c=W_kz+b_k\in\mathbb{R}^{7}.
\]

The 13 dimensions correspond to the nine inherited relations and the four
cognitive kinds. Cardinality classes are \(0,\ldots,6\).

With an optional positive calibration temperature \(T\), frame-kind proposal
scores are

\[
p_r=\sigma(a_r/T).
\]

Without a calibration artifact, \(T=1\) for computation and scores are labeled
`UNCALIBRATED`; they must not be presented as calibrated probabilities. The
cardinality estimate is

\[
\hat{k}=\arg\max_j c_j.
\]

Emitted proposal/evaluation scores use
\(\widetilde p_r=\operatorname{round}(p_r,6)\). This avoids implying
unwarranted precision and stabilizes public evidence across supported CPU math
backends; it does not convert an uncalibrated score into a probability claim.

For the default proposal threshold \(\tau=0.5\), the implementation selects

\[
S_\theta(x)=
\begin{cases}
\operatorname{Top}_{\min(\hat{k},6)}(\widetilde p), & \hat{k}>0,\\
\{r:\widetilde p_r\ge\tau\}, & \hat{k}=0.
\end{cases}
\]

This is a proposal set only. Parser-backed slot validation still decides
whether any proposed kind becomes a frame.

The public reference checkpoint has 285,092 parameters and is pinned by
[`models/public/incumbent-manifest.json`](../models/public/incumbent-manifest.json).
It is deterministic genesis initialization with zero training steps, so its
baseline is not evidence of learned improvement.

## 3. Implemented training objective

For \(N\) reviewed lessons and \(R=13\) frame kinds, let
\(Y\in\{0,1\}^{N\times R}\) be the multi-hot frame target matrix and let
\(F_i\) be the reviewed frame multiset for row \(i\). The cardinality target is

\[
y_i^{(k)}=\min\!\left(|F_i|,6\right).
\]

The relation loss is mean binary cross-entropy over independent frame logits:

\[
\mathcal{L}_{\text{relation}}
=-\frac{1}{NR}\sum_{i=1}^{N}\sum_{r=1}^{R}
\left[Y_{ir}\log\sigma(a_{ir})+(1-Y_{ir})\log(1-\sigma(a_{ir}))\right].
\]

The cardinality loss is multiclass cross-entropy:

\[
\mathcal{L}_{\text{cardinality}}
=-\frac{1}{N}\sum_{i=1}^{N}
\log\operatorname{softmax}(c_i)_{y_i^{(k)}}.
\]

The supervised contrastive term compares normalized embeddings only through
explicit reviewed paraphrase groups. A group is accepted only when it contains
at least two permission-bearing reviewed rows and every row has the same
canonical multiset of exact frame kind, relation, and typed slot values. Rows with
the same broad kind but different numbers, polarity, relation, or slots are
therefore not positive pairs. Let (g_i\ge0) identify an accepted group and
(g_i=-1) denote an ungrouped row. Define

\[
\mathcal{S}_{+}=\{(i,j):i\ne j\land g_i=g_j\land g_i\ge0\},
\qquad
\mathcal{S}_{-}=\{(i,j):i\ne j\land(i,j)\notin\mathcal{S}_{+}\}.
\]

When the corresponding set is non-empty,

\[
\mathcal{L}_{+}=\operatorname{mean}_{(i,j)\in\mathcal{S}_{+}}
\left(1-z_i^\top z_j\right),
\]

\[
\mathcal{L}_{-}=\operatorname{mean}_{(i,j)\in\mathcal{S}_{-}}
\max\left(0,z_i^\top z_j-0.20\right).
\]

`supervised_contrastive_loss()` averages the available components
\(\mathcal{L}_{+}\) and \(\mathcal{L}_{-}\); if neither exists, it returns
zero. The actual v0.52 objective is

\[
\boxed{
\mathcal{L}
=\mathcal{L}_{\text{relation}}
+0.20\,\mathcal{L}_{\text{cardinality}}
+0.35\,\mathcal{L}_{\text{contrastive}}
}
\]

Training uses AdamW with learning rate \(5\times10^{-4}\), weight decay
\(2\times10^{-4}\), and gradient norm clipping at 1.0. Every input row must be
explicitly `REVIEWED` and carry reviewer, review time, and permission
provenance. The pinned repository held-out vocabulary is always rejected before
optimization; caller-supplied terms can only extend that exclusion set. Exact
surface forms from every preserved frozen public suite, calibration-fit
artifact, and development-influenced known-failure audit are also loaded as
mandatory exclusions at the trainer boundary.
Repeated frame kinds remain separate cardinality targets even though the
frame-kind target is multi-hot. Paraphrase group membership, exact target
hashes, group hashes, and positive-pair counts are recorded in the training
receipt; ungrouped lessons still supervise the heads but never form positive
contrastive pairs.
During challenger training, the first hashed feature layer and normalization
parameters are frozen. Only the final 256-to-80 projection, frame-kind head,
and cardinality head receive gradients. The optional local sentence-encoder
adapter follows the same rule: encoder features are frozen; projection and
small typed heads are trainable. Before a caller-supplied local encoder can be
loaded, its local-only manifest binds permission review, license identity,
embedding dimension, and every declared file hash. This is an artifact
integrity boundary, not evidence that the public model already has broad
sentence semantics.

The selected adapter input is a normalized 384-dimensional mean-pooled output
from the hash-pinned `sentence-transformers/all-MiniLM-L6-v2` encoder. For token
states \(t_j\) and attention mask \(m_j\), its frozen feature is

\[
e(x)=\operatorname{normalize}_2\!\left(
\frac{\sum_j m_jt_j}{\max(1,\sum_j m_j)}
\right)\in\mathbb{R}^{384}.
\]

The trainable proposal embedding is
\(z=\operatorname{normalize}_2(W_pe+b_p)\in\mathbb{R}^{80}\). The encoder is
evaluated once per reviewed surface batch, and the exact float32 embedding
values are hash-bound in the training receipt. Its parameters never enter the
optimizer or challenger checkpoint.

The resulting checkpoint is an immutable `CHALLENGER`. Its checkpoint and
training receipt bind the parent, lesson-file, held-out-vocabulary, and trainer
source SHA-256 values; environment versions; seed; steps; objective; parameter
count; and loss history. Training loss is recorded for replay but excluded from
promotion.

For the completed v6 frozen-encoder experiment, the training-source manifest
bound `kev/sentence_training.py`, `kev/encoders.py`, and
`kev/uc51a2/semantic_breadth.py` under canonical SHA-256
`429d7674d614dddef1a88ccaea6f53385551a50fa41d479a2a225641a2b95359`.
All three seeds used the boxed objective above; the full loss trajectory was
logged but was not read by any promotion or aggregate-selection rule. The
reviewed corpus contains 197 permission-clean, repository-authored synthetic
rows. Its SHA-256 is
`c626da3350d873c1ccf62007b6a4c670b86c90840abbf3868db85c8aeae685ed`.
The review was same-party agent review, not independent human review, and the
corpus contains no ordinary chat or third-party text. Those limitations are
part of the evidence boundary.

## 4. Calibration

Calibration fits one scalar temperature without changing model weights. On a
separate calibration-fit set with multi-hot targets, it minimizes

\[
\mathcal{L}_{\text{cal}}(T)
=\operatorname{BCEWithLogits}(a/T,Y),
\qquad 0.05\le T\le20.
\]

The implementation optimizes \(\log T\) and writes a new immutable checkpoint
and `kev.calibration-receipt.v1`. Calibration and promotion item identifiers
and normalized surface forms must be disjoint. The receipt binds the source
checkpoint, calibrated child, calibration-fit bytes, and both file and
canonical hashes of the promotion suite. Evaluation marks scores `CALIBRATED`
only after those checkpoint and suite bindings verify. A temperature embedded
in a checkpoint without its receipt remains `UNCALIBRATED` evidence, and an
unreceipted generic scalar cannot mark scores calibrated. An already
temperature-scaled confidence is not scaled a second time. Calibration
status and temperature are evidence fields; they are not promotion gates in
v0.52.

## 5. Evaluation and strict promotion gates

For split \(s\), exact-match accuracy is

\[
A_s=\frac{\#\text{ exactly correct items in }s}{\#\text{ items in }s}.
\]

Compositional targets use order-independent exact multiset comparison, so
duplicate multiplicity remains significant. Frame-level
micro precision, recall, and F1 plus calibration diagnostics are recorded, but
the primary promotion values are the exact-match split accuracies.

Let \(I_s\) and \(C_s\) be incumbent and challenger accuracies on the same
frozen suite, let \(\varepsilon\ge0\) be the declared retention allowance, and
let \(\mathcal A\) be every audit family reported by either evaluation. A
challenger qualifies exactly when

\[
\boxed{
C_{\text{fresh}}>I_{\text{fresh}}
\;\land\;
C_{\text{retention}}+\varepsilon\ge I_{\text{retention}}
\;\land\;
C_{\text{oov}}\ge I_{\text{oov}}
\;\land\;
\left(C_{\text{composition}}>I_{\text{composition}}\right)
\;\land\;
\left(\forall a\in\mathcal A:\ C_a\ge I_a\right)
}
\]

where the composition conjunct applies whenever either report contains that
split. The default is \(\varepsilon=0\). A tie on fresh or composition rejects,
missing metrics reject, unequal split sizes reject, differing suite hashes
reject, and every audit family must exist with the same item total in both
reports. Training loss is never read by the gate.

The current suite, canonical hash, byte hash, calibration slice, and held-out
vocabulary are pinned by
[`evals/frozen/manifest-v7.json`](../evals/frozen/manifest-v7.json). Current
scores, source identities, and the baseline are in
[the v7 report](COMPOSITION_V7.md). The equations and strict gates above did
not change for this experiment.

The preceding v6 manifest has file SHA-256
`07f4e226773e8c68f58e0d6bbede97d4b34ffed40fc17f3b3e77ba695b69acce`.
The suite file SHA-256 is
`a91454be7c86bfa5e95ae873e3b618e4468961e00a1f9b8aaf206edf605ee029`
and its canonical SHA-256 is
`3d53b4b981c9ebb7ad6f4d6e73b82e56c19a275fdbd76922a865256c43ad334c`.
The item-level public genesis result is retained at
[`evals/evidence/v6-genesis-baseline.json`](../evals/evidence/v6-genesis-baseline.json)
with file SHA-256
`881e1f6911379f9bc158fde300147f1d41416b8076fbc1a972786c6ddc9915e4`.
V5 remains preserved as pre-v6 development evidence rather than being edited
after its freshness audit influenced the replacement of six items.
The older 190/260 tie is replayable only as the aggregate evidence that was
actually published; it is preserved at
[`evals/frozen/historical-190-260-replay.json`](../evals/frozen/historical-190-260-replay.json)
and is not misrepresented as a current item-level evaluation.

### Observed v6 frozen-encoder result

The preregistered plan at
[`experiments/frozen-encoder-v6-20260928-plan.json`](../experiments/frozen-encoder-v6-20260928-plan.json)
has SHA-256
`c39f830ea787d87b8e19dd52dfb2c65565ad1f2bd384cec2997ffa2c402d78ca`.
The observed exact-match counts were:

| Model | Fresh | Retention | OOV | Composition | Calibration |
|---|---:|---:|---:|---:|---:|
| Public incumbent | 48/80 | 40/40 | 3/40 | 50/80 | 0/20 |
| Seed 52031 challenger | 60/80 | 40/40 | 16/40 | 50/80 | 4/20 |
| Seed 52047 challenger | 58/80 | 40/40 | 9/40 | 50/80 | 4/20 |
| Seed 52069 challenger | 62/80 | 40/40 | 18/40 | 50/80 | 3/20 |

All challengers therefore satisfy the fresh, retention, and aggregate OOV
conditions but fail the strict composition condition because
(50/80=50/80). Each also regresses the four-item
`held-out-vocabulary:ameliorate` family from the incumbent's 3/4 to 0/4, 1/4,
and 0/4 respectively. Thus every per-seed decision is `REJECT`, regardless of
training loss or gains on other splits. The aggregate outcome is
`PRIMARY_REJECTED_NO_RECOMMENDATION`; the public incumbent remains checkpoint
SHA-256
`8f85375adcb63debafe3a9b34f095e066520ebb02585d5dbc6fb447c68bd3af6`.

## 6. Canonical hashes and evidence chains

For JSON-native evidence \(v\), KEV defines an internal deterministic encoding

\[
J_c(v)=\operatorname{UTF8}\!\left(
\operatorname{JSON}(v;\text{sorted keys, compact separators, finite numbers})
\right),
\]

and the canonical value hash

\[
H_c(v)=\operatorname{SHA256}(J_c(v)).
\]

This is an internal canonicalization contract, not a claim of full RFC 8785
compatibility. JSON artifacts retain both their byte-level file hash and, where
applicable, their canonical value hash.

For ledger event \(e_t\), whose body contains sequence number and
`prev_hash = h_{t-1}`,

\[
h_0=0^{64},
\qquad
h_t=H_c(e_t\setminus\{\texttt{hash}\}).
\]

The local head file records \((t,h_t,\text{ledger byte size})\). Verification
checks sequence continuity, previous hashes, event hashes, and agreement with
that local anchor. Because the anchor is local and unsigned, this is an
integrity chain rather than an independently witnessed transparency log.

For a state mutation, a pending journal commits to the intended state hash and
event body before either projection is considered complete. On recovery, KEV
restores the intended state if needed and appends the transaction-tagged event
only if it is absent. Thus retry is idempotent for one interrupted commit,
although the two files are not asserted to be a database transaction.

Model qualification adds an incumbent compare-and-swap condition. If
(m=(p,h,g)) is the evaluated incumbent's resolved path, checkpoint SHA-256,
and monotonic generation, qualification may commit only when

\[
(p_{\mathrm{current}},h_{\mathrm{current}},g_{\mathrm{current}})=m
\quad\land\quad
\operatorname{SHA256}(\operatorname{bytes}(p_{\mathrm{current}}))=h.
\]

On success the new model is installed with generation (g+1). A mismatch is
a stale-incumbent rejection, not permission to qualify against a different
parent. This closes an ABA case in which a path and hash could return to an
older value while the generation reveals intervening activations.

The preregistered runner also recomputes every evaluation from the captured
checkpoint bytes and frozen suite, requires byte-for-byte equality with the
stored evaluation card, verifies the calibration receipt's source/output and
fit/promotion bindings, and re-hashes the final per-seed file inventories. The
aggregate file
[`experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json`](../experiments/frozen-encoder-v6-20260928-evidence/aggregate-evidence.json)
has file SHA-256
`c45a30577d5022e77d44540ff0476c2629f0be33f6ff19f33423562998e08a77`
and canonical integrity SHA-256
`82654683003cd1255ca7c2b6039df42121acca9ccb2dbce45016c7de68ddc59a`.
Its aggregate record has no model-activation authority; the per-seed terminal
ledger events remain authoritative.

Action receipts use a labeled digest

\[
r=\texttt{"sha256:"}\,\Vert\,H_c(\text{receipt body}).
\]

A successful tool result becomes a verified observation only after the ledger
sink accepts this receipt. Thus

\[
\text{tool returned}\;\not\Rightarrow\;\text{verified observation},
\]

but

\[
\text{valid output}\land\text{persisted receipt}
\Rightarrow\text{verified observation}.
\]

## 7. Explicit-state planning and prediction scoring

Let \(\Pi\) project runtime state onto the six declared planning compartments:

\[
S_{\mathrm{explicit}}=\Pi(S)=
\{\text{facts, goals, constraints, observations, predictions, reviewed lessons}\}.
\]

An action plan stores the labeled state digest

\[
h_S=\texttt{"sha256:"}\,\Vert\,H_c(S_{\mathrm{explicit}})
\]

and stores a plan hash in the same labeled form over the tool name, validated
arguments, state hash, and optional goal identifier. A plan is executable only
when the tool is allowlisted, its input
matches the declared schema, and `production_mutation` is false. Output must
also match the declared schema before success can be receipted.

A prediction is compared only with a matching later observation; when records
lack timestamps, the caller asserts temporal order by supplying the later
observation collection. For a supported operator \(\circ\), expected value
\(e\), and actual value \(a\),

\[
\operatorname{score}=
\begin{cases}
1, & a\circ e\text{ is true},\\
0, & a\circ e\text{ is false},\\
\text{undefined}, & \text{no valid later match or comparison failure}.
\end{cases}
\]

These become `CONFIRMED`, `REFUTED`, or `UNRESOLVED` records with observation
and receipt references. They are state evidence, not automatic training truth.

## Claim boundary

These equations specify the current bounded implementation. The public genesis
head is untrained and uncalibrated, parser coverage is finite, the tool registry
is empty by default, and promotion metrics measure only the frozen suites named
in their hashed artifacts. None of these equations or measurements establishes
general intelligence.

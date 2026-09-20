# Novel Evaluation Algorithms — Technical Disclosure

Eight algorithms in `evalforge.novel`, each addressing a failure of conventional
LLM evaluation. For each one this document records the problem, the closest
prior art, the specific mechanism claimed to be new, a claim skeleton, and the
empirical evidence in the test suite.

---

## A necessary caveat on patentability

**This document does not establish that anything here is patentable, and I am
not able to assess that.** What follows is a technical disclosure written in a
form useful to a patent attorney. Turning any of it into a filing requires work
I have not done and cannot do:

- **A professional prior-art search.** The "prior art" sections below are
  written from my own knowledge of the literature, with a training cutoff. They
  are a starting point for a search, not a substitute for one. Machine
  learning evaluation is a crowded and fast-moving field; independent
  invention is common, and a real search may well surface closer art.
- **A subject-matter eligibility assessment.** In several jurisdictions,
  algorithms and mathematical methods face eligibility hurdles
  (in the US, *Alice/Mayo*; in the EPO, the technical-character requirement).
  Whether these claims clear those bars is a legal question requiring counsel.
- **A novelty and non-obviousness judgement**, which is the examiner's, informed
  by art I have not seen.

Two further points that materially affect filing:

- **Publishing this repository is a disclosure.** In most jurisdictions public
  disclosure starts a clock or forfeits rights outright. The US allows a
  12-month grace period; the EPO and most others do not. **If you intend to
  file, talk to an attorney before making this repository public.**
- Several components build on well-established published mathematics — Ville's
  inequality, split conformal prediction, item-response theory, Wasserstein
  barycenters, Thompson sampling, submodular maximization. The underlying
  mathematics is not novel and is not claimed to be. What is put forward as
  novel is the specific application, combination and adaptation to LLM
  evaluation, described in each section.

The honest summary: these are, I believe, genuinely new combinations that solve
real problems, and several are backed by measurable improvements over the
obvious baseline. Whether that clears the legal bar for patentability is
outside what I can determine.

---

## 1. Spectral Semantic Drift (SSD)

**Module:** `evalforge/novel/_drift.py`

**Problem.** Long conversations fail gradually. Every individual turn looks
acceptable to a per-turn evaluator, yet the assistant ends up answering a
different question than the one asked. Per-turn scoring cannot see this, and a
first-vs-last similarity cannot distinguish sustained *drift* from transient
*volatility*.

**Prior art.** Per-turn quality scoring; conversation-level aggregation;
embedding-similarity topic tracking; graph signal processing generally; spectral
clustering of documents.

**Claimed novelty — the dual-graph decomposition.** Two graphs over the same
turns answer two different questions:

- A **temporal graph** `W_ij = exp(-|i-j|/tau)`, whose edges depend only on turn
  distance. Its Laplacian eigenbasis is fixed and content-independent: low modes
  are smooth ramps, high modes alternate. The anchor-alignment signal
  `s_i = cos(v_i, anchor)` is projected onto it; low-band energy is drift,
  high-band energy is volatility.
- A **semantic graph** `W_ij = max(0, cos(v_i,v_j)) * exp(-|i-j|/tau)`, whose
  Fiedler vector partitions the conversation into two coherent blocks. Its sign
  change localises the drift **onset turn**; its eigenvalue says how cleanly the
  conversation splits at all.

The separation is load-bearing, and its necessity is demonstrable. Using a
single semantic graph for both — the obvious construction — weights edges *down*
exactly where the signal jumps, so an alternating conversation becomes smooth
with respect to its own graph and its volatility vanishes. This was observed
during development: a semantic-graph-only implementation scored an alternating
conversation at drift 0.216 / volatility 0.253; the dual-graph construction
scores the same conversation 0.004 / 0.987.

**Claim skeleton.**
> 1. A method for detecting semantic drift in a multi-turn exchange, comprising:
>    embedding each response turn; computing an anchor-alignment signal;
>    constructing a first graph whose edge weights are a function of turn
>    separation only; projecting the alignment signal onto the eigenbasis of
>    that first graph's Laplacian; attributing low-band spectral energy to
>    sustained drift and high-band energy to volatility; constructing a second
>    graph whose edge weights additionally depend on inter-turn semantic
>    similarity; and identifying a drift onset turn from a sign change of the
>    second graph's Fiedler vector.
> 2. The method of claim 1, wherein drift is reported only when alignment decays
>    between the first and second halves of the exchange.
> 3. The method of claim 1, wherein both quantities are gated on the absolute
>    power of the alignment signal.

**Evidence.** `TestSpectralSemanticDrift` pins the drift/volatility separation
and onset localisation.

---

## 2. Causal Ablation Groundedness with Provenance (CAG-P)

**Module:** `evalforge/novel/_causal.py`

**Problem.** "Groundedness 3.4" says something is wrong but not what, where, or
whether it matters. Correlational scoring fundamentally cannot localise the
failure, because a contradiction is *maximally similar* to the text it
contradicts.

**Prior art.** Claim decomposition; NLI entailment against retrieved passages;
attention and gradient attribution; citation/attribution benchmarks;
leave-one-out feature ablation in classical ML.

**Claimed novelty — four mechanisms combined.**

1. **Causal ablation over context spans**, not similarity: a span's contribution
   is the drop in claim support when it is *removed*.
2. **Necessity/sufficiency separation** — ablation score vs standalone score —
   which distinguishes redundant evidence from a single point of failure.
3. **Minimal witness extraction** by lazy-greedy submodular maximization. Claim
   support is monotone with diminishing returns in the span set, so greedy
   selection carries the standard `(1 - 1/e)` guarantee. The output is the
   smallest set of passages a human must read to check the claim.
4. **Corruption sensitivity** — the mechanism I consider most distinctive.
   The witness spans' numerals and proper nouns are perturbed and the claim
   re-scored. *If support survives, the claim was never relying on that
   evidence*, and the apparent grounding is incidental token overlap. This
   catches the specific failure that defeats similarity-based groundedness.

**Claim skeleton.**
> 1. A method for attributing claims in a generated response to source context,
>    comprising: segmenting the response into atomic claims and the context into
>    spans; for each claim, computing a necessity score per span by measuring the
>    reduction in support when that span is removed; computing a sufficiency
>    score per span from that span alone; selecting a minimal witness set by
>    greedy maximization of a monotone submodular support function to a threshold
>    fraction of full-context support; perturbing the factual commitments of the
>    witness spans and recomputing support to yield a corruption-sensitivity
>    score; and emitting a per-claim provenance certificate.
> 2. The method of claim 1, further comprising identifying as a phantom set
>    those claims whose support falls below a floor.
> 3. The method of claim 1, wherein a claim exceeding the support threshold with
>    corruption sensitivity below a floor is reported as incidentally grounded.

**Evidence.** `TestCausalGroundedness` verifies that a fabricated claim is
isolated as a phantom and that a claim fusing two facts selects both carrying
spans as its witness.

---

## 3. Judge Item-Response Calibration (JIRC)

**Module:** `evalforge/novel/_irt.py`

**Problem.** LLM-judge panels are combined by averaging or majority vote, which
treats every judge as equally trustworthy. Judges differ in severity,
discrimination and noise. Averaging lets a severe judge depress a leaderboard
and an uninformative judge dilute the signal, with no way to tell which.

**Prior art.** Item-response theory (decades old in psychometrics);
Dawid-Skene annotator models; LLM-as-judge ensembling; inter-annotator
agreement statistics. **The underlying IRT mathematics is not novel.**

**Claimed novelty — the application and its adaptations.** Fitting a graded
item-response model `x_ij = b_j + a_j*theta_i + noise(0, sigma_j^2)` to an **LLM
jury**, with: per-judge noise estimated from the panel itself rather than
assumed; alternating conditional estimation in which each judge's posterior
weight is *inverse to its own estimated noise*; sign and scale anchoring that
makes the fit reproducible **without gold labels**; and export of the
per-item posterior variance as a calibrated confidence signal that feeds
selective human review (and algorithm 5).

**Claim skeleton.**
> 1. A method for aggregating scores from a plurality of language-model judges,
>    comprising: fitting a latent-trait model in which each judge is
>    parameterised by a discrimination, a severity and a noise variance;
>    alternately estimating judge parameters by regression on current latent
>    quality and latent quality as a posterior mean weighting each judge
>    inversely to its estimated noise; resolving scale invariance by
>    standardising latent quality and sign invariance by fixing the aggregate
>    discrimination positive; and emitting a per-item posterior variance as a
>    confidence signal.
> 2. The method of claim 1, wherein items whose posterior variance exceeds a
>    threshold are routed to human review.

**Evidence.** `TestJudgeCalibration` shows recovery of an injected 1.5-point
severity offset to within 0.2, identification of a zero-discrimination judge,
and correlation with ground truth of **0.995 versus 0.937 for naive
averaging**.

---

## 4. Semantic Curvature Probe (SCP)

**Module:** `evalforge/novel/_curvature.py`

**Problem.** Self-consistency hallucination detectors measure the **variance** of
answers across resamples. Variance is first-order: it says answers are spread
out, not whether the spread is an orderly response to the input changing or a
sign the model is on unstable ground. A model that elaborates proportionally as
a question is rephrased is *sensitive*; a model that repeats itself twice and
then says something unrelated is *unstable*. Variance conflates them.

**Prior art.** Self-consistency decoding; SelfCheckGPT and sampling-based
hallucination detection; paraphrase-invariance testing; adversarial robustness
and Lipschitz analysis; semantic entropy.

**Claimed novelty — second-order geometry along an evenly spaced ladder.**
A *perturbation ladder* is constructed in which rung `k` applies exactly one
more meaning-preserving operator than rung `k-1`, making the rungs evenly
spaced by construction. Along each consecutive triple, the discrete second
difference `||y_{k+1} - 2y_k + y_{k-1}||` is normalised by the local
first-difference scale, giving a dimensionless discrete analogue of
`|y''|/|y'|` in `[0,1]`. Curvature and the first-order Lipschitz ratio are
reported **separately**, so the failure mode is identifiable rather than merely
flagged.

A negative result worth recording: normalising by the *input* second difference
— the obvious choice — fails, because meaning-preserving rephrasings barely move
the query embedding, leaving that denominator noise-dominated and the ratio
exploding for stable and unstable models alike.

**Claim skeleton.**
> 1. A method for detecting epistemic instability in a language model,
>    comprising: generating a graded ladder of phrasings in which each rung
>    applies one additional meaning-preserving transformation; obtaining a
>    response at each rung; embedding the responses; computing along each
>    consecutive triple a discrete second difference normalised by the local
>    first-difference magnitude; and reporting that curvature separately from a
>    first-order sensitivity ratio.
> 2. The method of claim 1, wherein high first-order sensitivity with low
>    curvature is classified as orderly sensitivity rather than instability.

**Evidence.** `TestSemanticCurvature` pins zero curvature for invariant answers
and strict ordering stable < orderly < unstable.

**Known limitation.** Curvature is a statement about embedding geometry, so its
discriminative power depends on the encoder more than the other metrics here.
The built-in hashing encoder sends unrelated n-grams to near-orthogonal
directions, which inflates the curvature of a progressively elaborated answer.
Ordering is preserved; separation sharpens materially with a real sentence
encoder.

---

## 5. Conformal Risk-Calibrated Evaluation (CRCE)

**Module:** `evalforge/novel/_conformal.py`

**Problem.** Evaluation gates releases but carries no guarantee. "Mean
groundedness 4.1" says nothing about how many bad outputs the judge missed, and
teams routing uncertain rows to humans have no principled way to set the
cut-off.

**Prior art.** Split conformal prediction; risk-controlling prediction sets
(Bates et al.); Learn-then-Test; selective prediction and abstention;
Hoeffding bounds. **The conformal machinery is established statistics.**

**Claimed novelty — the nonconformity score and the operating point.**
Applying distribution-free risk control to **LLM-judge evaluation** with a
composite nonconformity score assembled from quantities specific to this
setting and produced by the other algorithms here: judge-panel posterior
disagreement (§3), decision margin against the pass threshold, and rephrasing
instability (§4). The output is an operating point with a statement attached —
*with probability ≥ 1−δ, at most α of rows are failures accepted automatically*
— plus a defensible human-review budget, under only an exchangeability
assumption about calibration and production rows.

**Claim skeleton.**
> 1. A method for bounding undetected failures in automated evaluation,
>    comprising: computing for each calibration row a composite nonconformity
>    score combining a measure of disagreement among language-model judges, a
>    margin between a consensus score and a decision threshold, and a measure of
>    response instability under meaning-preserving input perturbation; defining a
>    risk as the fraction of rows auto-accepted at a cut-off that are labelled
>    failures; selecting the largest cut-off whose empirical risk plus a
>    finite-sample confidence term does not exceed a target; and routing rows
>    above the cut-off to human review.
> 2. The method of claim 1, further comprising reporting infeasibility and the
>    additional calibration size required when no cut-off satisfies the target.

**Evidence.** `TestConformalRisk` verifies the bound holds on fresh holdout
draws at multiple α, and that an undersized calibration set is correctly
reported infeasible with the required sample size.

---

## 6. Optimal-Transport Counterfactual Fairness Audit (OT-CFA)

**Module:** `evalforge/novel/_fairness.py` · résumé wrapper in `evalforge/contrib/_resume.py`

**Problem.** Fairness reporting stops at a disparity number, which says nothing
about how large an intervention would be needed, where in the score range the
unfairness sits, or which attribute is responsible. Disparities measured across
*different people* also confound bias with genuine population differences.

**Prior art.** Counterfactual fairness (Kusner et al.); demographic parity and
equalised odds; Wasserstein distances between group score distributions;
distribution alignment and fair representation learning; CDA-style attribute
substitution.

**Claimed novelty — the metric is the remediation.** Three elements bound
together: (a) the reported quantity is the **cost of the optimal repair** in
score units, not an abstract distance; (b) the transport map to the
one-dimensional barycenter `Q_bary(u) = (Q_A(u)+Q_B(u))/2` is returned as an
**executable, monotone, rank-preserving remediation** — the same object that
measures the disparity removes it, preserving within-cohort ranking while
equalising between cohorts; (c) the total is **decomposed across protected
attributes by ablation**, so a team learns which rewrite drives the disparity.

**Claim skeleton.**
> 1. A method for auditing a scoring system for attribute-dependent bias,
>    comprising: generating counterfactual pairs by rewriting a protected
>    attribute while holding other content fixed; scoring both cohorts;
>    computing a transport map carrying each cohort's score distribution onto
>    their barycenter; reporting the mean displacement as a repair cost in score
>    units; returning the transport map as a monotone remediation applicable to
>    future scores; and apportioning the disparity across attributes by
>    ablation.
> 2. The method of claim 1, further comprising a decision flip rate: the
>    fraction of items crossing a threshold on the attribute rewrite alone.

**Evidence.** `TestTransportFairness` shows exact recovery of an injected
disparity (W₁ = 0.20 against penalties summing to 0.20), attribution shares
proportional to each injected penalty (0.70/0.30), rank preservation under
repair, and both cohorts landing on a common mean after repair.

---

## 7. Coverage-Regularised Adaptive Attack Scheduling (CRAAS)

**Module:** `evalforge/novel/_bandit.py`

**Problem.** Exhaustive red-team scans grow as a cross product and real scans
run under budget. Uniform allocation wastes the budget; steering it toward what
succeeds collapses onto a single working exploit and reports a high
attack-success rate describing **one vulnerability found many times**. For a
safety scan that is the wrong objective.

**Prior art.** Thompson sampling and contextual bandits; automated red teaming
(PAIR, TAP, GCG); submodular coverage and facility-location maximization;
determinantal point processes for diversity.

**Claimed novelty — optimising coverage rather than success rate.** The bandit
reward is `success − kappa * redundancy`, where redundancy is the maximum
similarity between the current successful prompt and any previously successful
one — the marginal gain of a facility-location coverage objective. Bounded
rewards update the Beta posteriors via fractional pseudo-counts. Successful arms
additionally **chain**: the scheduler proposes the composition of a winning
strategy with the highest-posterior partner, discovering effective compositions
without enumerating them in advance.

**Claim skeleton.**
> 1. A method for allocating a bounded adversarial testing budget, comprising:
>    representing combinations of risk category and attack transformation as
>    bandit arms; selecting arms by posterior sampling; computing for each
>    successful attack a redundancy as its maximum similarity to previously
>    successful attack prompts; updating the selected arm's posterior with a
>    reward that penalises redundancy; and reporting the count of successes whose
>    redundancy falls below a novelty threshold as the coverage achieved.
> 2. The method of claim 1, further comprising adding, on a success, a composite
>    arm chaining that transformation with the highest-posterior-mean partner.

**Evidence.** `TestAdaptiveScheduler` pins the central claim: on an identical
budget against an identical target, the coverage-regularised objective finds
**more distinct weaknesses** than pure success-maximisation. The demo in
`examples/03_red_team.py` shows 19 distinct weaknesses versus 12 at an
*identical* ASR and hit count — the metric that matters diverging while the
headline number does not move.

---

## 8. Anytime-Valid Evaluation Regression Canary (AVERC)

**Module:** `evalforge/novel/_canary.py`

**Problem.** Evaluation metrics are inspected after every commit, nightly run
and deployment. Applying a fresh significance test at each check is
statistically invalid — peeking at a growing sample drives the false-alarm rate
toward one. Teams respond by alerting on noise or by widening thresholds until
real regressions pass unnoticed.

**Prior art.** E-values, test martingales and Ville's inequality; safe anytime-valid
inference (Grünwald, Ramdas, Shafer/Vovk); CUSUM and SPRT; Benjamini-Hochberg;
sequential A/B testing. **This mathematics is established.**

**Claimed novelty — the application, plus the baseline-error correction.**
A mixture e-process per metric, `E_t = prod exp(lambda*(mu0-x_i)/sigma0 - lambda^2/2)`
mixed over a grid of effect sizes (a convex combination of e-processes remains
an e-process), with the baseline estimated from evaluation history, an alarm
rule valid under the unlimited peeking CI performs by nature, and
BH-controlled fusion across a metric dashboard.

The correction is the part I would emphasise, because it was found empirically
rather than assumed. **Ville's inequality assumes a *known* null.** With `mu0`
estimated from finite history, the error is a fixed offset the martingale
accumulates evidence against, run after run, until it alarms. Measured: a canary
on a 30-run baseline false-alarmed at **0.103 against a nominal 0.05** — twice
its guarantee. The fix splits the error budget: half buys a one-sided confidence
bound on the baseline mean, so the e-process runs against a conservative null
`mu0 - z*sigma0/sqrt(n)` with predictive variance `sigma0*sqrt(1+1/n)`; half
buys an alarm boundary of `2/alpha`. A union bound restores the nominal
guarantee. Measured after the fix: **0.003–0.007**.

**Claim skeleton.**
> 1. A method for detecting regression in a continuously monitored evaluation
>    metric, comprising: estimating a baseline mean and scale from run history;
>    forming a conservative null by displacing the baseline mean by a confidence
>    term consuming a first portion of an error budget, and inflating the scale
>    to its predictive value; accumulating a mixture e-process over a grid of
>    effect sizes; raising an alarm when the mixture crosses a boundary set by
>    the second portion of the error budget; and controlling the false discovery
>    rate across several monitored metrics by a step-up procedure on
>    anytime-valid p-values derived from each e-process.

**Evidence.** `TestDriftCanary` measures the false-alarm rate under 150
consecutive peeks and asserts it respects the nominal bound, confirms detection
of genuine regressions in both directions, and confirms that multi-metric
monitoring flags only the regressed metric.

**Known limitation.** Sensitivity is baseline-limited, which is correct
behaviour rather than a defect: with a 30-run baseline the conservative
displacement is itself ≈0.36σ, so a 0.5σ regression is not detectable. With 100
runs the same regression is detected at observation 48.

---

## Reproducing the evidence

```bash
cd evalforge && PYTHONPATH=src python3 -m unittest discover -s tests
PYTHONPATH=src python3 examples/02_novel_algorithms.py
PYTHONPATH=src python3 examples/03_red_team.py
```

Every claim above is backed by an assertion in the test suite. Where a method
has a known limitation, it is stated in this document and in the module
docstring rather than left for a reader to discover.

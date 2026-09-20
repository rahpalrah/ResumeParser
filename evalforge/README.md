# EvalForge

A generative-AI evaluation SDK in two parts:

1. **A replica of the Azure AI Evaluation SDK surface** — the same evaluator
   classes, the same `evaluate()` contract, the same result-column names, the
   same simulators and red-team scanner. Written from the published API
   contract, with no third-party dependencies.
2. **`evalforge.novel`** — eight evaluation algorithms with no counterpart in
   the reference SDK, each targeting a specific failure of conventional
   evaluation. See [`docs/NOVEL_ALGORITHMS.md`](docs/NOVEL_ALGORITHMS.md).

Zero runtime dependencies. Every numerical routine, text encoder and HTTP
client is implemented on the standard library, so installs are trivial and
evaluation runs are reproducible bit-for-bit across machines.

## Install

```bash
pip install -e evalforge          # from the repository root
python -m unittest discover -s evalforge/tests
```

## Quick start

```python
from evalforge import evaluate, GroundednessEvaluator, F1ScoreEvaluator

result = evaluate(
    data="rows.jsonl",
    evaluators={
        "groundedness": GroundednessEvaluator(model_config),
        "f1": F1ScoreEvaluator(),
    },
    evaluator_config={
        "groundedness": {"column_mapping": {"response": "${data.answer}"}},
    },
    output_path="./results.json",
)

print(result["metrics"]["groundedness.groundedness"])
```

Every AI-assisted evaluator takes a model configuration. **Omit it and the
evaluator falls back to a deterministic in-process scorer**, so the entire
surface — including the LLM-judged metrics — runs and tests without credentials
or network access. That fallback is an approximation with documented blind
spots, not a claim of parity with a hosted judge; see
[`docs/COMPATIBILITY.md`](docs/COMPATIBILITY.md).

## What is included

**Quality** — `GroundednessEvaluator`, `GroundednessProEvaluator`,
`RelevanceEvaluator`, `CoherenceEvaluator`, `FluencyEvaluator`,
`SimilarityEvaluator`, `RetrievalEvaluator`, `QAEvaluator`

**Agents** — `IntentResolutionEvaluator`, `TaskAdherenceEvaluator`,
`ToolCallAccuracyEvaluator`, `ResponseCompletenessEvaluator`

**NLP overlap** — `F1ScoreEvaluator`, `BleuScoreEvaluator`,
`GleuScoreEvaluator`, `RougeScoreEvaluator`, `MeteorScoreEvaluator`
(exact implementations of the published definitions; verified against known
reference values in the test suite)

**Safety** — `ViolenceEvaluator`, `SexualEvaluator`, `SelfHarmEvaluator`,
`HateUnfairnessEvaluator`, `ContentSafetyEvaluator`,
`ProtectedMaterialEvaluator`, `IndirectAttackEvaluator`,
`CodeVulnerabilityEvaluator`, `UngroundedAttributesEvaluator`

**Simulation** — `Simulator`, `AdversarialSimulator`, `DirectAttackSimulator`,
`IndirectAttackSimulator`

**Red teaming** — `RedTeam` with 16 attack strategies (base64, ROT13, morse,
binary, Caesar, leetspeak, homoglyph, flip, URL, character-space, tense,
suffix, jailbreak, crescendo) and arbitrary compositions

**Résumé domain** — `evalforge.contrib.ResumeExtractionEvaluator` and
`ResumeFairnessProbe`, wired to this repository's parser output schema

## The novel algorithms

| Algorithm | Answers the question |
|---|---|
| `SpectralSemanticDrift` | Did this conversation slowly go off-goal, and at which turn? |
| `CausalAblationGroundedness` | *Which* claim is unsupported, and by which passage? |
| `JudgeItemResponseCalibrator` | Which of my LLM judges can I trust, and how much? |
| `SemanticCurvatureProbe` | Is the model answering, or guessing? |
| `ConformalRiskController` | How many failures did automated review miss — with a guarantee? |
| `TransportFairnessAuditor` | How large is the bias, and what removes it? |
| `AdaptiveAttackScheduler` | Where should a limited red-team budget go? |
| `SequentialDriftCanary` | Has the metric regressed, given that I check after every run? |

Each is documented — problem, method, prior art, and precisely what is claimed
to be new — in [`docs/NOVEL_ALGORITHMS.md`](docs/NOVEL_ALGORITHMS.md).

## Examples

```bash
PYTHONPATH=evalforge/src python3 evalforge/examples/01_basic_evaluation.py
PYTHONPATH=evalforge/src python3 evalforge/examples/02_novel_algorithms.py
PYTHONPATH=evalforge/src python3 evalforge/examples/03_red_team.py
```

## Tests

```bash
cd evalforge && PYTHONPATH=src python3 -m unittest discover -s tests
```

148 tests, no external dependencies. They include empirical checks of the
statistical guarantees: the conformal risk bound is verified against fresh
holdout draws, and the canary's false-alarm rate is measured under repeated
peeking.

## License

Apache-2.0. This is an independent implementation written from the published
API contract; it is not affiliated with or endorsed by Microsoft or Azure.

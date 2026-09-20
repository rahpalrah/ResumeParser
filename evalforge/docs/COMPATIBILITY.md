# Compatibility with the Azure AI Evaluation SDK

EvalForge mirrors the public surface of `azure-ai-evaluation`. Code written
against that SDK generally runs here by changing the import:

```python
# from azure.ai.evaluation import evaluate, GroundednessEvaluator
from evalforge import evaluate, GroundednessEvaluator
```

This document records what matches, what differs, and where the differences
will bite.

## Name mapping

| Reference SDK | EvalForge |
|---|---|
| `azure.ai.evaluation` | `evalforge` |
| `azure.ai.evaluation.simulator` | `evalforge.simulator` |
| `azure.ai.evaluation.red_team` | `evalforge.red_team` |
| — | `evalforge.novel` *(new)* |
| — | `evalforge.contrib` *(new)* |

Evaluator class names, constructor argument names and result-column names are
unchanged.

## The `evaluate()` contract

Matching behaviour:

- `data` accepts a `.jsonl` path; EvalForge also accepts `.json`, `.csv` and an
  in-memory list of dicts.
- `evaluators` maps a run-local name to a callable; that name prefixes every
  column the evaluator emits.
- `evaluator_config[name]["column_mapping"]` resolves `${data.x}`,
  `${target.x}` and `${run.outputs.x}`. A `"default"` entry applies to every
  evaluator.
- The return value is `{"metrics", "rows", "studio_url"}`. Row columns are
  `inputs.<column>` and `outputs.<evaluator>.<metric>`.
- `target` is invoked once per row; its returned keys become `${target.*}`.
- `output_path` writes rows for a `.jsonl` path, or the whole result object
  otherwise.

Additions: `max_workers` controls row parallelism, and the result carries a
`run_info` block (id, name, row count, evaluator list, duration).

## Result columns

AI-assisted quality evaluators emit, for metric `m`:
`m`, `gpt_m`, `m_reason`, `m_result`, `m_threshold`.

Safety evaluators emit `m` (a `Very low`/`Low`/`Medium`/`High` band),
`m_score` (0–7 severity), `m_reason`, `m_result`, `m_threshold`. For these,
*lower is better* and the pass condition is `score <= threshold`.

Conversation inputs add `evaluation_per_turn`.

## Substantive differences

**1. The judge backend is pluggable, and optional.**
The reference SDK requires a model configuration for AI-assisted evaluators.
EvalForge accepts one — `AzureOpenAIModelConfiguration` or
`OpenAIModelConfiguration`, spoken over `urllib` — and falls back to a
deterministic in-process scorer when none is supplied, or when one is supplied
without an API key.

That fallback is what makes the whole surface runnable offline and testable in
CI. It is *not* equivalent to an LLM judge. It is reference-free and carries no
world knowledge, so it can verify that a response is supported by supplied
context but cannot verify a bare factual assertion:
`RelevanceEvaluator()(query="Capital of France?", response="Paris.")` is scored
on responsiveness and form, not on the truth of "Paris". Use a real model
configuration for judgements requiring knowledge.

**2. Safety evaluation is local.**
The reference SDK calls a hosted Azure safety service. EvalForge scores harm
with a transparent, auditable lexicon (`evalforge._common._heuristics`), damped
by mitigating context so a refusal that names a harm is not scored as the harm.
The `azure_ai_project` and `credential` arguments are accepted for signature
compatibility. For production safety decisions, back these evaluators with a
real classifier.

**3. `studio_url` is constructed, not registered.**
Supplying `azure_ai_project` produces a correctly-shaped studio URL, but no
results are uploaded anywhere. Nothing leaves the process unless you configure
a hosted judge.

**4. Adversarial content is generated locally.**
Simulators and the red-team scanner use built-in templates and transformations
rather than fetching curated attack sets from a service, so scans are
reproducible under a seed and run offline. Seed objectives name the harm rather
than supplying operational detail, which is sufficient to test refusal.

**5. Prompts are data.**
Judge prompts live in `evalforge._evaluators._common._prompty` and can be
overridden with a `.prompty` file (YAML front matter plus role-delimited body)
through the `Prompty` loader.

## Not implemented

Azure-specific plumbing with no offline meaning: run upload and tracking,
`AIAgentConverter` and thread-based agent ingestion, and hosted-service
evaluator variants beyond the interfaces listed above.

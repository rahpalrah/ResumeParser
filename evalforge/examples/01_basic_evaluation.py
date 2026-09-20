"""Batch-evaluate a dataset with several evaluators.

Run: python examples/01_basic_evaluation.py
"""

from evalforge import (
    CoherenceEvaluator,
    F1ScoreEvaluator,
    GroundednessEvaluator,
    RelevanceEvaluator,
    evaluate,
)

ROWS = [
    {
        "query": "How tall is the Eiffel Tower?",
        "context": "The Eiffel Tower is in Paris and stands 330 metres tall.",
        "answer": "The Eiffel Tower is 330 metres tall.",
        "truth": "It is 330 metres tall.",
    },
    {
        "query": "When was it completed?",
        "context": "The Eiffel Tower was completed in 1889 for the World's Fair.",
        "answer": "It was completed in 1723 by Leonardo da Vinci.",
        "truth": "It was completed in 1889.",
    },
]


def main() -> None:
    # Passing no model configuration selects the deterministic offline judge,
    # so this example runs with no credentials. Pass a model configuration to
    # any evaluator to use a hosted judge instead.
    result = evaluate(
        data=ROWS,
        evaluators={
            "groundedness": GroundednessEvaluator(),
            "relevance": RelevanceEvaluator(),
            "coherence": CoherenceEvaluator(),
            "f1": F1ScoreEvaluator(),
        },
        evaluator_config={
            "default": {"column_mapping": {"response": "${data.answer}"}},
            "f1": {"column_mapping": {"response": "${data.answer}", "ground_truth": "${data.truth}"}},
        },
        evaluation_name="eiffel-tower-qa",
    )

    print("Run-level metrics")
    for name, value in sorted(result["metrics"].items()):
        print(f"  {name:45s} {value:.3f}")

    print("\nPer row")
    for index, row in enumerate(result["rows"]):
        print(
            f"  row {index}: groundedness={row['outputs.groundedness.groundedness']:.1f} "
            f"({row['outputs.groundedness.groundedness_result']})"
        )


if __name__ == "__main__":
    main()

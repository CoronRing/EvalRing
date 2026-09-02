"""
Example: LLM-as-a-Judge evaluation for suicide risk classification.

Shows the simplified API — just pass rubrics to LLMJudgeEvaluator.from_rubric().

Usage:
    python evaluate_rsd15k_llm_judge.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

current_file = Path(__file__).resolve()
# Sibling example modules (llm_agent) are not an installed package.
sys.path.append(str(current_file.parent))

from EvalRing.dataset import CSVDataset
from EvalRing.evaluator.llm_judge import LLMJudgeEvaluator, Rubric, RubricLevel

try:
    from llm_agent import OpenAISuicideDetectionAgent
except ImportError:
    OpenAISuicideDetectionAgent = None


# ──────────────────────────────────────────────────────────
# 1. Define Rubrics
# ──────────────────────────────────────────────────────────

CLASSIFICATION_RUBRIC = Rubric(
    name="classification_correctness",
    description="How well does the agent's classification match the ground truth label?",
    levels=[
        RubricLevel(0, "Wrong", "The prediction is completely wrong — different category with no reasonable justification."),
        RubricLevel(3, "Adjacent", "The prediction is in a neighboring/related category (e.g., Ideation vs Indicator). Understandable confusion but still incorrect."),
        RubricLevel(5, "Partially Correct", "The prediction captures some aspects of the correct category but misses key distinguishing features."),
        RubricLevel(8, "Near Perfect", "The prediction is very close — correct overall category, possibly a borderline case where both labels could be argued."),
        RubricLevel(10, "Perfect", "The prediction exactly matches the ground truth label."),
    ]
)

REASONING_RUBRIC = Rubric(
    name="reasoning_quality",
    description="How well does the classification reflect understanding of the text's risk signals?",
    levels=[
        RubricLevel(0, "No Understanding", "The classification shows no understanding of the risk signals in the text."),
        RubricLevel(3, "Surface Level", "The classification picks up on some keywords but misses the overall context and severity."),
        RubricLevel(5, "Moderate", "The classification reflects reasonable understanding but misinterprets some key signals."),
        RubricLevel(8, "Strong", "The classification demonstrates strong clinical understanding of the risk indicators."),
        RubricLevel(10, "Expert", "The classification perfectly captures the risk level and type, handling nuance and ambiguity expertly."),
    ]
)


# ──────────────────────────────────────────────────────────
# 2. Main evaluation — simplified with from_rubric()
# ──────────────────────────────────────────────────────────

def run_llm_judge_eval():
    NUM_SAMPLES = 10

    print(f"=== LLM-as-a-Judge Evaluation (RSD_15K, {NUM_SAMPLES} samples) ===\n")

    # -- Dataset --
    base_dir = current_file.parent
    data_path = base_dir / "data" / "rsd_15k.csv"
    
    dataset = CSVDataset(
        name="rsd_15k",
        description="Reddit Suicide Detection 15K",
    )
    dataset.load_data(
        str(data_path),
        text_field="text",
        label_field="sentiment",
        id_field="ID",
    )
    dataset._samples = dataset._samples[:NUM_SAMPLES]
    dataset.assert_unique_ids(expected_count=len(dataset._samples), context=f"llm_judge_eval.head({NUM_SAMPLES})")
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # -- Agent --
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    
    if not OpenAISuicideDetectionAgent:
        print("ERROR: Could not import OpenAISuicideDetectionAgent.")
        sys.exit(1)
    
    agent = OpenAISuicideDetectionAgent(
        name="gpt-5-mini-suicide-detector",
        model_name="gpt-5.2",
        api_key=api_key,
    )
    
    # -- Evaluator (one-liner with from_rubric) --
    evaluator = LLMJudgeEvaluator.from_rubric(
        rubric={
            "correctness": CLASSIFICATION_RUBRIC,
            "reasoning": REASONING_RUBRIC,
        },
        criteria={
            "correctness": (
                "Evaluate whether the agent's suicide risk classification is correct.\n"
                "The four valid categories are: Indicator, Ideation, Behavior, Attempt.\n"
                "Compare the agent's output to the ground truth label."
            ),
            "reasoning": (
                "Evaluate whether the agent's classification demonstrates genuine "
                "understanding of the suicide risk signals in the input text."
            ),
        },
        weights={"correctness": 2.0, "reasoning": 1.0},
        judge_model="gpt-5.2",
        api_key=api_key,
        max_workers=5,
        name="llm-judge-evaluator",
    )
    
    # -- Run --
    print("\nRunning evaluation...\n")
    result = evaluator.evaluate(
        agent=agent,
        dataset=dataset,
        task_name="suicide_risk_classification_judge",
    )
    
    # ──────────────────────────────────────────────
    # 3. Print results
    # ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    
    metrics_dict = result.metrics.to_dict()["metrics"]
    for key, value in metrics_dict.items():
        print(f"  {key}: {value}")
    
    print(f"\n--- Per-Sample Verdicts ({len(result.metrics.per_sample_metrics)} samples) ---\n")
    for record in result.metrics.per_sample_metrics:
        sid = record.get("sample_id", "?")
        output = record.get("agent_output", "")
        gt = record.get("ground_truth", "")
        composite = record.get("composite_score", 0)
        
        print(f"Sample {sid}: Pred={output}, GT={gt}, Composite={composite}")
        for m in evaluator.metrics:
            reason = record.get(f"{m.name}_reason", "")
            score = record.get(f"{m.name}_score", "")
            normalized = record.get(f"{m.name}_normalized", "")
            print(f"  [{m.name}] score={score} (norm={normalized}) — {reason[:120]}")
        print()
    
    # ──────────────────────────────────────────────
    # 4. Save reports (one-liner)
    # ──────────────────────────────────────────────
    run_dir = evaluator.save_reports(result, base_dir / "_EvalRing")
    print(f"\nReports saved to: {run_dir}")
    print("Done.")


if __name__ == "__main__":
    run_llm_judge_eval()

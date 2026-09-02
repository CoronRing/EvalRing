"""
Validation: Verify LLM-as-a-Judge matches ground truth exactly.

Approach:
1. Run agent on N samples → get predictions (generate once)
2. Check A: Deterministic exact-match against ground truth
3. Check B: LLM judge with strict binary rubric ("10/10 if exact match, 0/10 if not")
4. Compare both checks — they MUST agree on every sample

This validates the judge's reliability: if it can't even get
exact-match correctness right, it's not trustworthy for harder tasks.
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

current_file = Path(__file__).resolve()
# Sibling example modules are not an installed package.
sys.path.append(str(current_file.parent))

from EvalRing.dataset import CSVDataset
from EvalRing.evaluator.llm_judge import (
    LLMJudgeEvaluator,
    OpenAIJudge,
    JudgeMetric,
    ScoringCriteria,
)

try:
    from llm_agent import OpenAISuicideDetectionAgent
except ImportError:
    OpenAISuicideDetectionAgent = None


NUM_SAMPLES = 10


# ──────────────────────────────────────────────────────────
# Strict binary rubric as a plain string
# ──────────────────────────────────────────────────────────

EXACT_MATCH_RUBRIC = """Score MUST be exactly 10 or 0. No partial credit.

Score 10: The [output] is EXACTLY the same category as [ground_truth]. 
          Case-insensitive comparison: "Ideation" == "ideation" == "IDEATION".
          
Score 0:  The [output] does NOT match [ground_truth]. Any mismatch = 0.

There is no middle ground. Either the labels match or they don't."""


def run_validation():
    print(f"=== Judge Validation: Ground Truth Consistency ({NUM_SAMPLES} samples) ===\n")

    # ── Load dataset ──
    base_dir = current_file.parent
    data_path = base_dir / "data" / "rsd_15k.csv"

    dataset = CSVDataset(name="rsd_15k", description="RSD 15K")
    dataset.load_data(
        str(data_path),
        text_field="text",
        label_field="sentiment",
        id_field="ID",
    )
    dataset._samples = dataset._samples[:NUM_SAMPLES]
    dataset.assert_unique_ids(expected_count=len(dataset._samples), context=f"validate_judge.head({NUM_SAMPLES})")
    print(f"Dataset: {len(dataset)} samples loaded\n")

    # ── Agent ──
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key or not OpenAISuicideDetectionAgent:
        print("ERROR: OPENAI_API_KEY not set or agent not importable.")
        sys.exit(1)

    agent = OpenAISuicideDetectionAgent(
        name="gpt-5-mini-suicide-detector",
        model_name="gpt-5.2",
        api_key=api_key,
    )
    agent.initialize()

    # ══════════════════════════════════════════════════════
    # PHASE 1: Generate predictions once
    # ══════════════════════════════════════════════════════
    print("Phase 1: Generating agent predictions...\n")

    predictions = []  # list of (sample, prediction_str)
    for i, sample in enumerate(dataset):
        response = agent.predict(sample.input_text)
        pred = str(response.output).strip()
        gt = str(sample.target_output).strip()
        predictions.append({
            "index": i,
            "sample_id": sample.id,
            "input_text": sample.input_text[:100],
            "prediction": pred,
            "ground_truth": gt,
        })
        print(f"  [{i}] Pred={pred:10s}  GT={gt:10s}  {'✓' if pred.lower() == gt.lower() else '✗'}")

    # ══════════════════════════════════════════════════════
    # CHECK A: Deterministic exact-match
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CHECK A: Deterministic Exact-Match")
    print("=" * 60)

    ground_truth_results = []
    for p in predictions:
        match = p["prediction"].lower() == p["ground_truth"].lower()
        ground_truth_results.append(match)
        status = "PASS (10)" if match else "FAIL (0)"
        print(f"  [{p['index']}] {status}  Pred={p['prediction']}  GT={p['ground_truth']}")

    gt_accuracy = sum(ground_truth_results) / len(ground_truth_results)
    print(f"\n  Accuracy: {gt_accuracy:.0%}  ({sum(ground_truth_results)}/{len(ground_truth_results)})")

    # ══════════════════════════════════════════════════════
    # CHECK B: LLM-as-a-Judge with strict binary rubric
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("CHECK B: LLM-as-a-Judge (strict binary rubric)")
    print("=" * 60)

    judge = OpenAIJudge(
        model_name="gpt-5.2",
        api_key=api_key,
        temperature=0.0,
        max_completion_tokens=256,
    )

    exact_match_criteria = ScoringCriteria(
        name="exact_match",
        criteria="Check if the agent's output label exactly matches the ground truth label.",
        evaluation_steps=[
            "Read the [output] field (the agent's predicted label).",
            "Read the [ground_truth] field (the correct label).",
            "Compare them case-insensitively.",
            "If they match exactly, score 10. If they differ at all, score 0.",
        ],
        rubric=EXACT_MATCH_RUBRIC,  # str rubric
        weight=1.0,
    )

    metric = JudgeMetric(
        criteria=exact_match_criteria,
        judge=judge,
        parameters=["input", "output", "ground_truth"],
        threshold=1.0,  # must be perfect
    )

    judge_results = []
    for p in predictions:
        verdict = metric.score_simple(
            input_text=p["input_text"],
            output=p["prediction"],
            ground_truth=p["ground_truth"],
        )
        is_pass = verdict.score == 10.0
        judge_results.append(is_pass)
        status = f"PASS ({verdict.score:.0f})" if is_pass else f"FAIL ({verdict.score:.0f})"
        print(f"  [{p['index']}] {status}  Pred={p['prediction']}  GT={p['ground_truth']}")
        print(f"         Reason: {verdict.reason[:120]}")

    judge_accuracy = sum(judge_results) / len(judge_results)
    print(f"\n  Judge Accuracy: {judge_accuracy:.0%}  ({sum(judge_results)}/{len(judge_results)})")

    # ══════════════════════════════════════════════════════
    # COMPARISON: Do both checks agree?
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("COMPARISON: Ground Truth vs LLM Judge")
    print("=" * 60)

    mismatches = []
    for i, (gt_pass, judge_pass) in enumerate(zip(ground_truth_results, judge_results)):
        agree = gt_pass == judge_pass
        p = predictions[i]
        symbol = "✓ AGREE" if agree else "✗ MISMATCH"
        gt_label = "PASS" if gt_pass else "FAIL"
        jd_label = "PASS" if judge_pass else "FAIL"
        print(f"  [{i}] {symbol}  GT_check={gt_label}  Judge={jd_label}  Pred={p['prediction']}  GT={p['ground_truth']}")
        if not agree:
            mismatches.append(i)

    print(f"\n{'=' * 60}")
    if not mismatches:
        print("  ✓ PERFECT MATCH: Ground truth and LLM judge agree on all samples!")
        print(f"  Both report: {sum(ground_truth_results)} correct, "
              f"{len(ground_truth_results) - sum(ground_truth_results)} incorrect")
    else:
        print(f"  ✗ MISMATCH on {len(mismatches)} sample(s): {mismatches}")
        print("  The judge disagrees with deterministic ground truth — investigate!")
    print("=" * 60)

    # ── Save report ──
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / "_EvalRing" / f"judge_validation_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "timestamp": datetime.now().isoformat(),
        "num_samples": NUM_SAMPLES,
        "ground_truth_accuracy": gt_accuracy,
        "judge_accuracy": judge_accuracy,
        "perfect_agreement": len(mismatches) == 0,
        "mismatches": mismatches,
        "predictions": predictions,
        "ground_truth_checks": ground_truth_results,
        "judge_checks": judge_results,
    }
    with open(run_dir / "validation_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    print(f"\nReport saved to: {run_dir}")


if __name__ == "__main__":
    run_validation()

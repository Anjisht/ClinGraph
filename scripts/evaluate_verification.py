"""Benchmark evaluation for claim verification precision, recall, and abstention."""

import json
from collections import defaultdict
from pathlib import Path

from app.core.logging import logger
from app.verification.claim_verifier import DeterministicClaimVerifier


def evaluate_verification(
    dataset_path: Path = Path("data/samples/verification_dataset.json"),
    output_path: Path = Path("reports/verification_evaluation.json"),
) -> dict:
    """Evaluate verifier performance across ground-truth labeled clinical assertions."""
    if not dataset_path.exists():
        raise FileNotFoundError(f"Verification dataset not found: {dataset_path}")

    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    verifier = DeterministicClaimVerifier()

    classes = ["supported", "partially_supported", "contradicted", "unverified"]
    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)

    correct_count = 0
    total = len(dataset)
    predictions = []

    logger.info(f"Evaluating {total} labeled verification test samples...")

    for item in dataset:
        claim = item["claim"]
        passage = item["passage"]
        expected = item["expected_status"]

        result = verifier.verify(
            claim_index=0,
            claim_text=claim,
            passage_text=passage,
            chunk_id=item["id"],
        )
        predicted = result.status

        is_correct = predicted == expected
        if is_correct:
            correct_count += 1
            tp[expected] += 1
        else:
            fp[predicted] += 1
            fn[expected] += 1

        predictions.append(
            {
                "id": item["id"],
                "category": item.get("category"),
                "expected": expected,
                "predicted": predicted,
                "confidence": result.confidence,
                "is_correct": is_correct,
                "rationale": result.rationale,
            }
        )

    accuracy = correct_count / total if total > 0 else 0.0

    class_metrics = {}
    macro_f1_sum = 0.0

    for c in classes:
        c_tp = tp[c]
        c_fp = fp[c]
        c_fn = fn[c]

        prec = c_tp / (c_tp + c_fp) if (c_tp + c_fp) > 0 else 0.0
        rec = c_tp / (c_tp + c_fn) if (c_tp + c_fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        class_metrics[c] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": c_tp + c_fn,
        }
        macro_f1_sum += f1

    macro_f1 = macro_f1_sum / len(classes) if classes else 0.0

    report = {
        "total_samples": total,
        "overall_accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": class_metrics,
        "sample_predictions": predictions,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Print summary
    print("\n================== CLAIM VERIFICATION EVALUATION ==================")
    print(f"Total samples: {total} | Overall Accuracy: {accuracy:.2%}")
    print(f"{'Class':<22} | {'Precision':<12} | {'Recall':<12} | {'F1-Score':<10}")
    print("-" * 62)
    for c in classes:
        m = class_metrics[c]
        p_str = f"{m['precision']:.2%}"
        r_str = f"{m['recall']:.2%}"
        f_str = f"{m['f1']:.4f}"
        print(f"{c:<22} | {p_str:<12} | {r_str:<12} | {f_str:<10}")
    print("-" * 62)
    print(f"{'Macro Average F1':<22} | {'':<12} | {'':<12} | {macro_f1:<10.4f}")
    print("===================================================================\n")

    return report


if __name__ == "__main__":
    evaluate_verification()

"""Train baseline and classification models on the UCI Student Performance dataset.

This script executes the complete ML workflow:
1. Loads Mathematics or Portuguese dataset from UCI.
2. Constructs the 3-category performance target from G3.
3. Excludes G1, G2, and G3 to guarantee zero target leakage.
4. Evaluates a Majority-class Dummy baseline.
5. Trains and cross-validates Decision Tree and Logistic Regression classifiers.
6. Evaluates on held-out test data and saves pipeline artifact.
"""

import argparse
from pathlib import Path
from src.data import load_student_data
from src.model import train_and_save, DEFAULT_ARTIFACT_PATH


def main():
    parser = argparse.ArgumentParser(description="EduPredict Training & Evaluation")
    parser.add_argument(
        "--subject",
        choices=["mat", "por"],
        default="mat",
        help="Dataset subject: 'mat' (Mathematics, 395 rows) or 'por' (Portuguese, 649 rows)."
    )
    parser.add_argument(
        "--model",
        choices=["decision_tree", "logistic_regression"],
        default="decision_tree",
        help="Primary classifier architecture to train and save."
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(DEFAULT_ARTIFACT_PATH),
        help="Path to save the fitted pipeline artifact."
    )
    parser.add_argument(
        "--include-grades",
        action="store_true",
        help="Include G1 and G2 midterm grades for high-precision tracking (yields ~88-90% accuracy)."
    )
    args = parser.parse_args()

    mode_label = "MID-TERM TRACKING (G1 & G2 Included -> ~89% Acc)" if args.include_grades else "PRE-TERM SCREENING (Zero Leakage -> ~48% Acc)"
    print("=" * 65)
    print(f"🎓 EduPredict — Training Pipeline (Subject: {args.subject.upper()})")
    print(f"   Mode: {mode_label}")
    print("=" * 65)

    print(f"1. Loading UCI dataset for subject '{args.subject}'...")
    df = load_student_data(args.subject)
    print(f"   Loaded {df.shape[0]} students and {df.shape[1]} raw attributes.")

    print(f"2. Training models with selected feature configuration...")
    results = train_and_save(
        df=df,
        output_path=args.output,
        model_type=args.model,
        test_size=0.20,
        random_state=42,
        include_prior_grades=args.include_grades
    )

    primary_name = "Decision Tree" if args.model == "decision_tree" else "Logistic Regression"
    alt_name = "Logistic Regression" if args.model == "decision_tree" else "Decision Tree"

    print("\n" + "=" * 65)
    print("📊 MODEL EVALUATION SUMMARY (Held-out 20% Test Split)")
    print("=" * 65)
    print(f"{'Model':<25} {'Test Accuracy':<15} {'5-Fold CV Mean ± Std':<20}")
    print("-" * 65)
    print(f"{'Majority Baseline':<25} {results['baseline_accuracy']:.3f}{'':<10} {'N/A'}")
    print(f"{primary_name + ' (Primary)':<25} {results['accuracy']:.3f}{'':<10} {results['cv_scores'].mean():.3f} ± {results['cv_scores'].std():.3f}")
    print(f"{alt_name + ' (Stretch Comparison)':<25} {results['alt_accuracy']:.3f}{'':<10} {results['alt_cv_scores'].mean():.3f} ± {results['alt_cv_scores'].std():.3f}")
    print("-" * 65)

    print(f"\n📋 Classification Report ({primary_name}):")
    print(results["report"])

    print("🧩 Confusion Matrix (Rows: Actual, Columns: Predicted):")
    print(f"   Classes: {results['labels']}")
    for label, row in zip(results["labels"], results["confusion"]):
        print(f"   {label:<15}: {row}")

    print("\n🔍 Top 5 Informative Features:")
    for feat, imp in results["feature_importances"].head(5).items():
        print(f"   - {feat:<20}: {imp:.4f}")

    if not results["errors_df"].empty:
        print(f"\n⚠️  Diagnostic Error Check ({len(results['errors_df'])} misclassifications in test set):")
        two_errors = results["errors_df"].head(2)
        for idx, row in two_errors.iterrows():
            print(f"   Student #{idx}: Actual='{row['actual']}', Predicted='{row['predicted']}' (Study: {row['studytime']}, Failures: {row['failures']}, Absences: {row['absences']})")

    print("\n💾 Artifact saved to:", results["output_path"])
    print("=" * 65)


if __name__ == "__main__":
    main()

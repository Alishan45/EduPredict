from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ARTIFACT_PATH = PROJECT_ROOT / "artifacts" / "edupredict_pipeline.joblib"

TARGET = "performance_category"
TARGET_CLASSES = ["Needs Support", "Satisfactory", "Strong"]
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create discrete performance categories from final grade G3.

    - Needs Support: 0–9 (failing / below standard)
    - Satisfactory: 10–13 (passing / adequate)
    - Strong: 14–20 (good to excellent)
    """
    result = df.copy()
    result[TARGET] = pd.cut(
        result["G3"],
        bins=[-1, 9, 13, 20],
        labels=TARGET_CLASSES,
        include_lowest=True
    )
    return result


def build_pipeline(X: pd.DataFrame, model_type: str = "decision_tree", **model_kwargs) -> Pipeline:
    """Construct a scikit-learn Pipeline with leak-free preprocessing and a classifier."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", categorical_pipeline, categorical_features)
    ])

    if model_type == "decision_tree":
        default_params = {"max_depth": 3, "min_samples_leaf": 5, "random_state": 42}
        default_params.update(model_kwargs)
        classifier = DecisionTreeClassifier(**default_params)
    elif model_type == "logistic_regression":
        default_params = {"C": 0.1, "max_iter": 1000, "random_state": 42}
        default_params.update(model_kwargs)
        classifier = LogisticRegression(**default_params)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}. Choose 'decision_tree' or 'logistic_regression'.")

    return Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier)
    ])


def get_feature_names(fitted_pipeline: Pipeline, X: pd.DataFrame) -> list:
    """Extract processed feature names after ColumnTransformer transformation."""
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    preprocessor = fitted_pipeline.named_steps["preprocessing"]
    try:
        cat_transformer = preprocessor.named_transformers_["categorical"]
        ohe = cat_transformer.named_steps["onehot"]
        encoded_cats = ohe.get_feature_names_out(categorical_features).tolist()
    except Exception:
        encoded_cats = categorical_features

    return numeric_features + encoded_cats


def get_feature_importances(fitted_pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """Compute feature importances (Tree) or mean absolute coefficients (Logistic Regression)."""
    feature_names = get_feature_names(fitted_pipeline, X)
    classifier = fitted_pipeline.named_steps["classifier"]

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        # For multiclass logistic regression, take average absolute magnitude across classes
        importances = np.mean(np.abs(classifier.coef_), axis=0)
    else:
        importances = np.zeros(len(feature_names))

    return pd.Series(importances, index=feature_names).sort_values(ascending=False)


def train_and_save(
    df: pd.DataFrame,
    output_path: Path | str = None,
    model_type: str = "decision_tree",
    test_size: float = 0.20,
    random_state: int = 42,
    include_prior_grades: bool = False
) -> dict:
    """Train baseline and classifier on held-out split, run cross-validation, and save artifact.
    
    If include_prior_grades=True, G1 and G2 (1st and 2nd trimester grades) are included,
    yielding high-precision mid-term performance (~88-90% accuracy).
    If False, G1 and G2 are excluded to maintain strict zero-leakage pre-term screening (~48%).
    """
    data = add_target(df).dropna(subset=[TARGET])
    drop_cols = ["G3", TARGET] if include_prior_grades else LEAKAGE_COLUMNS + [TARGET]
    X = data.drop(columns=drop_cols)
    y = data[TARGET].astype(str)

    # Stratified split to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Baseline model: Majority class classifier
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    # 2. Primary classifier pipeline
    model = build_pipeline(X_train, model_type=model_type)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    # 3. 5-fold Cross-validation on training split (Stretch challenge)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv)

    # 4. Comparison classifier (Stretch challenge)
    alt_type = "logistic_regression" if model_type == "decision_tree" else "decision_tree"
    alt_model = build_pipeline(X_train, model_type=alt_type)
    alt_model.fit(X_train, y_train)
    alt_predictions = alt_model.predict(X_test)
    alt_accuracy = accuracy_score(y_test, alt_predictions)
    alt_cv_scores = cross_val_score(alt_model, X_train, y_train, cv=cv)

    # Feature importances
    feature_importances = get_feature_importances(model, X_train)

    # Error analysis: collect misclassified examples (Thursday deliverable)
    test_analysis = X_test.copy()
    test_analysis["actual"] = y_test
    test_analysis["predicted"] = predictions
    errors_df = test_analysis[test_analysis["actual"] != test_analysis["predicted"]]

    labels = TARGET_CLASSES
    report = classification_report(y_test, predictions, labels=labels, zero_division=0)
    conf_matrix = confusion_matrix(y_test, predictions, labels=labels)

    # Save artifact
    output = Path(output_path) if output_path else DEFAULT_ARTIFACT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "model_type": model_type,
        "alt_model": alt_model,
        "alt_type": alt_type,
        "feature_columns": X.columns.tolist(),
        "classes": labels,
        "target_definition": {
            "Needs Support": "Final grade G3 from 0 to 9",
            "Satisfactory": "Final grade G3 from 10 to 13",
            "Strong": "Final grade G3 from 14 to 20"
        },
        "metrics": {
            "baseline_accuracy": baseline_accuracy,
            "accuracy": accuracy,
            "cv_mean": float(cv_scores.mean()),
            "cv_std": float(cv_scores.std()),
            "alt_accuracy": alt_accuracy,
            "alt_cv_mean": float(alt_cv_scores.mean()),
            "alt_cv_std": float(alt_cv_scores.std())
        }
    }, output)

    return {
        "model": model,
        "model_type": model_type,
        "alt_model": alt_model,
        "alt_type": alt_type,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "predictions": predictions,
        "baseline_predictions": baseline_predictions,
        "accuracy": accuracy,
        "baseline_accuracy": baseline_accuracy,
        "cv_scores": cv_scores,
        "alt_accuracy": alt_accuracy,
        "alt_cv_scores": alt_cv_scores,
        "report": report,
        "confusion": conf_matrix,
        "labels": labels,
        "feature_importances": feature_importances,
        "errors_df": errors_df,
        "output_path": str(output)
    }

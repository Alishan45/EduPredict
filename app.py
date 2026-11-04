"""EduPredict — Student Performance Prediction & Analysis Streamlit Application.

Week 02 AI Engineering Internship | Stack & Scale
Interactive prototype for educational data exploration, target classification,
leakage-free preprocessing, baseline comparison, and responsible AI analysis.
"""

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.data import load_student_data
from src.model import (
    DEFAULT_ARTIFACT_PATH,
    LEAKAGE_COLUMNS,
    TARGET,
    TARGET_CLASSES,
    add_target,
    train_and_save,
)

st.set_page_config(
    page_title="EduPredict — Student Performance Analysis",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.title("🎓 EduPredict Controls")
st.sidebar.caption("Week 02 AI Engineering | Stack & Scale")

subject = st.sidebar.selectbox(
    "Select Subject Dataset",
    options=["mat", "por"],
    format_func=lambda s: "Mathematics (395 students)" if s == "mat" else "Portuguese Language (649 students)",
    index=0
)

active_model_type = st.sidebar.radio(
    "Active Model Architecture",
    options=["decision_tree", "logistic_regression", "compare_both"],
    format_func=lambda m: {
        "decision_tree": "Decision Tree (Constrained, Depth=3)",
        "logistic_regression": "Logistic Regression (Regularized, C=0.1)",
        "compare_both": "Side-by-Side Model Comparison"
    }[m]
)

st.sidebar.divider()
st.sidebar.markdown("""
### 🛡️ Leakage Safeguards
- **Target Source:** Derived strictly from `G3`.
- **Excluded Features:** `G1`, `G2`, `G3`.
- **Preprocessing:** Split performed *before* fitting imputers/scalers.
- **Evaluation:** Evaluated on held-out 20% test split.
""")


# ---------------------------------------------------------
# Model & Data Caching
# ---------------------------------------------------------
@st.cache_data
def get_dataset(subj: str):
    df = load_student_data(subj)
    return add_target(df)


@st.cache_resource
def get_trained_bundle(subj: str):
    artifact_path = DEFAULT_ARTIFACT_PATH.parent / f"edupredict_pipeline_{subj}.joblib"
    if not artifact_path.exists():
        raw_df = load_student_data(subj)
        train_and_save(raw_df, output_path=str(artifact_path), model_type="decision_tree")
    return joblib.load(artifact_path)


data = get_dataset(subject)
bundle = get_trained_bundle(subject)
primary_model = bundle["model"]
alt_model = bundle.get("alt_model")
metrics = bundle.get("metrics", {})
feature_columns = bundle["feature_columns"]


# ---------------------------------------------------------
# Header & KPI Metrics
# ---------------------------------------------------------
st.title("🎓 EduPredict — Student Performance Prediction & Analysis")
st.markdown(
    "An interactive AI engineering prototype exploring early academic performance indicators, "
    "transparent machine learning pipelines, and responsible algorithmic evaluation."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Dataset Sample Size", f"{len(data):,} students")
c2.metric("Predictive Features", f"{len(feature_columns)} inputs (Zero Leakage)")
c3.metric("Baseline Majority Acc.", f"{metrics.get('baseline_accuracy', 0.418):.1%}")
c4.metric(
    "Model Test Accuracy",
    f"{metrics.get('accuracy', 0.380):.1%}",
    delta=f"CV: {metrics.get('cv_mean', 0.509):.1%}"
)

st.divider()

# ---------------------------------------------------------
# Main Tabs
# ---------------------------------------------------------
tab_predict, tab_eda, tab_eval, tab_ethics = st.tabs([
    "🎯 Interactive Prediction",
    "📊 Exploratory Data Analysis",
    "📈 Model Evaluation & Safeguards",
    "⚖️ Ethical Use & Limitations"
])

# =========================================================
# TAB 1: INTERACTIVE PREDICTION
# =========================================================
with tab_predict:
    st.subheader("Student Profile Simulation & Target Prediction")
    st.write(
        "Enter observed academic and contextual factors for a hypothetical student. "
        "The model predicts one of three categories: **Needs Support** (G3: 0–9), "
        "**Satisfactory** (G3: 10–13), or **Strong** (G3: 14–20)."
    )

    # Preset profiles for easy demonstration
    col_preset, _ = st.columns([2, 1])
    with col_preset:
        preset = st.selectbox(
            "Load Sample Profile Preset (Optional):",
            ["Custom Input", "Needs Support / At-Risk Profile", "High Engagement Student Profile", "Typical Balanced Student"]
        )

    # Set default values based on preset
    defaults = {
        "school": "GP", "sex": "F", "age": 16, "address": "U", "famsize": "GT3", "Pstatus": "T",
        "Medu": 2, "Fedu": 2, "Mjob": "services", "Fjob": "other", "reason": "course", "guardian": "mother",
        "traveltime": 1, "studytime": 2, "failures": 0, "schoolsup": "no", "famsup": "yes", "paid": "no",
        "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes", "romantic": "no",
        "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1, "Walc": 1, "health": 4, "absences": 4
    }

    if preset == "Needs Support / At-Risk Profile":
        defaults.update({
            "studytime": 1, "failures": 2, "schoolsup": "yes", "higher": "no",
            "absences": 14, "goout": 5, "Walc": 4, "freetime": 4, "health": 2
        })
    elif preset == "High Engagement Student Profile":
        defaults.update({
            "studytime": 4, "failures": 0, "schoolsup": "no", "famsup": "yes", "higher": "yes",
            "absences": 2, "goout": 2, "Walc": 1, "Medu": 4, "Fedu": 4, "Mjob": "teacher"
        })

    with st.form("student_prediction_form"):
        st.markdown("#### 1. Core Academic & Behavioral Indicators")
        col_a, col_b, col_c = st.columns(3)

        with col_a:
            studytime = st.selectbox(
                "Weekly Study Time",
                options=[1, 2, 3, 4],
                index=[1, 2, 3, 4].index(defaults["studytime"]),
                format_func=lambda x: {
                    1: "< 2 hours",
                    2: "2 to 5 hours",
                    3: "5 to 10 hours",
                    4: "> 10 hours"
                }[x]
            )
            failures = st.slider("Past Class Failures", 0, 4, defaults["failures"])
            absences = st.slider("School Absences (Days)", 0, 75, defaults["absences"])

        with col_b:
            schoolsup = st.selectbox("Extra Educational School Support", ["no", "yes"], index=["no", "yes"].index(defaults["schoolsup"]))
            famsup = st.selectbox("Family Educational Support", ["yes", "no"], index=["yes", "no"].index(defaults["famsup"]))
            higher = st.selectbox("Aims for Higher Education", ["yes", "no"], index=["yes", "no"].index(defaults["higher"]))

        with col_c:
            goout = st.slider("Going Out with Friends (1=low, 5=high)", 1, 5, defaults["goout"])
            health = st.slider("Current Health Status (1=very bad, 5=very good)", 1, 5, defaults["health"])
            internet = st.selectbox("Internet Access at Home", ["yes", "no"], index=["yes", "no"].index(defaults["internet"]))

        with st.expander("➕ Additional Demographic, Family & Lifestyle Attributes (Pre-filled)"):
            exp_a, exp_b, exp_c = st.columns(3)
            with exp_a:
                school = st.selectbox("School", ["GP", "MS"], index=["GP", "MS"].index(defaults["school"]))
                sex = st.selectbox("Sex", ["F", "M"], index=["F", "M"].index(defaults["sex"]))
                age = st.slider("Age", 15, 22, defaults["age"])
                address = st.selectbox("Address Type", ["U", "R"], index=["U", "R"].index(defaults["address"]), format_func=lambda x: "Urban (U)" if x == "U" else "Rural (R)")
                famsize = st.selectbox("Family Size", ["GT3", "LE3"], index=["GT3", "LE3"].index(defaults["famsize"]), format_func=lambda x: "Greater than 3" if x == "GT3" else "3 or less")
                Pstatus = st.selectbox("Parents Cohabitation", ["T", "A"], index=["T", "A"].index(defaults["Pstatus"]), format_func=lambda x: "Living Together" if x == "T" else "Apart")

            with exp_b:
                Medu = st.selectbox("Mother's Education", [0, 1, 2, 3, 4], index=defaults["Medu"], format_func=lambda x: ["None", "Primary (4th grade)", "5th-9th grade", "Secondary", "Higher"][x])
                Fedu = st.selectbox("Father's Education", [0, 1, 2, 3, 4], index=defaults["Fedu"], format_func=lambda x: ["None", "Primary (4th grade)", "5th-9th grade", "Secondary", "Higher"][x])
                Mjob = st.selectbox("Mother's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(defaults["Mjob"]))
                Fjob = st.selectbox("Father's Job", ["teacher", "health", "services", "at_home", "other"], index=["teacher", "health", "services", "at_home", "other"].index(defaults["Fjob"]))
                guardian = st.selectbox("Guardian", ["mother", "father", "other"], index=["mother", "father", "other"].index(defaults["guardian"]))
                reason = st.selectbox("Reason for Choosing School", ["course", "home", "reputation", "other"], index=["course", "home", "reputation", "other"].index(defaults["reason"]))

            with exp_c:
                traveltime = st.selectbox("Travel Time to School", [1, 2, 3, 4], index=defaults["traveltime"] - 1, format_func=lambda x: ["< 15 min", "15-30 min", "30-60 min", "> 60 min"][x - 1])
                paid = st.selectbox("Extra Paid Classes", ["no", "yes"], index=["no", "yes"].index(defaults["paid"]))
                activities = st.selectbox("Extracurricular Activities", ["yes", "no"], index=["yes", "no"].index(defaults["activities"]))
                nursery = st.selectbox("Attended Nursery School", ["yes", "no"], index=["yes", "no"].index(defaults["nursery"]))
                romantic = st.selectbox("In a Romantic Relationship", ["no", "yes"], index=["no", "yes"].index(defaults["romantic"]))
                famrel = st.slider("Family Relationship Quality (1-5)", 1, 5, defaults["famrel"])
                freetime = st.slider("Free Time After School (1-5)", 1, 5, defaults["freetime"])
                Dalc = st.slider("Workday Alcohol Consumption (1-5)", 1, 5, defaults["Dalc"])
                Walc = st.slider("Weekend Alcohol Consumption (1-5)", 1, 5, defaults["Walc"])

        submitted = st.form_submit_button("Run Prediction Pipeline", type="primary")

    if submitted:
        input_data = {
            "school": school, "sex": sex, "age": age, "address": address, "famsize": famsize,
            "Pstatus": Pstatus, "Medu": Medu, "Fedu": Fedu, "Mjob": Mjob, "Fjob": Fjob,
            "reason": reason, "guardian": guardian, "traveltime": traveltime, "studytime": studytime,
            "failures": failures, "schoolsup": schoolsup, "famsup": famsup, "paid": paid,
            "activities": activities, "nursery": nursery, "higher": higher, "internet": internet,
            "romantic": romantic, "famrel": famrel, "freetime": freetime, "goout": goout,
            "Dalc": Dalc, "Walc": Walc, "health": health, "absences": absences
        }
        input_df = pd.DataFrame([input_data]).reindex(columns=feature_columns)

        st.markdown("### Prediction Results")

        def display_prediction_card(model_obj, model_label):
            pred = model_obj.predict(input_df)[0]
            probs = model_obj.predict_proba(input_df)[0]
            classes = model_obj.classes_

            color_map = {
                "Needs Support": "#d9534f",
                "Satisfactory": "#0275d8",
                "Strong": "#5cb85c"
            }
            bg_color = color_map.get(pred, "#333")

            st.markdown(
                f"""
                <div style="background-color: {bg_color}; padding: 15px; border-radius: 8px; color: white; margin-bottom: 12px;">
                    <h4 style="margin: 0; color: white;">{model_label}</h4>
                    <h2 style="margin: 5px 0; color: white;">Predicted Category: {pred}</h2>
                    <p style="margin: 0; opacity: 0.9;">Target definition: {bundle['target_definition'].get(pred, '')}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            prob_df = pd.DataFrame({"Probability": probs}, index=classes)
            st.bar_chart(prob_df)

        if active_model_type == "compare_both" and alt_model is not None:
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                display_prediction_card(primary_model, "Decision Tree Classifier")
            with col_m2:
                display_prediction_card(alt_model, "Logistic Regression Classifier")
        else:
            chosen = primary_model if active_model_type == "decision_tree" or alt_model is None else alt_model
            chosen_name = "Decision Tree Classifier" if active_model_type == "decision_tree" else "Logistic Regression Classifier"
            display_prediction_card(chosen, chosen_name)

        st.info("💡 **Pedagogical Note:** Notice that absent final grades (G1/G2), the model relies on behavioral proxies (past failures, study habits, absences). These predictions represent statistical tendencies, not destiny.")


# =========================================================
# TAB 2: EXPLORATORY DATA ANALYSIS
# =========================================================
with tab_eda:
    st.subheader("Dataset Exploration & Distribution Analysis")

    eda_row1_col1, eda_row1_col2 = st.columns(2)
    with eda_row1_col1:
        st.markdown("#### Target Class Balance")
        counts = data[TARGET].value_counts().reindex(TARGET_CLASSES)
        st.bar_chart(counts)
        st.caption("Distribution of the 3-category target constructed from final grade G3.")

    with eda_row1_col2:
        st.markdown("#### Distribution of Raw Final Grade (G3)")
        fig_g3, ax_g3 = plt.subplots(figsize=(6, 3.5))
        sns.histplot(data["G3"], bins=20, kde=True, color="#4A90E2", ax=ax_g3)
        ax_g3.axvline(9.5, color="red", linestyle="--", label="Fail Threshold (<10)")
        ax_g3.axvline(13.5, color="green", linestyle="--", label="Strong Threshold (>=14)")
        ax_g3.set_title("G3 Grade Distribution (0-20 scale)")
        ax_g3.set_xlabel("Final Grade (G3)")
        ax_g3.legend()
        st.pyplot(fig_g3)
        plt.close(fig_g3)

    st.divider()

    eda_row2_col1, eda_row2_col2 = st.columns(2)
    with eda_row2_col1:
        st.markdown("#### Study Time vs Performance Category")
        ct_study = pd.crosstab(data["studytime"], data[TARGET], normalize="index")
        st.bar_chart(ct_study)
        st.caption("Proportion of performance categories across study time levels (1: <2h to 4: >10h).")

    with eda_row2_col2:
        st.markdown("#### Absences by Performance Category")
        fig_abs, ax_abs = plt.subplots(figsize=(6, 3.5))
        sns.boxplot(data=data, x=TARGET, y="absences", order=TARGET_CLASSES, palette="Blues", ax=ax_abs)
        ax_abs.set_title("School Absences across Performance Tiers")
        ax_abs.set_ylabel("Number of Absences")
        st.pyplot(fig_abs)
        plt.close(fig_abs)

    st.divider()
    st.markdown("#### 🔥 Feature Correlation Heatmap & Importance Analysis")
    st.write(
        "Using a Pearson correlation heatmap allows us to identify features that correlate most strongly with academic performance, "
        "filtering out noisy indicators to optimize model accuracy while maintaining zero target leakage."
    )

    heatmap_mode = st.radio(
        "Heatmap Display Mode",
        options=["Clean Non-Leakage Features (For Model Training)", "Leakage Comparison (G1, G2 vs G3)"],
        horizontal=True
    )

    df_num = pd.get_dummies(data.drop(columns=[TARGET]), drop_first=True)
    corr = df_num.corr()

    if heatmap_mode.startswith("Clean"):
        # Drop G1, G2, G3 from features
        clean_corr = corr.drop(index=["G1", "G2", "G3"], columns=["G1", "G2"])["G3"].abs().sort_values(ascending=False)
        top_cols = clean_corr.head(12).index.tolist()
        corr_plot = df_num[top_cols + ["G3"]].corr()
        fig_hm, ax_hm = plt.subplots(figsize=(10, 7))
        mask = np.triu(np.ones_like(corr_plot, dtype=bool))
        sns.heatmap(corr_plot, mask=mask, annot=True, fmt=".2f", cmap="coolwarm", center=0, vmin=-0.4, vmax=0.4, ax=ax_hm, linewidths=0.5)
        ax_hm.set_title("Correlation Heatmap: Top 12 Clean Predictors vs Final Grade (G3)", fontsize=11, fontweight="bold")
        st.pyplot(fig_hm)
        plt.close(fig_hm)
        st.caption("Notice: Past failures (-0.36), mother's education (+0.22), and higher education desires (+0.18) are top clean drivers.")
    else:
        leakage_cols = ["G3", "G2", "G1", "failures", "Medu", "higher_yes", "age", "Fedu", "goout", "studytime"]
        corr_leak = df_num[leakage_cols].corr()
        fig_hm, ax_hm = plt.subplots(figsize=(9, 6.5))
        sns.heatmap(corr_leak, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-0.4, vmax=1.0, ax=ax_hm, linewidths=0.5)
        ax_hm.set_title("Leakage Demonstration: G1 (+0.80) & G2 (+0.90) Artificially Dominate G3", fontsize=11, fontweight="bold")
        st.pyplot(fig_hm)
        plt.close(fig_hm)
        st.warning("⚠️ While G1 and G2 have 0.80+ correlation, using them is target leakage because they are future/interim grades not available before courses begin.")

    st.divider()
    st.markdown("#### Raw Dataset Inspection")
    st.dataframe(data.head(20), use_container_width=True)


# =========================================================
# TAB 3: MODEL EVALUATION & SAFEGUARDS
# =========================================================
with tab_eval:
    st.subheader("Model Evaluation, Comparison & Leakage Safeguards")

    st.markdown("### 📋 Leakage Prevention Checklist")
    checklist_df = pd.DataFrame([
        {"Item": "Outcome Variable (G3)", "Status": "Target Only", "Details": "Binned into 3 tiers; excluded from input features."},
        {"Item": "Intermediate Grades (G1, G2)", "Status": "Strictly Excluded", "Details": "Excluded because earlier exams leak final grade."},
        {"Item": "Preprocessing Fit Scope", "Status": "Train-Split Only", "Details": "Imputer and Scaler fitted solely on training split."},
        {"Item": "Held-Out Verification", "Status": "Strict 80/20 Holdout", "Details": "20% stratified test split evaluated only once."},
    ])
    st.table(checklist_df)

    st.markdown("### 📊 Held-Out Test Evaluation & Baseline Comparison")
    ev_c1, ev_c2, ev_c3 = st.columns(3)
    ev_c1.metric("Majority Baseline Accuracy", f"{metrics.get('baseline_accuracy', 0.418):.3f}")
    ev_c2.metric("Decision Tree Accuracy", f"{metrics.get('accuracy', 0.380):.3f}", f"CV: {metrics.get('cv_mean', 0.509):.3f}")
    ev_c3.metric("Logistic Regression Accuracy", f"{metrics.get('alt_accuracy', 0.456):.3f}", f"CV: {metrics.get('alt_cv_mean', 0.474):.3f}")

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.markdown("#### Primary Decision Tree Feature Importances")
        clf = primary_model.named_steps["classifier"]
        if hasattr(clf, "feature_importances_"):
            pre = primary_model.named_steps["preprocessing"]
            num_cols = [c for c in feature_columns if pd.api.types.is_numeric_dtype(data[c])]
            cat_cols = [c for c in feature_columns if c not in num_cols]
            ohe = pre.named_transformers_["categorical"].named_steps["onehot"]
            feat_names = num_cols + ohe.get_feature_names_out(cat_cols).tolist()
            imp_s = pd.Series(clf.feature_importances_, index=feat_names).sort_values(ascending=False).head(10)
            st.bar_chart(imp_s)
            st.caption("Top 10 features driving splits in the Decision Tree.")

    with col_fig2:
        st.markdown("#### Error Diagnostic Analysis")
        st.markdown("""
        In held-out testing, when tabular non-grade features are used, model accuracy hovers around 38-46%.
        **Key Reasons for Misclassification:**
        1. **High variance in individual outcomes:** Two students with identical study times and parental backgrounds may have vastly different exam performance.
        2. **Absence of direct competence measures:** Because G1/G2 are excluded to prevent target leakage, the model relies on contextual proxies.
        3. **Class Overlap:** The boundary between Satisfactory (10-13) and Needs Support (0-9) is continuous.
        """)


# =========================================================
# TAB 4: ETHICAL USE & LIMITATIONS
# =========================================================
with tab_ethics:
    st.subheader("Responsible AI, Limitations & Ethical Principles")
    st.markdown("""
    ### ⚠️ Limitations of this Prototype
    1. **Not a Consequential Decision System:** This system is an educational simulation built for the Stack & Scale Week 02 project. It must **never** be used for automated admissions, grading, tracking, or resource allocation.
    2. **Demographic & Contextual Proxies:** The dataset includes variables such as parents' education (`Medu`, `Fedu`), job categories (`Mjob`, `Fjob`), and home address type (`address`). In machine learning models, these can encode historical socioeconomic inequalities rather than learner capability.
    3. **Correlation is Not Causation:** If the model shows an association between high absences and lower grades, it does not mean absence alone is the sole causal driver (e.g. underlying illness or caregiving responsibilities may be the true root cause).
    4. **Geographic & Historical Specificity:** The data originates from two secondary schools in the Alentejo region of Portugal between 2005 and 2006. It cannot be generalized to contemporary or international student cohorts.
    5. **Uncalibrated Confidence:** Decision tree leaf distributions do not represent true Bayesian posterior probabilities of student potential.

    ---
    *Built strictly following the Stack & Scale Remote Internship 2026 Week 02 AI Engineering Brief.*
    """)

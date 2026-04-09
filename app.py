import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LoanIntel",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title { font-size: 2.8rem; font-weight: 800; color: #1F5C99; margin-bottom: 0; }
    .sub-title   { font-size: 1.1rem; color: #555; margin-top: 0; margin-bottom: 2rem; }
    .approved    { background: #D4EDDA; color: #155724; padding: 1.5rem; border-radius: 12px;
                   font-size: 1.8rem; font-weight: 700; text-align: center; border-left: 6px solid #28a745; }
    .rejected    { background: #FDECEA; color: #721c24; padding: 1.5rem; border-radius: 12px;
                   font-size: 1.8rem; font-weight: 700; text-align: center; border-left: 6px solid #dc3545; }
    .metric-box  { background: #F0F4FA; border-radius: 10px; padding: 1rem; text-align: center; }
    .metric-val  { font-size: 1.8rem; font-weight: 700; color: #1F5C99; }
    .metric-lbl  { font-size: 0.85rem; color: #666; }
    .section-hdr { color: #1F5C99; font-size: 1.3rem; font-weight: 700; border-bottom: 2px solid #1F5C99;
                   padding-bottom: 0.3rem; margin-top: 1.5rem; margin-bottom: 1rem; }
    .info-box    { background: #EBF3FB; border-left: 4px solid #1F5C99; padding: 0.8rem 1rem;
                   border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #333; margin: 0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ─── DATA & MODEL PIPELINE ────────────────────────────────────────────────────
@st.cache_data
def load_and_train():
    df = pd.read_csv("loan_approval_data.csv")

    cat_cols = df.select_dtypes(include=["object"]).columns
    num_cols = df.select_dtypes(include="number").columns

    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

    df = df.drop("Applicant_ID", axis=1)

    le = LabelEncoder()
    df["Education_Level"] = le.fit_transform(df["Education_Level"])
    df["Loan_Approved"]   = le.fit_transform(df["Loan_Approved"])

    ohe_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                "Property_Area", "Gender", "Employer_Category"]
    ohe = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
    encoded    = ohe.fit_transform(df[ohe_cols])
    encoded_df = pd.DataFrame(encoded, columns=ohe.get_feature_names_out(ohe_cols), index=df.index)
    df = pd.concat([df.drop(columns=ohe_cols), encoded_df], axis=1)

    df["DTI_Ratio_sq"]    = df["DTI_Ratio"] ** 2
    df["Credit_Score_sq"] = df["Credit_Score"] ** 2

    X = df.drop(columns=["Loan_Approved", "Credit_Score", "DTI_Ratio"])
    y = df["Loan_Approved"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes":         GaussianNB(),
    }
    trained, metrics = {}, {}
    for name, m in models.items():
        m.fit(X_train_s, y_train)
        yp = m.predict(X_test_s)
        trained[name] = m
        metrics[name] = {
            "Accuracy":  round(accuracy_score(y_test, yp), 3),
            "Precision": round(precision_score(y_test, yp), 3),
            "Recall":    round(recall_score(y_test, yp), 3),
            "F1 Score":  round(f1_score(y_test, yp), 3),
            "CM":        confusion_matrix(y_test, yp),
        }

    return trained, metrics, scaler, ohe, X.columns.tolist(), df


# ─── GET UNIQUE VALUES FOR DROPDOWNS ──────────────────────────────────────────
@st.cache_data
def get_raw_options():
    df = pd.read_csv("loan_approval_data.csv").dropna()
    return {
        "Employment_Status":  sorted(df["Employment_Status"].dropna().unique()),
        "Marital_Status":     sorted(df["Marital_Status"].dropna().unique()),
        "Loan_Purpose":       sorted(df["Loan_Purpose"].dropna().unique()),
        "Property_Area":      sorted(df["Property_Area"].dropna().unique()),
        "Gender":             sorted(df["Gender"].dropna().unique()),
        "Employer_Category":  sorted(df["Employer_Category"].dropna().unique()),
        "Education_Level":    sorted(df["Education_Level"].dropna().unique()),
    }


trained_models, metrics, scaler, ohe, feature_cols, full_df = load_and_train()
options = get_raw_options()


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 LoanIntel")
    st.markdown("*Loan Approval Prediction System*")
    st.divider()
    page = st.radio("Navigate", ["🔮 Predict", "📊 Model Performance", "📈 Data Insights"])
    st.divider()
    selected_model = st.selectbox("Active Model", list(trained_models.keys()), index=2)
    st.markdown(f"""
    <div class='info-box'>
    <b>Selected:</b> {selected_model}<br>
    <b>Accuracy:</b> {metrics[selected_model]['Accuracy']*100:.1f}%<br>
    <b>Precision:</b> {metrics[selected_model]['Precision']*100:.1f}%
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    st.markdown("**About**")
    st.caption("Minor Project — ML Engineering\nDataset: 1000 applicants, 19 features\nBinary Classification: Approved / Rejected")


# ─── PAGE: PREDICT ────────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.markdown('<p class="main-title">🏦 LoanIntel</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-title">Loan Approval Prediction System using Machine Learning</p>', unsafe_allow_html=True)

    with st.form("predict_form"):
        st.markdown('<p class="section-hdr">👤 Personal Information</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            gender         = st.selectbox("Gender", options["Gender"])
            age            = st.number_input("Age", min_value=18, max_value=80, value=35)
        with c2:
            marital_status = st.selectbox("Marital Status", options["Marital_Status"])
            dependents     = st.number_input("Dependents", min_value=0, max_value=10, value=1)
        with c3:
            education      = st.selectbox("Education Level", options["Education_Level"])
            employment     = st.selectbox("Employment Status", options["Employment_Status"])

        st.markdown('<p class="section-hdr">💰 Financial Information</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            applicant_income   = st.number_input("Applicant Income (₹)", min_value=0, value=50000, step=1000)
            coapplicant_income = st.number_input("Co-applicant Income (₹)", min_value=0, value=0, step=1000)
        with c2:
            credit_score   = st.slider("Credit Score", min_value=300, max_value=900, value=650)
            savings        = st.number_input("Savings (₹)", min_value=0, value=100000, step=5000)
        with c3:
            existing_loans = st.number_input("Existing Loans", min_value=0, max_value=20, value=1)
            dti_ratio      = st.slider("DTI Ratio", min_value=0.0, max_value=1.0, value=0.3, step=0.01,
                                       help="Debt-to-Income Ratio: monthly debt / monthly income")

        st.markdown('<p class="section-hdr">🏠 Loan & Property Details</p>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            loan_amount    = st.number_input("Loan Amount (₹)", min_value=0, value=200000, step=5000)
            loan_term      = st.number_input("Loan Term (months)", min_value=6, max_value=360, value=120)
        with c2:
            loan_purpose   = st.selectbox("Loan Purpose", options["Loan_Purpose"])
            property_area  = st.selectbox("Property Area", options["Property_Area"])
        with c3:
            employer_cat   = st.selectbox("Employer Category", options["Employer_Category"])
            collateral_val = st.number_input("Collateral Value (₹)", min_value=0, value=300000, step=10000)

        submitted = st.form_submit_button("🔮 Predict Loan Approval", use_container_width=True, type="primary")

    if submitted:
        edu_map = {"Graduate": 1, "Not Graduate": 0}
        edu_encoded = edu_map.get(education, 0)

        input_dict = {
            "Applicant_Income":   applicant_income,
            "Coapplicant_Income": coapplicant_income,
            "Age":                age,
            "Dependents":         dependents,
            "Existing_Loans":     existing_loans,
            "Savings":            savings,
            "Collateral_Value":   collateral_val,
            "Loan_Amount":        loan_amount,
            "Loan_Term":          loan_term,
            "Education_Level":    edu_encoded,
            "Employment_Status":  employment,
            "Marital_Status":     marital_status,
            "Loan_Purpose":       loan_purpose,
            "Property_Area":      property_area,
            "Gender":             gender,
            "Employer_Category":  employer_cat,
            "DTI_Ratio":          dti_ratio,
            "Credit_Score":       credit_score,
        }
        raw_input = pd.DataFrame([input_dict])

        cat_cols_ohe = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                        "Property_Area", "Gender", "Employer_Category"]
        ohe_out  = ohe.transform(raw_input[cat_cols_ohe])
        ohe_part = pd.DataFrame(ohe_out, columns=ohe.get_feature_names_out(cat_cols_ohe))

        num_part = raw_input.drop(columns=cat_cols_ohe + ["DTI_Ratio", "Credit_Score"]).reset_index(drop=True)
        num_part["DTI_Ratio_sq"]    = dti_ratio ** 2
        num_part["Credit_Score_sq"] = credit_score ** 2

        final_input = pd.concat([num_part, ohe_part], axis=1)
        final_input = final_input.reindex(columns=feature_cols, fill_value=0)

        scaled = scaler.transform(final_input)

        model = trained_models[selected_model]
        pred  = model.predict(scaled)[0]
        prob  = model.predict_proba(scaled)[0] if hasattr(model, "predict_proba") else None

        st.divider()
        col1, col2 = st.columns([2, 1])
        with col1:
            if pred == 1:
                st.markdown('<div class="approved">✅ LOAN APPROVED</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="rejected">❌ LOAN REJECTED</div>', unsafe_allow_html=True)
        with col2:
            if prob is not None:
                st.metric("Model Confidence", f"{prob[pred]*100:.1f}%")
                st.metric("Model Used", selected_model)

        if prob is not None:
            st.divider()
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(5, 2.5))
                bars = ax.barh(["Rejected", "Approved"], [prob[0]*100, prob[1]*100],
                               color=["#dc3545", "#28a745"], height=0.5)
                ax.set_xlim(0, 100)
                ax.bar_label(bars, fmt="%.1f%%", padding=4)
                ax.set_xlabel("Probability (%)")
                ax.set_title("Approval Probability")
                ax.spines[["top", "right"]].set_visible(False)
                st.pyplot(fig)
                plt.close()
            with c2:
                st.markdown("**Key Risk Indicators**")
                dti_risk   = "🔴 High"   if dti_ratio > 0.45    else ("🟡 Medium" if dti_ratio > 0.3    else "🟢 Low")
                cred_risk  = "🔴 Low"    if credit_score < 550  else ("🟡 Fair"   if credit_score < 700 else "🟢 Good")
                loans_risk = "🔴 High"   if existing_loans >= 3 else ("🟡 Medium" if existing_loans >= 2 else "🟢 Low")
                st.markdown(f"**DTI Ratio ({dti_ratio:.2f}):** {dti_risk}")
                st.markdown(f"**Credit Score ({credit_score}):** {cred_risk}")
                st.markdown(f"**Existing Loans ({existing_loans}):** {loans_risk}")


# ─── PAGE: MODEL PERFORMANCE ──────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.markdown('<p class="main-title">📊 Model Performance</p>', unsafe_allow_html=True)
    st.markdown("Comparison of all three classifiers trained on the LoanIntel dataset.")
    st.divider()

    metric_names = ["Accuracy", "Precision", "Recall", "F1 Score"]
    rows = []
    for mname in trained_models:
        row = {"Model": mname}
        for mn in metric_names:
            row[mn] = metrics[mname][mn]
        rows.append(row)
    mdf = pd.DataFrame(rows).set_index("Model")

    st.markdown('<p class="section-hdr">Metric Comparison Table</p>', unsafe_allow_html=True)
    st.dataframe(
        mdf.style.highlight_max(axis=0, color="#D4EDDA").format("{:.3f}"),
        use_container_width=True
    )
    st.caption("Green = best value per metric column")

    st.divider()
    st.markdown('<p class="section-hdr">Bar Chart Comparison</p>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(metric_names))
    w = 0.25
    colors = ["#1F5C99", "#E07B3F", "#2A9D5C"]
    for i, (mname, color) in enumerate(zip(trained_models, colors)):
        vals = [metrics[mname][mn] for mn in metric_names]
        bars = ax.bar(x + i*w, vals, w, label=mname, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.2f", fontsize=8, padding=2)
    ax.set_xticks(x + w)
    ax.set_xticklabels(metric_names)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown('<p class="section-hdr">Confusion Matrices</p>', unsafe_allow_html=True)
    cols = st.columns(3)
    for i, mname in enumerate(trained_models):
        with cols[i]:
            cm = metrics[mname]["CM"]
            fig, ax = plt.subplots(figsize=(3.5, 3))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"],
                        linewidths=0.5, cbar=False)
            ax.set_title(mname, fontsize=10, fontweight="bold")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
            tn, fp, fn, tp = cm.ravel()
            st.caption(f"TP:{tp}  FP:{fp}  TN:{tn}  FN:{fn}")

    st.divider()
    st.markdown('<p class="section-hdr">Why Naive Bayes was Selected</p>', unsafe_allow_html=True)
    st.info("""
    **Business Rationale:** In banking, approving a bad loan (False Positive) is more costly than
    rejecting a good customer (False Negative). Therefore, **Precision** is the primary selection metric.

    Naive Bayes is the recommended model due to its speed, interpretability, and strong probabilistic
    foundation — making it ideal for explaining decisions to stakeholders.
    """)


# ─── PAGE: DATA INSIGHTS ──────────────────────────────────────────────────────
elif page == "📈 Data Insights":
    st.markdown('<p class="main-title">📈 Data Insights</p>', unsafe_allow_html=True)
    st.markdown("Exploratory Data Analysis on the LoanIntel dataset.")
    st.divider()

    raw_df = pd.read_csv("loan_approval_data.csv")
    raw_df_clean = raw_df.copy()
    cat_c = raw_df_clean.select_dtypes(include=["object"]).columns
    num_c = raw_df_clean.select_dtypes(include="number").columns
    raw_df_clean[num_c] = SimpleImputer(strategy="mean").fit_transform(raw_df_clean[num_c])
    raw_df_clean[cat_c] = SimpleImputer(strategy="most_frequent").fit_transform(raw_df_clean[cat_c])

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", "1,000")
    c2.metric("Features", "19")
    c3.metric("Approved",  f"{(raw_df_clean['Loan_Approved'] == 'Yes').sum()} (29.8%)")
    c4.metric("Rejected",  f"{(raw_df_clean['Loan_Approved'] == 'No').sum()} (65.2%)")

    st.divider()
    st.markdown('<p class="section-hdr">Class Distribution</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 4))
        counts = raw_df_clean["Loan_Approved"].value_counts()
        ax.pie(counts, labels=["No", "Yes"], autopct="%1.1f%%",
               colors=["#dc3545", "#28a745"], startangle=90)
        ax.set_title("Loan Approval Distribution")
        st.pyplot(fig)
        plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(data=raw_df_clean, x="Credit_Score", hue="Loan_Approved",
                     bins=20, multiple="dodge", ax=ax,
                     palette={"No": "#dc3545", "Yes": "#28a745"})
        ax.set_title("Credit Score by Approval Status")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown('<p class="section-hdr">Income & Loan Distribution</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=raw_df_clean, x="Loan_Approved", y="Applicant_Income",
                    ax=ax, palette={"No": "#dc3545", "Yes": "#28a745"})
        ax.set_title("Applicant Income vs Approval")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.boxplot(data=raw_df_clean, x="Loan_Approved", y="DTI_Ratio",
                    ax=ax, palette={"No": "#dc3545", "Yes": "#28a745"})
        ax.set_title("DTI Ratio vs Approval")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()

    st.divider()
    st.markdown('<p class="section-hdr">Categorical Breakdowns</p>', unsafe_allow_html=True)
    cat_feature = st.selectbox("Select category", [
        "Property_Area", "Education_Level", "Employment_Status",
        "Gender", "Loan_Purpose", "Marital_Status", "Employer_Category"
    ])
    fig, ax = plt.subplots(figsize=(8, 4))
    ct = pd.crosstab(raw_df_clean[cat_feature], raw_df_clean["Loan_Approved"])
    ct.plot(kind="bar", ax=ax, color=["#dc3545", "#28a745"], rot=30, alpha=0.85)
    ax.set_title(f"{cat_feature} vs Loan Approval")
    ax.set_ylabel("Count")
    ax.legend(["No", "Yes"])
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig)
    plt.close()

    st.divider()
    st.markdown('<p class="section-hdr">Feature Correlation with Target</p>', unsafe_allow_html=True)
    target_corr = full_df.select_dtypes(include="number").corr()["Loan_Approved"].drop("Loan_Approved").sort_values(ascending=False)
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 5))
        target_corr.plot(kind="barh", ax=ax,
                         color=["#28a745" if v > 0 else "#dc3545" for v in target_corr])
        ax.set_title("Feature Correlation with Loan_Approved")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig)
        plt.close()
    with c2:
        st.markdown("**Top Positive Correlators**")
        for feat, val in target_corr[target_corr > 0.05].items():
            st.markdown(f"🟢 `{feat}` → **{val:.3f}**")
        st.markdown("**Top Negative Correlators**")
        for feat, val in target_corr[target_corr < -0.05].items():
            st.markdown(f"🔴 `{feat}` → **{val:.3f}**")

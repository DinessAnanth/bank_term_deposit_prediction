import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import shap
import matplotlib.pyplot as plt

# ==============================================================================
# 1. Page Configuration
# ==============================================================================
st.set_page_config(
    page_title="Bank Term Deposit Predictor",
    page_icon="🏦",  # we can get the icon from https://hexmos.com/freedevtools/emojis/ , copy & paste the icon into the code
    layout="wide"
)

st.title("🏦 Bank Term Deposit Prediction Dashboard")
st.markdown("""
Welcome to the predictive dashboard. Enter the customer's details in the sidebar to predict the probability 
that they will subscribe to a term deposit. This tool is powered by our dynamically optimized **Champion Machine Learning Model**.
""")

# ==============================================================================
# 2. Load Artifacts
# ==============================================================================


@st.cache_resource
def load_artifacts():
    # Dynamically locate the project root, assuming this script is in the 'app/' folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    # Construct paths to the models and data
    preprocessor_path = os.path.join(
        project_root, 'models', 'preprocessor.joblib')
    model_path = os.path.join(project_root, 'models', 'champion_model.joblib')
    background_path = os.path.join(
        project_root, 'data', 'X_train_processed.npy')

    preprocessor = joblib.load(preprocessor_path)
    model = joblib.load(model_path)

    # Load background data for SHAP (only the first 100 rows to save memory and optimize speed)
    background_data = np.load(background_path)[:100]

    return preprocessor, model, background_data


try:
    preprocessor, model, background_data = load_artifacts()
    model_name = type(model).__name__
    st.sidebar.success(f"✅ Preprocessor & {model_name} Loaded Successfully")
except Exception as e:
    st.sidebar.error(
        f"Error loading artifacts: {e}. Ensure you have trained the models and they exist in the '/models' and '/data' folders.")

# ==============================================================================
# 3. User Input Panel (Sidebar)
# ==============================================================================
st.sidebar.header("📝 Customer Profile")


def get_user_input():
    age = st.sidebar.slider("Age", 18, 120, 35)
    job = st.sidebar.selectbox("Job", ["management", "technician", "entrepreneur", "blue-collar", "unknown",
                               "retired", "admin.", "services", "self-employed", "unemployed", "housemaid", "student"])
    marital = st.sidebar.selectbox(
        "Marital Status", ["married", "single", "divorced"])
    education = st.sidebar.selectbox(
        "Education", ["tertiary", "secondary", "unknown", "primary"])
    default = st.sidebar.selectbox("Has Credit in Default?", ["no", "yes"])
    balance = st.sidebar.number_input("Yearly Balance (EUR)", value=1500)
    housing = st.sidebar.selectbox("Has Housing Loan?", ["yes", "no"])
    loan = st.sidebar.selectbox("Has Personal Loan?", ["no", "yes"])

    st.sidebar.markdown("---")
    st.sidebar.header("📞 Campaign Contact details")
    contact = st.sidebar.selectbox("Contact Communication Type", [
                                   "cellular", "unknown", "telephone"])
    day = st.sidebar.slider("Last Contact Day of Month", 1, 31, 15)
    month = st.sidebar.selectbox("Last Contact Month", [
                                 "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
    duration = st.sidebar.number_input(
        "Last Contact Duration (seconds)", value=200)
    campaign = st.sidebar.slider(
        "Number of Contacts (Current Campaign)", 1, 100, 2)
    pdays = st.sidebar.number_input(
        "Days Since Previous Contact (-1 = never)", value=-1)
    previous = st.sidebar.number_input(
        "Number of Contacts (Previous Campaign)", value=0)
    poutcome = st.sidebar.selectbox("Previous Campaign Outcome", [
                                    "unknown", "other", "failure", "success"])

    # Empty space at the bottom of the sidebar to prevent dropdown clipping
    st.sidebar.markdown("<br><br><br>", unsafe_allow_html=True)

    # Store inputs in a DataFrame matching the original train.csv columns
    user_data = pd.DataFrame({
        'age': [age], 'job': [job], 'marital': [marital], 'education': [education],
        'default': [default], 'balance': [balance], 'housing': [housing], 'loan': [loan],
        'contact': [contact], 'day': [day], 'month': [month], 'duration': [duration],
        'campaign': [campaign], 'pdays': [pdays], 'previous': [previous], 'poutcome': [poutcome]
    })

    return user_data


input_df = get_user_input()

st.subheader("Current Customer Data")
st.dataframe(input_df)

# ==============================================================================
# 4. Feature Engineering & Prediction Logic
# ==============================================================================
if st.button("🚀 Predict Subscription Probability", type="primary"):

    # Step A: Apply exact feature engineering mapped from Notebook 2
    engineered_df = input_df.copy()
    engineered_df["was_previously_contacted"] = np.where(
        engineered_df["pdays"] == -1, 0, 1)
    engineered_df["balance_status"] = np.where(
        engineered_df["balance"] >= 0, "non_negative", "negative")
    engineered_df["campaign_intensity"] = pd.cut(
        engineered_df["campaign"],
        bins=[-np.inf, 1, 3, np.inf],
        labels=["low", "medium", "high"]
    ).astype("object")

    # Step B: Transform through the ColumnTransformer
    try:
        processed_data = preprocessor.transform(engineered_df)
        if hasattr(processed_data, "toarray"):
            processed_data = processed_data.toarray()

        # Step C: Predict probabilities
        probability = model.predict_proba(processed_data)[0][1]
        prediction = model.predict(processed_data)[0]

        # Step D: Display Results
        st.markdown("---")
        st.subheader("🔮 Prediction Results")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.success("Prediction: **CUSTOMER WILL SUBSCRIBE**")
            else:
                st.error("Prediction: **CUSTOMER WILL NOT SUBSCRIBE**")

        with col2:
            st.metric(label="Probability of Subscription",
                      value=f"{probability * 100:.2f}%")

        st.progress(float(probability))

        # ==============================================================================
        # 5. Explainable AI (SHAP)
        # ==============================================================================
        st.markdown("---")
        st.subheader(f"🧠 Model Explainability ({model_name})")

        st.info("""
        **💡 How to read the SHAP chart below:**
        * **Bottom to Top:** Read the chart from the bottom (the average baseline probability) up to the top (the final predicted probability).
        * 🔴 **Red Bars:** Customer traits pushing the probability **UP** (making them more likely to subscribe).
        * 🔵 **Blue Bars:** Customer traits pulling the probability **DOWN** (making them less likely to subscribe).
        * **Bar Width:** The wider the bar, the stronger the impact that specific detail had on the final decision.
        """)

        with st.spinner("Generating SHAP explanation..."):
            try:
                # Extract clean feature names
                try:
                    raw_features = list(preprocessor.get_feature_names_out())
                    feature_names = [f.replace('num__', '').replace(
                        'cat__', '') for f in raw_features]
                except:
                    feature_names = [f"Feature {i}" for i in range(
                        processed_data.shape[1])]

                # DYNAMIC EXPLAINER INITIALIZATION
                model_type = type(model).__name__
                if model_type in ['RandomForestClassifier', 'DecisionTreeClassifier', 'XGBClassifier', 'LGBMClassifier']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer(processed_data)

                    # Ensure 2D array for waterfall plot logic
                    if len(shap_values.shape) == 3:
                        shap_values = shap_values[:, :, 1]
                else:
                    # Fallback Explainer for Linear Models/Neural Networks
                    # Utilizing the loaded 100-row static background_data slice to prevent 0 variance failures
                    explainer = shap.Explainer(model.predict, background_data)
                    shap_values = explainer(processed_data)

                # Assign feature names directly to the Explanation object
                shap_values.feature_names = feature_names
                single_shap = shap_values[0]

                # Generate Waterfall Plot
                fig, ax = plt.subplots(figsize=(8, 5))
                shap.plots.waterfall(single_shap, show=False)
                st.pyplot(fig)
                plt.clf()

                # --- PLAIN ENGLISH BREAKDOWN ---
                def clean_feature_name(name):
                    clean_name = str(name).replace('_', ' ').title()
                    return clean_name

                st.markdown("### 📝 Summary Of The Graph")

                raw_names = single_shap.feature_names
                shap_impacts = single_shap.values
                feature_values = single_shap.data
                clean_names = [clean_feature_name(n) for n in raw_names]

                feature_details = list(
                    zip(clean_names, shap_impacts, feature_values))
                feature_details.sort(key=lambda x: abs(x[1]), reverse=True)

                pushing_yes = [f for f in feature_details if f[1] > 0]
                pushing_no = [f for f in feature_details if f[1] < 0]

                # Dynamic Conclusive Summary
                if prediction == 1:
                    top_reasons = [f[0] for f in pushing_yes[:2]]
                    reasons_str = " and ".join(
                        top_reasons) if top_reasons else "various factors"
                    st.success(
                        f"**Final Verdict:** The model predicts this customer **WILL SUBSCRIBE**, primarily driven by their **{reasons_str}**.")
                else:
                    top_reasons = [f[0] for f in pushing_no[:2]]
                    reasons_str = " and ".join(
                        top_reasons) if top_reasons else "various factors"
                    st.error(
                        f"**Final Verdict:** The model predicts this customer **WILL NOT SUBSCRIBE**, primarily driven by their **{reasons_str}**.")

                st.markdown("<br>", unsafe_allow_html=True)

                if pushing_yes:
                    st.markdown(
                        "**🟢 Top factors making them MORE likely to subscribe:**")
                    for name, impact, val in pushing_yes[:3]:
                        display_val = round(val, 2) if isinstance(
                            val, (int, float)) else val
                        st.write(
                            f"- **{name}** (Current Value: `{display_val}`)")

                if pushing_no:
                    st.markdown(
                        "**🔴 Top factors making them LESS likely to subscribe:**")
                    for name, impact, val in pushing_no[:3]:
                        display_val = round(val, 2) if isinstance(
                            val, (int, float)) else val
                        st.write(
                            f"- **{name}** (Current Value: `{display_val}`)")

            except Exception as shap_error:
                st.warning(
                    "⚠️ Could not generate SHAP explanation for this exact instance.")
                st.info(f"SHAP Error Detail: {shap_error}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

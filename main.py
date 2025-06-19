import os
import pickle
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import lime.lime_tabular
import shap
import matplotlib.pyplot as plt
import warnings

# === Silence warnings ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# === Page Configuration ===
st.set_page_config(page_title="XAI-NIDS", layout="wide")
st.title("🛡️ Explainable AI - Network Intrusion Detection System")

# === Model Accuracies ===
MODEL_ACCURACIES = {
    "Random Forest": 0.9966,
    "Logistic Regression": 0.9042,
    "Decision Tree": 0.9965,
    "XGBoost": 0.9956,
    "LightGBM": 0.6808
}

# === Sidebar ===
dataset = st.sidebar.selectbox("Select the dataset", [" 📀 CICIDS - 2017", " 📀 SIMARGI - 2022"], index=0)
model_options = [f"{name} (Acc: {MODEL_ACCURACIES[name]*100:.2f}%)" for name in MODEL_ACCURACIES]
selected_model_display = st.sidebar.selectbox("Select the model", model_options, index=0)

# Defensive fix if user switches selectbox to multiselect
if isinstance(selected_model_display, list):
    selected_model_display = selected_model_display[0]

if selected_model_display.split(" ("[0])[0].lower() in [ "xgboost", "lightgbm"]:
    selected_model_name = selected_model_display.split(" ("[0])[0].lower()
else:
    selected_model_name = selected_model_display.split(" ("[0])[0].lower() + "_" +  selected_model_display.split(" ("[0])[1].lower()

if "simargi" in dataset.lower():
    st.warning("🚧 SIMARGI dataset support is under development. Please select CICIDS-2017.")
    st.stop()

MODEL_PATH = os.path.join(".", dataset.lower().split()[1], "models")
model_filename = f"{selected_model_name.lower()}.pkl"

@st.cache_data
def load_model_artifacts(model_file):
    scaler = joblib.load(os.path.join(MODEL_PATH, "scaler.pkl"))
    model = joblib.load(os.path.join(MODEL_PATH, model_file))
    with open(os.path.join(MODEL_PATH, "label_encoder.pkl"), "rb") as f:
        label_encoder = pickle.load(f)
    return scaler, model, label_encoder

model_loaded = False
try:
    scaler, model, le = load_model_artifacts(model_filename)
    st.sidebar.success("✅ Model loaded successfully.")
    model_loaded = True
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")

# === Input Section ===
st.subheader("📥 Input Network Traffic Features")
default_input = {
    "total_length_of_fwd_packets": 6.0,
    "total_length_of_bwd_packets": 6.0,
    "fwd_packet_length_max": 6.0,
    "bwd_packet_length_max": 6.0,
    "bwd_packet_length_mean": 6.0,
    "max_packet_length": 6.0,
    "packet_length_mean": 6.0,
    "packet_length_std": 0.0,
    "packet_length_variance": 0.0,
    "average_packet_size": 9.0,
    "avg_bwd_segment_size": 6.0,
    "subflow_fwd_bytes": 6.0,
    "subflow_bwd_bytes": 6.0,
    "init_win_bytes_forward": 237.0,
    "init_win_bytes_backward": 256.0
}

input_json = {}
cols = st.columns(3)
for i, key in enumerate(default_input.keys()):
    with cols[i % 3]:
        input_json[key] = st.number_input(key, value=default_input[key], format="%.4f")

predict_btn = st.button("🚀 Predict")

if predict_btn and model_loaded:
    st.divider()
    st.subheader("🎯 Model Prediction")

    feature_names = list(input_json.keys())
    input_df = pd.DataFrame([input_json])
    input_scaled = scaler.transform(input_df)

    y_pred = model.predict(input_scaled)
    y_proba = model.predict_proba(input_scaled)
    predicted_label = y_pred[0]
    predicted_class_name = le.inverse_transform([predicted_label])[0]

    st.success(f"🟢 Predicted Class: **{predicted_class_name}**")

    st.subheader("📊 Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": le.inverse_transform(np.arange(len(y_proba[0]))),
        "Probability": y_proba[0]
    }).query("Probability > 0.01").sort_values(by="Probability", ascending=False)
    st.dataframe(prob_df, use_container_width=True)

    def generate_synthetic_data(n=100):
        np.random.seed(42)
        return pd.DataFrame([{k: np.random.normal(loc=v, scale=1) if isinstance(v, float) else v
                              for k, v in default_input.items()} for _ in range(n)])

    training_df = generate_synthetic_data()
    training_scaled = scaler.transform(training_df)

    # === LIME ===
    st.subheader("🧠 LIME Explanation")
    explainer = lime.lime_tabular.LimeTabularExplainer(
    training_scaled,
    feature_names=feature_names,
    class_names=list(le.classes_),
    mode='classification',
    discretize_continuous=True,
    random_state=42
    )

    lime_exp = explainer.explain_instance(input_scaled[0], model.predict_proba, num_features=15, top_labels=5)

    st.markdown(f"### 🔍 Top 5 Feature Contributions for `{predicted_class_name}`")
    for i, (feat, val) in enumerate(sorted(lime_exp.as_list(label=predicted_label), key=lambda x: abs(x[1]), reverse=True)[:5], 1):
        direction = "↑ increases" if val > 0 else "↓ decreases"
        st.write(f"{i}. `{feat}` → **{val:+.4f}** ({direction} prediction)")

    # Plot with title
    fig = lime_exp.as_pyplot_figure(label=predicted_label)
    st.pyplot(fig)

    st.markdown("**Interactive LIME Chart:**")
    st.components.v1.html(lime_exp.as_html(labels=[predicted_label]), height=600, scrolling=True)


   # === SHAP ===
    st.subheader("📌 SHAP Explanation")
    try:
        explainer_shap = shap.Explainer(model, training_scaled)
        shap_values_input = explainer_shap(input_scaled)
        shap_values_train = explainer_shap(training_scaled)

        shap_values_input.feature_names = feature_names
        shap_values_train.feature_names = feature_names

        if len(shap_values_input.shape) == 3:
            num_classes = shap_values_input.shape[2]
            safe_class_index = min(predicted_label, num_classes - 1)
            shap_input = shap_values_input[:, :, safe_class_index]
            shap_train = shap_values_train[:, :, safe_class_index]
        else:
            shap_input = shap_values_input
            shap_train = shap_values_train

        # === Bar Plot
        st.markdown(f"### 📊 SHAP Bar Plot for `{predicted_class_name}`")
        fig, ax = plt.subplots()
        shap.plots.bar(shap_input[0], max_display=15, show=False)
        plt.title(f'SHAP Bar Plot – Class: {predicted_class_name}')
        st.pyplot(fig)

        # === Beeswarm Plot
        st.markdown(f"### 🐝 SHAP Beeswarm Plot for `{predicted_class_name}`")
        fig2, ax2 = plt.subplots()
        shap.plots.beeswarm(shap_train, max_display=15, show=False)
        plt.title(f'SHAP Beeswarm Plot – Class: {predicted_class_name}')
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ SHAP explanation failed: {str(e)}")

elif predict_btn:
    st.error("❌ Model not loaded. Please check the selected dataset or model.")

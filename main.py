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
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*SHAP.*")
warnings.filterwarnings("ignore", message=".*shap.*")
warnings.filterwarnings("ignore", message=".*LightGBM.*")
warnings.filterwarnings("ignore", message=".*lightgbm.*")

# === Page Configuration ===
st.set_page_config(page_title="XAI-NIDS", layout="wide")
st.title("🛡️ Explainable AI - Network Intrusion Detection System")

# === Model Accuracies ===
MODEL_ACCURACIES = {
    "Logistic Regression": 0.9042,
    "KNN": 0.9954,
    "Decision Tree": 0.9965,
    "Random Forest": 0.9966,
    "XGBoost": 0.9956,
    "LightGBM": 0.6808
}

# === Sidebar Selectboxes ===
dataset = st.sidebar.selectbox(
    "Select the dataset",
    [" 📀 CICIDS - 2017", " 📀 SIMARGI - 2022"],
    index=0,
    key="dataset_select"
)

model_options = [
    f"{name} (Acc: {MODEL_ACCURACIES[name]*100:.2f}%)"
    for name in MODEL_ACCURACIES
]
selected_model_display = st.sidebar.selectbox("Select the model", model_options, index=3)
selected_model_name = selected_model_display.split(" (")[0]

if "simargi" in dataset.lower():
    st.warning("🚧 SIMARGI dataset support is under development. Please select CICIDS-2017.")
    st.stop()

# === Prepare model file name and path ===
model_filename = f"{selected_model_name.lower().replace(' ', '_')}.pkl"
dataset_folder = dataset.lower().split()[1]
MODEL_PATH = os.path.join(".", dataset_folder, "models")

# === Load Model, Scaler, and LabelEncoder ===
@st.cache_data
def load_model_artifacts(model_file):
    scaler_path = os.path.join(MODEL_PATH, "scaler.pkl")
    label_encoder_path = os.path.join(MODEL_PATH, "label_encoder.pkl")
    model_path = os.path.join(MODEL_PATH, model_file)

    scaler = joblib.load(scaler_path)
    model = joblib.load(model_path)
    with open(label_encoder_path, "rb") as f:
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
    # === Prediction Section ===
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
    st.info(f"🔢 Probability: `{y_proba[0][predicted_label]:.4f}`")

    st.subheader("📊 Class Probabilities")
    prob_df = pd.DataFrame({
        "Class": le.inverse_transform(np.arange(len(y_proba[0]))),
        "Probability": y_proba[0]
    }).query("Probability > 0.01").sort_values(by="Probability", ascending=False)
    st.dataframe(prob_df, use_container_width=True)

    # === Generate synthetic data for explanation ===
    def generate_synthetic_data(n=100):
        np.random.seed(42)
        return pd.DataFrame([{
            "total_length_of_fwd_packets": np.random.randint(100, 10000),
            "total_length_of_bwd_packets": np.random.randint(1000, 50000),
            "fwd_packet_length_max": np.random.randint(100, 5000),
            "bwd_packet_length_max": np.random.randint(1000, 15000),
            "bwd_packet_length_mean": np.random.uniform(500, 5000),
            "max_packet_length": np.random.randint(1000, 15000),
            "packet_length_mean": np.random.uniform(200, 2000),
            "packet_length_std": np.random.uniform(500, 5000),
            "packet_length_variance": np.random.uniform(1e6, 1e7),
            "average_packet_size": np.random.uniform(200, 2000),
            "avg_bwd_segment_size": np.random.uniform(500, 5000),
            "subflow_fwd_bytes": np.random.randint(100, 10000),
            "subflow_bwd_bytes": np.random.randint(1000, 50000),
            "init_win_bytes_forward": np.random.randint(100, 1000),
            "init_win_bytes_backward": np.random.randint(100, 1000)
        } for _ in range(n)])

    training_df = generate_synthetic_data()
    training_scaled = scaler.transform(training_df)

    # === LIME Explanation ===
    st.subheader("🧠 LIME Explanation")
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_scaled,
        feature_names=feature_names,
        class_names=list(le.classes_),
        mode='classification',
        discretize_continuous=True,
        random_state=42
    )

    lime_exp = explainer.explain_instance(
        input_scaled[0],
        model.predict_proba,
        num_features=15,
        top_labels=5
    )

    exp_list = lime_exp.as_list(label=predicted_label)
    st.markdown("Top 5 Feature Contributions")
    for i, (feat, val) in enumerate(sorted(exp_list, key=lambda x: abs(x[1]), reverse=True)[:5], 1):
        direction = "↑ increases" if val > 0 else "↓ decreases"
        st.write(f"{i}. `{feat}` → **{val:+.4f}** ({direction} prediction)")


    fig_lime = lime_exp.as_pyplot_figure(label=predicted_label)
    st.pyplot(fig_lime)

    # === HTML Output with Light Background ===
    st.markdown("**Interactive LIME Chart:**")
    html_content = lime_exp.as_html(labels=[predicted_label])
    st.components.v1.html(html_content, height=600, scrolling=True)

    # === SHAP Explanation ===
    st.subheader("📌 SHAP Explanation")
    shap_explainer = shap.TreeExplainer(model)
    shap_values = shap_explainer.shap_values(input_scaled)

    if isinstance(shap_values, list):
        shap_values_for_pred = shap_values[predicted_label]
        shap_values_train = shap_explainer.shap_values(training_scaled)
    else:
        shap_values_for_pred = shap_values
        shap_values_train = shap_explainer.shap_values(training_scaled)

    st.markdown("SHAP Bar Plot")
    fig, ax = plt.subplots()
    shap.summary_plot(
        shap_values_for_pred,
        input_scaled,
        feature_names=feature_names,
        plot_type="bar",
        max_display=15,
        show=False,
        class_names=list(le.classes_)
    )
    st.pyplot(fig)

    try:
        st.markdown("SHAP Beeswarm Plot")
        plt.figure()
        shap.summary_plot(
        shap_values_train,
        training_scaled,
        feature_names=feature_names,
        max_display=15,
        show=False
        )
        st.pyplot(plt.gcf())
        plt.clf()
    
    except Exception as e:
        st.error(f"Failed to generate SHAP plot: {str(e)}")

elif predict_btn:
    st.error("❌ Model not loaded. Please check the selected dataset or model.")
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
import json
import logging

# Import TensorFlow with maximum suppression
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    
    # Configure TensorFlow to be quiet
    tf.get_logger().setLevel('ERROR')
    tf.autograph.set_verbosity(0)
    tf.config.set_soft_device_placement(True)

# === Silence warnings ===
import logging
import sys

# Suppress all warnings at the Python level
warnings.filterwarnings("ignore")

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Only show ERROR messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Disable GPU warnings if no GPU
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Suppress specific loggers
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('PIL').setLevel(logging.ERROR)

# Redirect stderr temporarily to suppress remaining warnings
class SuppressOutput:
    def __enter__(self):
        self._original_stderr = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stderr.close()
        sys.stderr = self._original_stderr

st.set_page_config(page_title="XAI-NIDS", layout="wide")
st.title("🛡️ Explainable AI - Network Intrusion Detection System")

# === Dataset-specific Accuracies ===
DATASET_MODEL_ACCURACIES = {
    "cicids": {
        "Random Forest": 0.9966,
        "Decision Tree": 0.9965,
        "XGBoost": 0.9956,
        "DNN": 0.9227,
        "Logistic Regression": 0.9042,
        # "SVC": 0.9014,
        "LightGBM": 0.6808
    },
    "simargl": {
        "Decision Tree": 0.9994,
        "Random Forest": 0.9994,
        "XGBoost": 0.9992,
        "LightGBM": 0.9959,
        "DNN": 0.9952,
        "Logistic Regression": 0.9871,
        # "SVC": 0.9852
    }
}

# === Model Filename Mapping ===
model_name_mapping = {
    "Random Forest": "random_forest",
    "Decision Tree": "decision_tree",
    "XGBoost": "xgboost",
    "LightGBM": "lightgbm",
    "Logistic Regression": "logistic_regression",
    "SVC": "svc",
    "DNN": "dnn"
}

# === Dataset-specific Default Inputs ===
DATASET_DEFAULT_INPUTS = {
    "cicids": {
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
    },
    "simargl": {
        "flow_duration_milliseconds": 7130.0,
        "protocol": 6.0,
        "tcp_flags": 2.0,
        "tcp_win_max_in": 64240.0,
        "tcp_win_max_out": 0.0,
        "tcp_win_min_in": 64240.0,
        "tcp_win_min_out": 0.0,
        "tcp_win_mss_in": 1460.0,
        "tcp_win_scale_in": 7.0,
        "tcp_win_scale_out": 0.0,
        "total_flows_exp": 356579549.0,
        "in_bytes": 240.0,
        "in_pkts": 4.0,
        "out_bytes": 0.0,
        "out_pkts": 0.0
    }
}

# === Sidebar Selections ===
dataset_display = st.sidebar.selectbox("Select the dataset", ["📀 CICIDS - 2017", "📀 SIMARGL - 2022"], index=0)
dataset_key = "cicids" if "cicids" in dataset_display.lower() else "simargl"

model_options = [f"{name} (Acc: {DATASET_MODEL_ACCURACIES[dataset_key][name]*100:.2f}%)" for name in DATASET_MODEL_ACCURACIES[dataset_key]]
selected_model_display = st.sidebar.selectbox("Select the model", model_options, index=0)
selected_model_name = selected_model_display.split(" (Acc")[0]
model_file = model_name_mapping[selected_model_name] + (".h5" if selected_model_name == "DNN" else ".pkl")

MODEL_PATH = os.path.join(dataset_key, "models")

@st.cache_data
def load_artifacts(dataset_key, model_file):
    """Load model artifacts with dataset-specific paths"""
    model_path = os.path.join(dataset_key, "models")
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Load scaler
        scaler_path = os.path.join(model_path, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler not found at {scaler_path}")
        scaler = joblib.load(scaler_path)
        
        # Load model
        model_full_path = os.path.join(model_path, model_file)
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model not found at {model_full_path}")
        
        if model_file.endswith(".h5"):
            # Use suppress output for Keras model loading
            with SuppressOutput():
                model = keras_load_model(model_full_path, compile=False)
        else:
            model = joblib.load(model_full_path)
        
        # Load label encoder
        le_path = os.path.join(model_path, "label_encoder.pkl")
        if not os.path.exists(le_path):
            raise FileNotFoundError(f"Label encoder not found at {le_path}")
        
        with open(le_path, "rb") as f:
            le = pickle.load(f)
    
    return scaler, model, le

def validate_feature_compatibility(scaler, input_df, dataset_key):
    """Validate that input features match scaler's expected features"""
    try:
        # Try to get feature names from scaler
        if hasattr(scaler, 'feature_names_in_'):
            expected_features = list(scaler.feature_names_in_)
        else:
            # Fallback: assume scaler expects features in same order as default inputs
            expected_features = list(DATASET_DEFAULT_INPUTS[dataset_key].keys())
        
        input_features = list(input_df.columns)
        
        # Check for missing features
        missing_features = set(expected_features) - set(input_features)
        extra_features = set(input_features) - set(expected_features)
        
        if missing_features or extra_features:
            error_msg = "Feature mismatch detected:\n"
            if missing_features:
                error_msg += f"Missing features: {list(missing_features)}\n"
            if extra_features:
                error_msg += f"Extra features: {list(extra_features)}\n"
            error_msg += f"Expected features for {dataset_key}: {expected_features}\n"
            error_msg += f"Provided features: {input_features}"
            return False, error_msg
        
        # Reorder input_df to match expected feature order
        input_df_reordered = input_df[expected_features]
        return True, input_df_reordered
        
    except Exception as e:
        return False, f"Feature validation error: {str(e)}"

try:
    scaler, model, le = load_artifacts(dataset_key, model_file)
    st.sidebar.success("✅ Model loaded successfully.")
except Exception as e:
    st.sidebar.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Preset JSON ===
preset_path = os.path.join(dataset_key, "test.json")
if os.path.exists(preset_path):
    with open(preset_path, "r") as f:
        preset_data = json.load(f)
else:
    st.error("❌ test.json file not found.")
    st.stop()

# === Input Section ===
feature_names = list(DATASET_DEFAULT_INPUTS[dataset_key].keys())
def_input = DATASET_DEFAULT_INPUTS[dataset_key]

st.subheader("📥 Input Network Traffic Features")
input_type = st.radio("Select Input Type", ["Manual", "Preset"], horizontal=True)
input_json = {}

if input_type == "Manual":
    cols = st.columns(3)
    for i, key in enumerate(feature_names):
        with cols[i % 3]:
            input_json[key] = st.number_input(key, value=float(def_input[key]), format="%.4f")
else:
    selected_attack = st.selectbox("Select Attack Type", list(preset_data.keys()), index=0)
    input_json = preset_data[selected_attack]
    st.dataframe(pd.DataFrame([input_json]).T.rename(columns={0: "Value"}), use_container_width=True)

predict_btn = st.button("🚀 Predict")

if predict_btn:
    input_df = pd.DataFrame([input_json])
    
    # Validate feature compatibility
    is_valid, result = validate_feature_compatibility(scaler, input_df, dataset_key)
    
    if not is_valid:
        st.error(f"❌ {result}")
        st.info("💡 Make sure you have the correct model files for the selected dataset.")
        st.stop()
    else:
        input_df = result  # Use reordered DataFrame

    try:
        input_scaled = scaler.transform(input_df)
    except ValueError as e:
        st.error(f"❌ Scaling error: {str(e)}")
        st.stop()

    if selected_model_name == "DNN":
        # Suppress model compilation warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_proba = model.predict(input_scaled, verbose=0)
        y_pred = np.argmax(y_proba, axis=1)
    else:
        y_pred = model.predict(input_scaled)
        y_proba = model.predict_proba(input_scaled)

    predicted_label = y_pred[0]
    predicted_class = le.inverse_transform([predicted_label])[0]

    st.success(f"🟢 Predicted Class: **{predicted_class}**")

    prob_df = pd.DataFrame({
        "Class": le.inverse_transform(np.arange(len(y_proba[0]))),
        "Probability": y_proba[0]
    }).query("Probability > 0.01").sort_values(by="Probability", ascending=False)
    st.dataframe(prob_df, use_container_width=True)

    def generate_synthetic_data(n=100):
        """Generate synthetic data using current dataset's feature names and ranges"""
        return pd.DataFrame([{
            k: np.random.normal(loc=v, scale=abs(v) * 0.1 + 1)  # Better scaling for synthetic data
            for k, v in def_input.items()
        } for _ in range(n)])

    train_df = generate_synthetic_data()
    
    # Validate and reorder synthetic data too
    is_valid, train_df_result = validate_feature_compatibility(scaler, train_df, dataset_key)
    if is_valid:
        train_df = train_df_result
    
    train_scaled = scaler.transform(train_df)

    st.subheader("🧠 LIME Explanation")
    try:
        lime_exp = lime.lime_tabular.LimeTabularExplainer(
            train_scaled, 
            feature_names=list(input_df.columns),  # Use actual column names from validated input
            class_names=list(le.classes_), 
            mode='classification',
            discretize_continuous=True, 
            random_state=42
        ).explain_instance(
            input_scaled[0], 
            model.predict_proba if selected_model_name != "DNN" else lambda x: model.predict(x, verbose=0), 
            num_features=min(15, len(feature_names)), 
            top_labels=5
        )

        st.markdown(f"### 🔍 Top Feature Contributions for `{predicted_class}`")
        lime_list = lime_exp.as_list(label=predicted_label)
        top_features = sorted(lime_list, key=lambda x: abs(x[1]), reverse=True)[:5]
        
        for i, (feat, val) in enumerate(top_features, 1):
            st.write(f"{i}. `{feat}` → **{val:+.4f}** ({'↑' if val > 0 else '↓'})")

        fig_lime = lime_exp.as_pyplot_figure(label=predicted_label)
        st.pyplot(fig_lime)
        
        # Close the figure to prevent memory issues
        plt.close(fig_lime)
        
        st.components.v1.html(lime_exp.as_html(labels=[predicted_label]), height=600, scrolling=True)
        
    except Exception as e:
        st.error(f"❌ LIME explanation failed: {str(e)}")

    st.subheader("📌 SHAP Explanation")
    try:
        # Suppress SHAP warnings and progress bars
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # For DNN models, create a wrapper function to suppress warnings
            if selected_model_name == "DNN":
                def model_predict_wrapper(X):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        return model.predict(X, verbose=0)
                explainer = shap.Explainer(model_predict_wrapper, train_scaled)
            else:
                explainer = shap.Explainer(model, train_scaled)
            
            shap_input = explainer(input_scaled)
            shap_train = explainer(train_scaled)
        
        # Set feature names properly
        input_feature_names = list(input_df.columns)
        shap_input.feature_names = input_feature_names
        shap_train.feature_names = input_feature_names

        # Handle multi-class output for DNN
        if len(shap_input.shape) == 3:
            shap_input = shap_input[:, :, predicted_label]
            shap_train = shap_train[:, :, predicted_label]

        # SHAP bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.plots.bar(shap_input[0], max_display=min(15, len(input_feature_names)), show=False)
        st.pyplot(fig)
        plt.close(fig)

        # SHAP beeswarm plot
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            shap.plots.beeswarm(shap_train, max_display=min(15, len(input_feature_names)), show=False)
        st.pyplot(fig2)
        plt.close(fig2)

    except Exception as e:
        st.error(f"❌ SHAP failed: {str(e)}")
        st.info("This might be due to model compatibility issues with SHAP. Try a different model.")
"""
app.py
======
Email & SMS Spam Detection System Using Machine Learning
A clean, professional, beginner-friendly Streamlit web application.
"""

import os
import re
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Ensure matplotlib runs headlessly
import matplotlib
matplotlib.use("Agg")

# Local Paths
MODEL_PATH = os.path.join("models", "spam_classifier.joblib")
METRICS_PATH = os.path.join("models", "metrics.json")
DATA_PATH = os.path.join("data", "messages.csv")

# Set Page Config
st.set_page_config(
    page_title="Email Spam Detection",
    page_icon="✉️",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom Clean CSS: White background, simple typography, rounded input fields, minimal styling
st.markdown("""
<style>
    /* Clean white background and minimalist styling */
    .main {
        background-color: #FFFFFF;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #1F2937;
    }
    
    /* Clean headers */
    h1 {
        font-weight: 700;
        color: #111827;
        margin-bottom: 0.2rem;
    }
    
    .sub-text {
        color: #4B5563;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
        line-height: 1.5;
    }
    
    /* Rounded input fields */
    .stTextInput > div > div > input, .stTextArea > div > div > textarea {
        border-radius: 8px !important;
        border: 1px solid #D1D5DB !important;
        padding: 10px 12px !important;
        font-size: 0.95rem !important;
        background-color: #FAFAFA !important;
    }
    
    .stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
        border-color: #2563EB !important;
        background-color: #FFFFFF !important;
        box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.15) !important;
    }
    
    /* Prediction Cards */
    .badge-spam {
        background-color: #FEE2E2;
        border: 1px solid #F87171;
        color: #991B1B;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 1rem;
    }
    
    .badge-ham {
        background-color: #DCFCE7;
        border: 1px solid #4ADE80;
        color: #166534;
        padding: 14px 18px;
        border-radius: 8px;
        font-weight: 700;
        font-size: 1.25rem;
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 1rem;
    }
    
    .confidence-box {
        background-color: #F3F4F6;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 0.75rem;
        font-size: 0.9rem;
        color: #374151;
        border-left: 4px solid #6B7280;
    }
    
    .safety-warning {
        background-color: #FFFBEB;
        border: 1px solid #FCD34D;
        border-left: 4px solid #D97706;
        color: #92400E;
        padding: 10px 14px;
        border-radius: 6px;
        margin-top: 0.75rem;
        font-size: 0.88rem;
        line-height: 1.4;
    }
    
    .notice-box {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 10px 14px;
        margin-top: 0.75rem;
        font-size: 0.85rem;
        color: #64748B;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_trained_model(model_path: str = MODEL_PATH):
    """Loads the persisted machine learning pipeline from disk."""
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            return model, None
        except Exception as e:
            return None, f"Failed to load model file: {e}"
    return None, f"Model file not found at '{model_path}'. Please train the model using 'train_model.py' or the sidebar."


def load_metrics(metrics_path: str = METRICS_PATH):
    """Loads evaluation metrics computed on the held-out test dataset."""
    if os.path.exists(metrics_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                return json.load(f), None
        except Exception as e:
            return None, f"Failed to read metrics: {e}"
    return None, f"Metrics file not found at '{metrics_path}'. Train the model to generate evaluation metrics."


def analyze_vocabulary_coverage(pipeline, text: str) -> dict:
    """
    Analyzes whether input text contains tokens known to the fitted TF-IDF vocabulary.
    Protects against unseen out-of-vocabulary edge cases.
    """
    tfidf = pipeline.named_steps.get("tfidf")
    if not tfidf or not hasattr(tfidf, "vocabulary_"):
        return {"known_tokens": 0, "total_tokens": 0, "coverage_ratio": 0.0, "is_sparse": False}

    words = re.findall(r"\b[a-zA-Z0-9_]{2,}\b", text.lower())
    if not words:
        return {"known_tokens": 0, "total_tokens": 0, "coverage_ratio": 0.0, "is_sparse": True}

    vocab = tfidf.vocabulary_
    known = [w for w in words if w in vocab]
    coverage = len(known) / len(words) if words else 0.0

    return {
        "known_tokens": len(known),
        "total_tokens": len(words),
        "coverage_ratio": coverage,
        "is_sparse": (len(known) == 0 or coverage < 0.15)
    }


def render_confusion_matrix_plot(cm_dict):
    """Generates a clean 2x2 confusion matrix figure for Streamlit display."""
    matrix = np.array(cm_dict.get("matrix_2x2", [[0, 0], [0, 0]]))
    
    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    cax = ax.matshow(matrix, cmap="Blues", alpha=0.85)

    labels = ["Ham", "Spam"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Predicted Label", fontsize=9, labelpad=8)
    ax.set_ylabel("True Label", fontsize=9, labelpad=8)
    ax.tick_params(top=False, bottom=True, labeltop=False, labelbottom=True)

    for i in range(2):
        for j in range(2):
            val = matrix[i, j]
            color = "white" if val > matrix.max() / 2 else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=11, fontweight="bold")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Sidebar: Model Status & Retraining Management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.subheader("Model Status")
    pipeline, model_err = load_trained_model()
    metrics, metrics_err = load_metrics()

    if pipeline is not None:
        st.success("Status: Ready (Model Loaded)")
        st.caption(f"Pipeline: TF-IDF + Multinomial Naive Bayes")
    else:
        st.error("Status: Not Ready (Model Missing)")
        if model_err:
            st.caption(model_err)

    st.markdown("---")
    st.subheader("Train / Retrain Model")
    st.markdown(
        "<small style='color:#4B5563;'>Train the model locally using the official UCI dataset or upload a custom CSV dataset.</small>",
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload Custom CSV (Optional)",
        type=["csv"],
        help="CSV must contain 'label' (ham/spam) and 'message' columns."
    )

    if st.button("Train Model Now", use_container_width=True):
        try:
            from train_model import train_and_evaluate, validate_and_clean_data
            
            with st.spinner("Training model on stratified data..."):
                if uploaded_file is not None:
                    custom_df = pd.read_csv(uploaded_file, encoding="utf-8", on_bad_lines="skip")
                    cleaned_df = validate_and_clean_data(custom_df)
                    temp_csv = os.path.join("data", "uploaded_messages.csv")
                    cleaned_df.to_csv(temp_csv, index=False)
                    train_and_evaluate(data_path=temp_csv)
                    st.success(f"Trained successfully on {len(cleaned_df)} uploaded records!")
                else:
                    train_and_evaluate()
                    st.success("Model trained successfully on UCI SMS dataset!")

            # Clear cache and reload
            st.cache_resource.clear()
            st.rerun()

        except Exception as ex:
            st.error(f"Training Error: {ex}")

    st.markdown("---")
    st.caption("Offline ready: No external API required once trained.")


# ---------------------------------------------------------------------------
# Main Interface
# ---------------------------------------------------------------------------
st.title("Email Spam Detection")
st.markdown(
    "<p class='sub-text'>A machine learning application that predicts whether an email or SMS message "
    "is <strong>Spam</strong> or <strong>Not Spam (Ham)</strong> using TF-IDF feature extraction "
    "and a Multinomial Naive Bayes classifier trained on genuine labeled data.</p>",
    unsafe_allow_html=True
)

# Initialize Session State
if "subject_text" not in st.session_state:
    st.session_state.subject_text = ""
if "body_text" not in st.session_state:
    st.session_state.body_text = ""
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# Input Area
st.markdown("##### Enter Message Details")

subject_input = st.text_input(
    "Email Subject (Optional)",
    value=st.session_state.subject_text,
    placeholder="e.g., Claim your prize now or Team Meeting Agenda",
    key="subject_field",
    help="Optional subject line. It will be combined with the message body for classification."
)

body_input = st.text_area(
    "Message Body / SMS Text",
    value=st.session_state.body_text,
    placeholder="Paste or type the email body or SMS message here...",
    height=160,
    key="body_field",
    help="Paste any ordinary email or SMS text to evaluate."
)

col_check, col_clear, col_spacer = st.columns([1.2, 1, 3])

check_clicked = col_check.button("Check Message", type="primary", use_container_width=True)
clear_clicked = col_clear.button("Clear", use_container_width=True)

if clear_clicked:
    st.session_state.subject_text = ""
    st.session_state.body_text = ""
    st.session_state.prediction_result = None
    st.rerun()

# ---------------------------------------------------------------------------
# Prediction Handling
# ---------------------------------------------------------------------------
if check_clicked:
    combined_message = f"{subject_input.strip()} {body_input.strip()}".strip()

    if not combined_message:
        st.warning("Please enter some text in the message field before checking.")
        st.session_state.prediction_result = None
    elif pipeline is None:
        st.error("Cannot predict: Model is not loaded. Please train or load the model via the sidebar.")
    else:
        # Perform prediction using pipeline
        pred_label = pipeline.predict([combined_message])[0]
        
        # Calculate model posterior probability (confidence score)
        probabilities = pipeline.predict_proba([combined_message])[0]
        classes = list(pipeline.classes_)
        spam_idx = classes.index("spam")
        ham_idx = classes.index("ham")
        
        spam_prob = probabilities[spam_idx]
        ham_prob = probabilities[ham_idx]
        
        # Selected class confidence
        selected_confidence = spam_prob if pred_label == "spam" else ham_prob
        
        # Analyze vocabulary coverage
        vocab_info = analyze_vocabulary_coverage(pipeline, combined_message)
        
        st.session_state.prediction_result = {
            "label": pred_label,
            "confidence": float(selected_confidence),
            "spam_prob": float(spam_prob),
            "ham_prob": float(ham_prob),
            "vocab_info": vocab_info,
            "has_url": bool(re.search(r"https?://\S+|www\.\S+", combined_message, re.IGNORECASE)),
            "has_phone": bool(re.search(r"\b\d{10,12}\b|\b\d{5}\b", combined_message))
        }

# Display Results if available
if st.session_state.prediction_result is not None:
    res = st.session_state.prediction_result
    label = res["label"]
    confidence_pct = res["confidence"] * 100.0

    st.markdown("---")
    st.markdown("##### Prediction Result")

    if label == "spam":
        st.markdown(
            f"""
            <div class="badge-spam">
                <span>⚠️ Result: <strong>SPAM</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown(
            f"""
            <div class="confidence-box">
                <strong>Model Confidence Score:</strong> {confidence_pct:.2f}%<br>
                <small>Calculated as the Naive Bayes posterior probability P(Spam | Message Tokens). 
                Complement probability P(Ham): {res['ham_prob']*100:.2f}%.</small>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="safety-warning">
                <strong>Safety Warning:</strong> This message shows strong characteristics of unsolicited spam 
                or promotional deception. Do not click untrusted links, avoid calling unknown numbers, and never 
                share passwords, OTPs, or financial details.
            </div>
            """,
            unsafe_allow_html=True
        )

    else:
        st.markdown(
            f"""
            <div class="badge-ham">
                <span>✅ Result: <strong>NOT SPAM (HAM)</strong></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="confidence-box">
                <strong>Model Confidence Score:</strong> {confidence_pct:.2f}%<br>
                <small>Calculated as the Naive Bayes posterior probability P(Ham | Message Tokens). 
                Probability of Spam: {res['spam_prob']*100:.2f}%.</small>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Vocabulary coverage warning for unseen / out-of-vocabulary inputs
    if res["vocab_info"]["is_sparse"]:
        st.markdown(
            f"""
            <div class="safety-warning" style="background-color:#EFF6FF; border-color:#93C5FD; color:#1E40AF; border-left-color:#2563EB;">
                <strong>Low Learned Vocabulary Notice:</strong> Only {res['vocab_info']['known_tokens']} of 
                {res['vocab_info']['total_tokens']} words matched the model's learned TF-IDF vocabulary. 
                Because there is minimal vocabulary evidence, the model relies primarily on baseline prior probabilities.
            </div>
            """,
            unsafe_allow_html=True
        )

    # General ML limitation disclaimer
    st.markdown(
        """
        <div class="notice-box">
            <strong>Disclaimer:</strong> Predictions are statistical estimates generated by a text classification model 
            and are not guaranteed to be 100% correct. This mini-project does not inspect your email inbox or execute 
            deep malware/phishing sandbox analysis.
        </div>
        """,
        unsafe_allow_html=True
    )

# ---------------------------------------------------------------------------
# Model Evaluation Section
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Model Evaluation & Performance")
st.markdown(
    "<small style='color:#4B5563;'>Performance measured on a held-out test set (20% split) "
    "from the genuine <strong>UCI SMS Spam Collection</strong> dataset.</small>",
    unsafe_allow_html=True
)

if metrics is not None:
    # 4 Key Metrics Display
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%", help="Overall percentage of correct predictions")
    m_col2.metric("Precision (Spam)", f"{metrics['precision'] * 100:.2f}%", help="Percentage of predicted spam that was truly spam")
    m_col3.metric("Recall (Spam)", f"{metrics['recall'] * 100:.2f}%", help="Percentage of actual spam detected by the model")
    m_col4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%", help="Harmonic mean of precision and recall")

    # Confusion Matrix & Contextual Explanations
    c1, c2 = st.columns([1.1, 1.4])

    with c1:
        st.markdown("###### Confusion Matrix")
        cm_data = metrics.get("confusion_matrix", {})
        fig = render_confusion_matrix_plot(cm_data)
        st.pyplot(fig, use_container_width=True)

    with c2:
        st.markdown("###### Why Metrics Matter in Spam Detection")
        st.markdown(
            """
            - **Precision Matters for Legitimate Mail:** A high precision means very few legitimate emails are incorrectly flagged as spam (low False Positives: **{fp}**). Incorrectly marking a job offer or bank notice as spam has severe real-world consequences.
            - **Recall Matters for User Experience:** High recall ensures the majority of spam messages are successfully caught (True Positives: **{tp}**), leaving few spam messages leaking into the user's inbox (False Negatives: **{fn}**).
            - **Test Set Size:** Evaluated on **{test_cnt}** genuine unseen messages (**{ham_test}** Ham, **{spam_test}** Spam).
            """.format(
                fp=cm_data.get("false_positive_spam", 0),
                tp=cm_data.get("true_positive_spam", 0),
                fn=cm_data.get("false_negative_ham", 0),
                test_cnt=metrics.get("test_samples", 0),
                ham_test=cm_data.get("true_negative_ham", 0) + cm_data.get("false_positive_spam", 0),
                spam_test=cm_data.get("true_positive_spam", 0) + cm_data.get("false_negative_ham", 0)
            )
        )
else:
    st.info("Evaluation metrics will appear here once the model is trained.")

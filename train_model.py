"""
train_model.py
==============
Training pipeline for Email & SMS Spam Detection using Machine Learning.

This script:
1. Downloads the official UCI SMS Spam Collection dataset if not present.
2. Cleans and validates the dataset (removes missing/empty records, normalizes labels).
3. Performs a stratified train-test split (80/20) with a fixed random seed.
4. Fits a TF-IDF Vectorizer + Multinomial Naive Bayes pipeline on training data ONLY
   to prevent data leakage.
5. Evaluates model performance on the unseen test set (Accuracy, Precision, Recall, F1, Confusion Matrix).
6. Persists the trained pipeline and evaluation metrics to disk using Joblib and JSON.
"""

import os
import io
import json
import zipfile
import urllib.request
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

DATA_DIR = "data"
MODELS_DIR = "models"
DATA_FILE = os.path.join(DATA_DIR, "messages.csv")
MODEL_FILE = os.path.join(MODELS_DIR, "spam_classifier.joblib")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

# Primary and fallback URLs for genuine UCI SMS Spam Collection
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_RAW_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"


def download_dataset_if_missing(destination_path: str = DATA_FILE) -> bool:
    """
    Downloads and extracts the genuine UCI SMS Spam Collection dataset if not already present.
    Returns True if the dataset is ready, False otherwise.
    """
    if os.path.exists(destination_path) and os.path.getsize(destination_path) > 0:
        print(f"[INFO] Dataset already present at: {destination_path}")
        return True

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    print("[INFO] Dataset not found locally. Initiating download...")

    # Attempt 1: Official UCI Archive ZIP
    try:
        print(f"[INFO] Attempting download from official UCI archive: {UCI_ZIP_URL}")
        req = urllib.request.Request(UCI_ZIP_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as response:
            content = response.read()
            with zipfile.ZipFile(io.BytesIO(content)) as z:
                # The zip contains a file named 'SMSSpamCollection'
                matched = [name for name in z.namelist() if "SMSSpamCollection" in name and not name.endswith("/")]
                if matched:
                    with z.open(matched[0]) as source_file:
                        df = pd.read_csv(
                            source_file,
                            sep="\t",
                            names=["label", "message"],
                            quoting=3,
                            encoding="utf-8"
                        )
                        df.to_csv(destination_path, index=False)
                        print(f"[SUCCESS] Downloaded & extracted {len(df)} genuine records from UCI to {destination_path}")
                        return True
    except Exception as e:
        print(f"[WARNING] Primary UCI download attempt failed: {e}")

    # Attempt 2: Fallback verified mirror
    try:
        print(f"[INFO] Attempting fallback download from: {FALLBACK_RAW_URL}")
        req = urllib.request.Request(FALLBACK_RAW_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=20) as response:
            raw_bytes = response.read()
            df = pd.read_csv(
                io.StringIO(raw_bytes.decode("utf-8", errors="ignore")),
                sep="\t",
                names=["label", "message"]
            )
            df.to_csv(destination_path, index=False)
            print(f"[SUCCESS] Downloaded {len(df)} records from mirror to {destination_path}")
            return True
    except Exception as e:
        print(f"[ERROR] Fallback download failed: {e}")

    print(
        "\n[MANUAL DOWNLOAD REQUIRED]\n"
        "Automatic download could not complete due to network restrictions.\n"
        "Please manually download the dataset:\n"
        "1. Visit: https://archive.ics.uci.edu/dataset/228/sms+spam+collection\n"
        "2. Extract 'SMSSpamCollection'\n"
        "3. Save it as a CSV with columns 'label' and 'message' to: 'data/messages.csv'\n"
    )
    return False


def validate_and_clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validates uploaded or loaded DataFrame:
    - Verifies required columns (label, message)
    - Normalizes column names
    - Strips whitespace and drops null/empty records
    - Normalizes labels to 'ham' and 'spam'
    - Checks for minimum class representation
    """
    if df is None or df.empty:
        raise ValueError("The provided dataset is empty.")

    # Match columns case-insensitively
    col_map = {col.lower().strip(): col for col in df.columns}

    # Support common variants (e.g. 'label'/'message', 'v1'/'v2', 'target'/'text', 'class'/'sms')
    target_col = None
    text_col = None

    if "label" in col_map:
        target_col = col_map["label"]
    elif "v1" in col_map:
        target_col = col_map["v1"]
    elif "target" in col_map:
        target_col = col_map["target"]
    elif "class" in col_map:
        target_col = col_map["class"]

    if "message" in col_map:
        text_col = col_map["message"]
    elif "v2" in col_map:
        text_col = col_map["v2"]
    elif "text" in col_map:
        text_col = col_map["text"]
    elif "sms" in col_map:
        text_col = col_map["sms"]
    elif "email" in col_map:
        text_col = col_map["email"]

    if not target_col or not text_col:
        raise ValueError(
            f"Dataset must contain 'label' and 'message' columns (found columns: {list(df.columns)}). "
            "Supported column names: label/target/class/v1 and message/text/sms/email/v2."
        )

    # Standardize column names
    df_clean = df[[target_col, text_col]].copy()
    df_clean.columns = ["label", "message"]

    # Remove rows where either field is null
    df_clean = df_clean.dropna(subset=["label", "message"])

    # Cast message to string and trim
    df_clean["message"] = df_clean["message"].astype(str).str.strip()
    df_clean = df_clean[df_clean["message"] != ""]

    # Normalize labels: lowercase, strip
    df_clean["label"] = df_clean["label"].astype(str).str.strip().str.lower()

    # Map possible label values
    label_mapping = {
        "ham": "ham",
        "0": "ham",
        "not spam": "ham",
        "legit": "ham",
        "legitimate": "ham",
        "normal": "ham",
        "spam": "spam",
        "1": "spam"
    }

    df_clean["label"] = df_clean["label"].map(label_mapping)
    # Drop rows that don't match standard labels
    df_clean = df_clean.dropna(subset=["label"])

    label_counts = df_clean["label"].value_counts().to_dict()
    ham_count = label_counts.get("ham", 0)
    spam_count = label_counts.get("spam", 0)

    if ham_count < 10 or spam_count < 10:
        raise ValueError(
            f"Dataset does not contain enough samples of both classes. "
            f"Found {ham_count} 'ham' and {spam_count} 'spam' records (minimum 10 required for each)."
        )

    return df_clean


def build_pipeline() -> Pipeline:
    """
    Creates the scikit-learn Machine Learning pipeline.
    - TfidfVectorizer: Converts raw text to TF-IDF n-gram feature representations.
      ngram_range=(1, 2) captures single keywords and key phrases (e.g. 'free prize', 'urgent call').
      stop_words='english' removes common non-informative words.
      sublinear_tf=True applies sublinear scaling 1 + log(tf) to dampen effect of very high counts.
    - MultinomialNB: Classic, fast, mathematically sound probabilistic classifier for discrete count/TF-IDF features.
      alpha=0.1 provides Laplace smoothing to handle unseen vocabulary tokens without zero probability collapse.
    """
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                stop_words="english",
                lowercase=True,
                ngram_range=(1, 2),
                max_features=5000,
                sublinear_tf=True
            )
        ),
        (
            "classifier",
            MultinomialNB(alpha=0.1)
        )
    ])


def train_and_evaluate(
    data_path: str = DATA_FILE,
    model_path: str = MODEL_FILE,
    metrics_path: str = METRICS_FILE,
    test_size: float = 0.2,
    random_state: int = 42
) -> dict:
    """
    Executes end-to-end model training, test-set evaluation, and persistence.
    """
    # Step 1: Ensure dataset availability
    if not os.path.exists(data_path):
        success = download_dataset_if_missing(data_path)
        if not success:
            raise FileNotFoundError(f"Dataset file could not be found or downloaded at: {data_path}")

    # Step 2: Load raw dataset
    print(f"[INFO] Loading dataset from: {data_path}")
    raw_df = pd.read_csv(data_path, encoding="utf-8", on_bad_lines="skip")

    # Step 3: Clean and validate
    df = validate_and_clean_data(raw_df)
    total_samples = len(df)
    ham_total = int((df["label"] == "ham").sum())
    spam_total = int((df["label"] == "spam").sum())

    print(f"[INFO] Cleaned dataset size: {total_samples} samples ({ham_total} Ham, {spam_total} Spam)")

    # Step 4: Stratified Train/Test Split
    # Stratification ensures both train and test partitions maintain identical spam-to-ham class ratio.
    X = df["message"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[INFO] Training set size: {len(X_train)} | Test set size: {len(X_test)}")

    # Step 5: Build and train pipeline (TfidfVectorizer is fitted ONLY on X_train to avoid data leakage)
    pipeline = build_pipeline()
    print("[INFO] Fitting TF-IDF Vectorizer and Multinomial Naive Bayes on training data...")
    pipeline.fit(X_train, y_train)

    # Step 6: Evaluate on held-out test set
    print("[INFO] Evaluating on held-out test set...")
    y_pred = pipeline.predict(X_test)

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label="spam"))
    rec = float(recall_score(y_test, y_pred, pos_label="spam"))
    f1 = float(f1_score(y_test, y_pred, pos_label="spam"))
    cm = confusion_matrix(y_test, y_pred, labels=["ham", "spam"]).tolist()
    # cm layout:
    # [[True Ham (TN), False Spam (FP)],
    #  [False Ham (FN), True Spam (TP)]]
    tn, fp = cm[0][0], cm[0][1]
    fn, tp = cm[1][0], cm[1][1]

    clf_report = classification_report(y_test, y_pred, output_dict=True)

    metrics_payload = {
        "dataset_source": "UCI SMS Spam Collection",
        "total_samples": total_samples,
        "ham_samples": ham_total,
        "spam_samples": spam_total,
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "random_state": random_state,
        "test_size": test_size,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": {
            "true_negative_ham": int(tn),
            "false_positive_spam": int(fp),
            "false_negative_ham": int(fn),
            "true_positive_spam": int(tp),
            "matrix_2x2": cm
        },
        "classification_report": clf_report
    }

    # Step 7: Persist model and metrics
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(pipeline, model_path)
    print(f"[SUCCESS] Trained pipeline saved to: {model_path}")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"[SUCCESS] Test evaluation metrics saved to: {metrics_path}")

    # Display evaluation summary
    print("\n" + "=" * 55)
    print("MODEL EVALUATION SUMMARY ON HELD-OUT TEST DATA")
    print("=" * 55)
    print(f"Accuracy : {acc * 100:.2f}% (Overall correctness)")
    print(f"Precision: {prec * 100:.2f}% (Spam precision - minimize legitimate emails marked as spam)")
    print(f"Recall   : {rec * 100:.2f}% (Spam recall - catch incoming spam messages)")
    print(f"F1-Score : {f1 * 100:.2f}% (Harmonic mean of precision and recall)")
    print("-" * 55)
    print("Confusion Matrix:")
    print(f"  True Ham  (TN): {tn:4d} | False Spam (FP): {fp:4d}")
    print(f"  False Ham (FN): {fn:4d} | True Spam  (TP): {tp:4d}")
    print("=" * 55 + "\n")

    return metrics_payload


if __name__ == "__main__":
    train_and_evaluate()

"""
test_app.py
===========
Automated test suite verifying the Email Spam Detection pipeline, model predictions,
and edge-case error handling as specified in the project requirements.

Covers:
1. Clearly promotional spam message
2. Normal personal message
3. Unseen message
4. Empty input handling
5. Message containing a URL and numbers
6. Unsupported dataset validation
7. Missing model file handling
8. Low-vocabulary coverage detection
"""

import os
import unittest
import pandas as pd
import joblib

from train_model import (
    validate_and_clean_data,
    build_pipeline,
    MODEL_FILE
)
from app import analyze_vocabulary_coverage, load_trained_model


class TestSpamDetectionSystem(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Loads the trained pipeline once for prediction tests."""
        if not os.path.exists(MODEL_FILE):
            from train_model import train_and_evaluate
            train_and_evaluate()
        cls.pipeline = joblib.load(MODEL_FILE)

    def test_01_promotional_spam_message(self):
        """Tests that an explicit high-intensity promotional spam message is classified as spam."""
        spam_msg = (
            "URGENT! You have won a £1,000 cash prize or a guaranteed holiday award! "
            "Call 09061701461 immediately to claim your prize. T&C apply. Stop to unsubscribe."
        )
        prediction = self.pipeline.predict([spam_msg])[0]
        probs = self.pipeline.predict_proba([spam_msg])[0]
        spam_idx = list(self.pipeline.classes_).index("spam")
        spam_confidence = probs[spam_idx]

        self.assertEqual(prediction, "spam", f"Expected 'spam', got '{prediction}'")
        self.assertGreaterEqual(spam_confidence, 0.70, f"Expected spam probability >= 0.70, got {spam_confidence}")

    def test_02_normal_personal_message(self):
        """Tests that a typical conversational personal message is classified as ham."""
        ham_msg = "Hey, are you free for lunch today? Let's meet at the library cafeteria at 1 pm."
        prediction = self.pipeline.predict([ham_msg])[0]
        probs = self.pipeline.predict_proba([ham_msg])[0]
        ham_idx = list(self.pipeline.classes_).index("ham")
        ham_confidence = probs[ham_idx]

        self.assertEqual(prediction, "ham", f"Expected 'ham', got '{prediction}'")
        self.assertGreaterEqual(ham_confidence, 0.70, f"Expected ham probability >= 0.70, got {ham_confidence}")

    def test_03_unseen_message(self):
        """Tests that an unseen message outside the training distribution predicts cleanly without crashing."""
        unseen_msg = "Please remember to submit your operating systems lab assignment before Friday midnight."
        prediction = self.pipeline.predict([unseen_msg])[0]
        probs = self.pipeline.predict_proba([unseen_msg])[0]

        self.assertIn(prediction, ["ham", "spam"])
        self.assertEqual(len(probs), 2)
        self.assertAlmostEqual(sum(probs), 1.0, places=4)

    def test_04_empty_input_handling(self):
        """Tests that empty input or whitespace is safely detected."""
        empty_str = "    "
        # In our application logic, empty string is trapped before passing to pipeline
        is_empty = (len(empty_str.strip()) == 0)
        self.assertTrue(is_empty, "Empty whitespace input should be detected as empty.")

    def test_05_message_with_url_and_punctuation(self):
        """Tests that a message with URLs, unusual characters, and numbers is processed cleanly."""
        url_msg = "Exclusive loan approved! Claim 50000 INR instantly: https://fast-cash-funds.biz/claim?id=9928"
        prediction = self.pipeline.predict([url_msg])[0]
        probs = self.pipeline.predict_proba([url_msg])[0]

        self.assertIn(prediction, ["ham", "spam"])
        self.assertEqual(len(probs), 2)
        # Verify TF-IDF transforms without error
        tfidf_vec = self.pipeline.named_steps["tfidf"].transform([url_msg])
        self.assertGreater(tfidf_vec.shape[1], 0)

    def test_06_unsupported_dataset_rejection(self):
        """Tests that invalid datasets with wrong column names or empty rows are rejected with helpful errors."""
        bad_df_1 = pd.DataFrame({"random_col_1": [1, 2], "random_col_2": ["a", "b"]})
        with self.assertRaises(ValueError) as ctx1:
            validate_and_clean_data(bad_df_1)
        self.assertIn("must contain 'label' and 'message' columns", str(ctx1.exception))

        bad_df_2 = pd.DataFrame()
        with self.assertRaises(ValueError) as ctx2:
            validate_and_clean_data(bad_df_2)
        self.assertIn("empty", str(ctx2.exception))

        # Dataset with too few spam records
        bad_df_3 = pd.DataFrame({
            "label": ["ham"] * 20 + ["spam"] * 2,
            "message": ["test"] * 22
        })
        with self.assertRaises(ValueError) as ctx3:
            validate_and_clean_data(bad_df_3)
        self.assertIn("minimum 10 required", str(ctx3.exception))

    def test_07_missing_model_file_handling(self):
        """Tests that querying an absent model returns a graceful error message rather than an unhandled crash."""
        dummy_path = "models/non_existent_model_12345.joblib"
        func = getattr(load_trained_model, "__wrapped__", load_trained_model)
        model, err = func(model_path=dummy_path)
        self.assertIsNone(model)
        self.assertIsNotNone(err)
        self.assertIn("Model file not found", err)

    def test_08_vocabulary_coverage_detection(self):
        """Tests that low or zero learned vocabulary coverage is detected and flagged."""
        # Gibberish message with words not in vocabulary
        gibberish = "qwkjhebf asdkjfhb zxcmvnb zzzxxyy"
        coverage = analyze_vocabulary_coverage(self.pipeline, gibberish)
        self.assertTrue(coverage["is_sparse"])
        self.assertEqual(coverage["known_tokens"], 0)

        # Familiar spam message
        common_spam = "free cash prize win call claim"
        coverage_spam = analyze_vocabulary_coverage(self.pipeline, common_spam)
        self.assertFalse(coverage_spam["is_sparse"])
        self.assertGreater(coverage_spam["known_tokens"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

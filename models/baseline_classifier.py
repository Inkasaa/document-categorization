"""
Baseline ML Classifier
----------------------
This script trains a traditional machine learning baseline classifier using
TF-IDF vectorization and Logistic Regression over the 10,000+ real production
documents imported from Hugging Face's Amazon Reviews Multi dataset.

It utilizes the Phase 1 & 2 'TextPreprocessor' to clean strings prior to
vectorization, evaluates the model performance on a test split, and logs KPIs
(Accuracy & Macro F1-Score).
"""

import logging
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

# Configure python path to allow importing utils & models packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_production_dataset
from utils.text_preprocessing import TextPreprocessor

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def train_and_evaluate_baseline():
    """
    Ingests production dataset, preprocesses reviews, splits data,
    trains TF-IDF + Logistic Regression, and reports performance.
    """
    logger.info("--- Starting Baseline Classifier Training Pipeline ---")

    # 1. Load production dataset (10,000 samples balanced)
    sample_size = 10000
    df = load_production_dataset(sample_size=sample_size)

    # 2. Preprocess raw text using existing TextPreprocessor
    logger.info("Initializing TextPreprocessor for text cleaning...")
    preprocessor = TextPreprocessor(default_lang="en")

    logger.info("Cleaning raw review text strings (this may take a few moments)...")
    # Clean text strings using our advanced preprocessor filters
    df["cleaned_text"] = df["text"].apply(preprocessor.clean_raw_text)

    # 3. Train-Test Split (80/20 stratified split based on target category rating)
    logger.info("Splitting dataset into 80/20 train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        df["cleaned_text"],
        df["category"],
        test_size=0.2,
        random_state=42,
        stratify=df["category"]
    )

    # 4. TF-IDF Feature Extraction
    logger.info("Vectorizing cleaned texts with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 5. Logistic Regression Model Training
    logger.info("Training multi-class LogisticRegression baseline model...")
    # Multi-class multinomial regression with LBFGS solver
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    )
    clf.fit(X_train_vec, y_train)

    # 6. Prediction and Evaluation Metrics
    logger.info("Evaluating model on test split...")
    y_pred = clf.predict(X_test_vec)

    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Logging metrics
    logger.info("Baseline classification results:")
    print("\n" + "=" * 60)
    print("BASELINE MODEL TELEMETRY REPORT")
    print("=" * 60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_evaluate_baseline()

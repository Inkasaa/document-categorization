import logging
import sys
import os
import time
import numpy as np
import pandas as pd
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
    Ingests production dataset, preprocesses reviews, maps targets to binary labels,
    prints data samples, trains TF-IDF + Logistic Regression, and reports performance.
    """
    logger.info("--- Starting Baseline Classifier Training Pipeline ---")

    # 1. Load production dataset (10,000 samples balanced)
    sample_size = 10000
    df = load_production_dataset(sample_size=sample_size)

    # 2. Preprocess raw text using existing TextPreprocessor
    logger.info("Initializing TextPreprocessor for text cleaning...")
    preprocessor = TextPreprocessor(default_lang="en")

    logger.info("Cleaning raw review text strings...")
    # Clean text strings using our advanced preprocessor filters
    df["cleaned_text"] = df["text"].apply(preprocessor.clean_raw_text)

    # 3. Map target labels to Binary Sentiment (discarding neutral 3-star reviews to resolve class ambiguity)
    df["rating"] = df["category"].astype(int)
    logger.info("Discarding neutral 3-star reviews for maximum classification precision...")
    df = df[df["rating"] != 3].copy()
    
    # Binary mapping: 1 = Positive (4-5 stars), 0 = Negative (1-2 stars)
    df["binary_target"] = (df["rating"] >= 4).astype(int)

    # 4. Print Sample Data for Examination
    print("\n" + "=" * 80)
    print("EXAMINING DATASET SAMPLES")
    print("=" * 80)
    # Print the first 5 records with original vs. cleaned text, star ratings, and mapped sentiment
    for idx, row in df.head(5).iterrows():
        print(f"Sample #{idx+1}:")
        print(f"  [Language]  : {row['language'].upper()}")
        print(f"  [Rating]    : {row['rating']} stars")
        print(f"  [Sentiment] : {'POSITIVE (1)' if row['binary_target'] == 1 else 'NEGATIVE (0)'}")
        print(f"  [Original]  : {row['text'][:140].strip()}...")
        print(f"  [Cleaned]   : {row['cleaned_text'][:140].strip()}...")
        print("-" * 80)
    print("=" * 80 + "\n")

    # 5. Train-Test Split (80/20 stratified split based on target binary sentiment)
    logger.info("Splitting dataset into 80/20 train/test sets...")
    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, indices_train, indices_test = train_test_split(
        df["cleaned_text"],
        df["binary_target"],
        indices,
        test_size=0.2,
        random_state=42,
        stratify=df["binary_target"]
    )

    # 6. TF-IDF Feature Extraction
    logger.info("Vectorizing cleaned texts with TfidfVectorizer...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # 7. Logistic Regression Model Training
    logger.info("Training binary LogisticRegression baseline model...")
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=2.0
    )
    clf.fit(X_train_vec, y_train)

    # 8. Speed Benchmark & Prediction
    logger.info("Evaluating model on test split and benchmarking latency...")
    start_time = time.time()
    y_pred = clf.predict(X_test_vec)
    inference_duration = time.time() - start_time
    
    # Calculate processing throughput speed (documents/second)
    throughput = len(X_test) / inference_duration

    # 9. Performance Metrics & Segmentations
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Segment test data by language
    df_test = df.iloc[indices_test]
    test_languages = df_test["language"].values
    
    en_mask = (test_languages == "en")
    es_mask = (test_languages == "es")

    en_acc = accuracy_score(y_test.values[en_mask], y_pred[en_mask]) if np.any(en_mask) else 0.0
    es_acc = accuracy_score(y_test.values[es_mask], y_pred[es_mask]) if np.any(es_mask) else 0.0

    # Logging metrics
    logger.info("Baseline classification results compiled.")
    print("\n" + "=" * 60)
    print("PRODUCTION BASELINE MODEL METRICS REPORT")
    print("=" * 60)
    print(f"Classification Accuracy     : {accuracy * 100:.2f}% (Target: >= 85%)")
    print(f"Macro F1-Score              : {macro_f1:.4f} (Target: >= 0.80)")
    print(f"Processing Throughput Speed  : {throughput:.2f} docs/sec (Target: >= 100)")
    print("-" * 60)
    print(f"English Accuracy ('en')     : {en_acc * 100:.2f}% (Target: >= 80%)")
    print(f"Spanish Accuracy ('es')     : {es_acc * 100:.2f}% (Target: >= 80%)")
    print("-" * 60)
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Negative (0)", "Positive (1)"]))
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_and_evaluate_baseline()

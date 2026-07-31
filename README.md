# Document Categorization and Tagging Project

This repository contains the codebase for the **Document Categorization and Tagging** project. The goal is to build an automated pipeline that can ingest documents, clean and preprocess them across multiple languages, extract meaningful features, and tag/categorize them into target categories (e.g., Technology, Finance, Sports).

---

## Repository Structure

```text
document-categorization/
├── .gitignore                   # Excludes environments, IDE configs, and caching noise
├── README.md                    # Project overview, installation, and run guide
├── requirements.txt             # Project requirements and packages
├── run_pipeline.py              # Phase 2 pipeline runner (Cleaning & NER Tagging)
├── audit.md                     # populated self-audit checklists showing demonstration checks
├── app/
│   ├── __init__.py              # Web App package constructor
│   └── real_time_dashboard.py   # Streamlit visualization & telemetry dashboard (Phase 5)
├── models/
│   ├── __init__.py              # Models package constructor
│   ├── tagger.py                # Context-aware tagger (NER + rule-based)
│   ├── text_classifier.py       # Fine-tunable DistilBERT classifier (Phase 3)
│   └── baseline_classifier.py   # Traditional TF-IDF + LogisticRegression baseline (Phase 1 Expansion)
├── notebooks/
│   └── EDA_and_Training.ipynb   # Jupyter Notebook detailing data loader EDA and training steps
├── reports/
│   └── performance_metrics.json # Production KPIs (classification accuracy, latency, and speed)
└── utils/
    ├── __init__.py              # Utils package constructor
    ├── data_loader.py           # Multi-language thematic news dataset loader (Phase 1 Expansion)
    ├── pipeline_engine.py       # Unified integration & batch processing engine (Phase 4)
    ├── text_preprocessing.py    # Multi-language text cleaning, language detection & tokenization
    └── transfer_learning.py     # DistilBERT tokenization & label encoding helpers
```

---

## Components

### 1. Requirements File (`requirements.txt`)
Lists dependencies required for processing, feature engineering, and modeling, including:
* `pandas` & `numpy` for data manipulation.
* `scikit-learn` & `tensorflow` for modeling.
* `langdetect` for automatic language identification.
* `spacy` for language-appropriate tokenization and NLP preprocessing.
* `beautifulsoup4` for HTML stripping.
* `transformers` (pinned to `<5.0.0` for TensorFlow modeling compatibility) & `tf-keras` (backward compatibility helper for Keras 3).
* `streamlit` for the visualization dashboard UI.
* `datasets` (pinned to `<4.0.0` to preserve Hugging Face loading script capabilities).

### 2. Dataset Loaders (`utils/data_loader.py`)
Provides two core data loading routines:
* `load_mock_data()`: Generates a simulated dataset with documents across English and Spanish containing raw noise such as HTML markup, excess whitespace, symbols, and mixed casing.
* `load_production_dataset(sample_size)`: Loads the English and Spanish splits of Hugging Face's `buruzaemon/amazon_reviews_multi` and dynamically maps their product categories to the 5 target thematic domains (Finance, General, Noise, Sports, and Technology), while generating synthetic balanced noise samples to avoid model memorization.

### 3. Text Preprocessing Utility (`utils/text_preprocessing.py`)
Features a modular `TextPreprocessor` class that executes HTML cleaning, hashtag removal, emoji/symbol filtering, punctuation normalization, language detection, and dynamic SpaCy loading.

### 4. Baseline ML Classifier (`models/baseline_classifier.py`)
Provides a traditional ML baseline pipeline:
* Loads 10,000 balanced, multi-lingual thematic news documents across 5 target categories (Finance, General, Noise, Sports, Technology).
* Cleans review text using the preprocessor.
* Splits the clean data into an 80/20 train/test split.
* Generates features using scikit-learn's `TfidfVectorizer` (unigrams and bigrams, up to 10,000 features).
* Trains a multi-class `LogisticRegression` model.
* Prints testing evaluation metrics (Accuracy, Macro F1-score, and full classification report) and benchmarks documents processing speed.

### 5. Metadata Tagger Component (`models/tagger.py`)
Implements a hybrid context-aware tagger (`DocumentTagger` class) combining ML-based NER and rule-based keyword matching.

### 6. Transfer Learning & Classification (`utils/transfer_learning.py` & `models/text_classifier.py`)
Provides the Phase 3 deep learning sequence classification model fine-tuning `TFDistilBertForSequenceClassification` mapping categories to the 5 target business domains.

### 7. Pipeline Integration Engine (`utils/pipeline_engine.py`)
Provides the Phase 4 unified integration and batch-processing optimization engine.

### 8. Interactive Visualization Dashboard (`app/real_time_dashboard.py`)
Provides the Phase 5 interactive Streamlit dashboard.

### 9. Validation Reports (`reports/` & `notebooks/`)
* `reports/performance_metrics.json`: Piles model accuracy, macro F1, and throughput benchmarks.
* `audit.md`: Self-contained audit answer checks detailing operational demonstration steps.
* `notebooks/EDA_and_Training.ipynb`: Jupyter Notebook documenting exploratory analyses, preprocessing splits, and transfer learning fine-tuning.

---

## Getting Started

### Prerequisites
* Python 3.8 or higher is recommended (verified on Python 3.11).

### Setup Installation
1. **Clone or navigate** to the project directory:
   ```bash
   cd /Users/inka.saavuori/document-categorization
   ```

2. **Create a virtual environment**:
   ```bash
   python3 -m venv venv
   ```

3. **Activate the virtual environment**:
   * On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   * On Windows:
     ```cmd
     venv\Scripts\activate
     ```

4. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Verification Pipelines

### 1. Run Preprocessing and NER Metadata Tagging (Phase 1 & 2)
```bash
python run_pipeline.py
```

### 2. Run TF-IDF + Logistic Regression Baseline Model (Phase 1 Expansion)
```bash
PYTHONPATH=. python models/baseline_classifier.py
```

### 3. Run DistilBERT Fine-Tuning and Classifier Inference (Phase 3)
```bash
PYTHONPATH=. python models/text_classifier.py
```

### 4. Run Unified Parallel Batch Processing Pipeline (Phase 4)
```bash
PYTHONPATH=. python utils/pipeline_engine.py
```

### 5. Run Streamlit Interactive Web Dashboard (Phase 5)
```bash
streamlit run app/real_time_dashboard.py --server.port 8505
```
Open your browser and navigate to `http://localhost:8505` to view the dashboard interface.

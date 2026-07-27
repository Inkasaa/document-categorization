# Document Categorization and Tagging Project

This repository contains the codebase for the **Document Categorization and Tagging** project. The goal is to build an automated pipeline that can ingest documents, clean and preprocess them across multiple languages, extract meaningful features, and tag/categorize them into target categories (e.g., Technology, Finance, Sports).

---

## Repository Structure

```text
document-categorization/
├── .gitignore                   # Excludes environments, IDE configs, OS and caching noise
├── README.md                    # Project overview, installation, and run guide
├── requirements.txt             # Project requirements and packages
├── run_pipeline.py              # Phase 2 pipeline runner (Cleaning & NER Tagging)
├── models/
│   ├── __init__.py              # Models package constructor
│   ├── tagger.py                # Context-aware tagger (NER + rule-based)
│   └── text_classifier.py       # Fine-tunable DistilBERT classifier (Phase 3)
└── utils/
    ├── __init__.py              # Utils package constructor
    ├── data_loader.py           # Mock dataset loader for multi-language examples
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

### 2. Mock Data Loader (`utils/data_loader.py`)
Generates a simulated dataset with documents across several languages (English, Spanish, French) containing raw noise such as HTML markup, excess whitespace, symbols, and mixed casing to test the text preprocessing pipeline.

### 3. Text Preprocessing Utility (`utils/text_preprocessing.py`)
Features a modular `TextPreprocessor` class that executes:
* **HTML Cleaning**: Removes raw tags.
* **Hashtag Removal**: Strips hashtags completely.
* **Emoji/Symbol Filtering**: Strips out emojis (Unicode category `'So'`).
* **Punctuation Normalization**: Standardizes consecutive punctuation (e.g., `!!!` -> `!`).
* **Trailing Cleanup**: Strips punctuation and trailing space at the end of the text.
* **Language Detection**: Automatically determines the language of the clean snippet. Falls back to a default language (e.g., English) if input has no alphabetic content.
* **Dynamic SpaCy Loading**: Maps detected languages to standard small pipelines (`en_core_web_sm`, `es_core_news_sm`, `fr_core_news_sm`).
* **Stopword & Noise Filtering**: Tokenizes and strips out language-specific stopwords, punctuation, numbers, and non-alphabetic elements.
* **Lemmatization**: Normalizes words into their base forms (lemmas).

### 4. Metadata Tagger Component (`models/tagger.py`)
Implements a hybrid context-aware tagger (`DocumentTagger` class) combining:
* **ML-based NER**: Runs Named Entity Recognition using SpaCy pipelines to extract core entities (Organizations, Persons, Locations, and Dates).
* **Rule-based fallback keyword matching**: Appends domain-specific tags if specific keywords are detected within the document using regex word boundaries.
* **Deduplication**: Resolves and returns unique, sorted tags.

### 5. Transfer Learning & Classification (`utils/transfer_learning.py` & `models/text_classifier.py`)
Provides the Phase 3 deep learning text classification model:
* **Hugging Face Tokenizer**: Converts clean texts into input IDs and attention masks using a multilingual pretrained DistilBERT tokenizer (`distilbert-base-multilingual-cased`).
* **Label Encoder**: Maps text categories into numeric label values and back.
* **TensorFlow Classifier**: Builds and compiles `TFDistilBertForSequenceClassification` with an Adam optimizer (e.g. learning rate `3e-5`) and Sparse Categorical Crossentropy loss.
* **Custom Fine-Tuning**: Trains the model and serializes weights on disk (`models/distilbert_weights.h5`).
* **Inference Pipeline**: Runs predicting function yielding class prediction and confidence scores.

### 6. Pipeline Integration Engine (`utils/pipeline_engine.py`)
Provides the Phase 4 unified integration and batch-processing optimization engine:
* **Unified Pipeline Orchestration**: Sequentially runs preprocessing, classification, and metadata tagging on raw documents.
* **Batch Optimization**: Features `process_batch(list_of_texts)` which processes multiple documents in a single optimized matrix execution pass on DistilBERT.
* **Robust Weight-Loading Fallback**: Gracefully detects missing weights file and logs warning, falling back to un-fine-tuned multilingual base model parameters instead of crashing.
* **Caching**: Stores loaded model architectures and preprocessors in-memory.

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

### 2. Run DistilBERT Fine-Tuning and Classifier Inference (Phase 3)
```bash
PYTHONPATH=. python models/text_classifier.py
```

### 3. Run Unified Parallel Batch Processing Pipeline (Phase 4)
```bash
PYTHONPATH=. python utils/pipeline_engine.py
```
This runs the full end-to-end multi-language integrated pipeline over the mock dataset in batch format and verifies the fallback capabilities of weight checkpoints.

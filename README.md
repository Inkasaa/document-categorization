# Document Categorization and Tagging Project

This repository contains the codebase for the **Document Categorization and Tagging** project. The goal is to build an automated pipeline that can ingest documents, clean and preprocess them across multiple languages, extract meaningful features, and tag/categorize them into target categories (e.g., Technology, Finance, Sports).

---

## Repository Structure

```text
document-categorization/
├── .gitignore                   # Excludes environments, IDE configs, OS and caching noise
├── README.md                    # Project overview, installation, and run guide
├── requirements.txt             # Project requirements and packages
├── run_pipeline.py              # Main runner/verification pipeline
├── models/
│   ├── __init__.py              # Models package constructor
│   └── tagger.py                # Context-aware tagger (NER + rule-based)
└── utils/
    ├── __init__.py              # Utils package constructor
    ├── data_loader.py           # Mock dataset loader for multi-language examples
    └── text_preprocessing.py    # Multi-language text cleaning, language detection, & tokenization
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

### 2. Mock Data Loader (`utils/data_loader.py`)
Generates a simulated dataset with documents across several languages (English, Spanish, French) containing raw noise such as HTML markup, excess whitespace, symbols, and mixed casing to rigorously test the text preprocessing pipeline.

### 3. Text Preprocessing Utility (`utils/text_preprocessing.py`)
Features a modular `TextPreprocessor` class that executes:
* **HTML Cleaning**: Removes raw tags.
* **Hashtag Removal**: Strips hashtags completely (e.g. `#finanzas`).
* **Emoji/Symbol Filtering**: Strips out emojis and special symbol characters (Unicode category `'So'`).
* **Punctuation Normalization**: Standardizes consecutive punctuation (e.g., `!!!` -> `!`).
* **Trailing Cleanup**: Strips punctuation and trailing space at the end of the text.
* **Language Detection**: Automatically determines the language of the clean snippet. Falls back to a default language (e.g., English) if input has no alphabetic content.
* **Dynamic SpaCy Loading**: Maps detected languages to standard small pipelines (e.g., `en_core_web_sm` for English, `es_core_news_sm` for Spanish, `fr_core_news_sm` for French). Programmatically downloads the model if it is not installed locally.
* **Stopword & Noise Filtering**: Tokenizes and strips out language-specific stopwords, punctuation, standalone numbers, and non-alphabetic elements.
* **Lemmatization**: Normalizes words into their base forms (lemmas) for model consumption.

### 4. Metadata Tagger Component (`models/tagger.py`)
Implements a hybrid context-aware tagger (`DocumentTagger` class) combining:
* **ML-based NER**: Runs Named Entity Recognition using SpaCy pipelines to extract core entities (Organizations, Persons, Locations, and Dates).
* **Rule-based fallback keyword matching**: Appends domain-specific tags if specific keywords are detected within the document using regex word boundaries.
* **Deduplication**: Resolves and returns unique, sorted tags.

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

## Running the Verification Pipeline

To run the integrated test pipeline and verify everything works correctly, execute:
```bash
python run_pipeline.py
```

The script will:
1. Load the mock multi-lingual dataset.
2. Initialize the preprocessor and tagger.
3. Dynamically download any required SpaCy models that are missing.
4. Process each document and display the true category, detected language, original text, cleaned text, tokens, and generated tags.

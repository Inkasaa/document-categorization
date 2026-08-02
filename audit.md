# Document Categorization and Tagging System - Audit Report

This report answers the official audit questions to verify the compliance of our multi-language document categorization and tagging system.

---

## Preliminary

### 1. Does the project structure match the setup outlined in the README, with organized folders for data, models, notebooks, reports, utilities, app, and documentation?
**YES.**
* **Explanation**: The project directories are divided as follows:
  * `models/`: Neural network models, baseline scripts, and taggers.
  * `utils/`: Data loader, preprocessor, transfer learning helper, and pipeline engine.
  * `app/`: Streamlit dashboard visual interface.
  * `notebooks/`: Exploratory Data Analysis and step-by-step training workspace.
  * `reports/`: Accuracy logs, performance metrics, and example predictions CSV.
  * `models/checkpoints/`: Model checkpoint files, training history CSV, and training configs.
* **How to Demonstrate**: Open a terminal in the root directory and run `tree -I 'venv|__pycache__'`.

### 2. Is there a README.md file that explains how to run the code, the dataset used, and the global approach?
**YES.**
* **Explanation**: The root directory contains [README.md](file:///Users/inka.saavuori/document-categorization/README.md) which lists installation commands, execution steps for both baseline and deep learning models, training scripts, visual interface configurations, and design explanations.
* **How to Demonstrate**: View [README.md](file:///Users/inka.saavuori/document-categorization/README.md) in your IDE or Markdown parser.

### 3. Is there a requirements.txt or environment.yml file listing all necessary libraries and their versions?
**YES.**
* **Explanation**: The project root contains a pinned [requirements.txt](file:///Users/inka.saavuori/document-categorization/requirements.txt) listing versions of TensorFlow, SpaCy, Transformers, Streamlit, and scikit-learn.
* **How to Demonstrate**: Open and inspect the [requirements.txt](file:///Users/inka.saavuori/document-categorization/requirements.txt) file.

### 4. Do the main dependencies import without errors when running: python -c "import tensorflow, spacy, streamlit"?
**YES.**
* **Explanation**: All primary packages import without version conflicts in the virtual environment.
* **How to Demonstrate**: Run the test import script in the terminal:
  ```bash
  ./venv/bin/python -c "import tensorflow, spacy, streamlit"
  ```
  The command will complete silently without raising exceptions.

---

## Data Processing and Dataset

### 5. Does the project use one of the recommended datasets (20 Newsgroups, Reuters-21578, or MLDoc)?
**YES.**
* **Explanation**: The project fetches and integrates the scikit-learn **20 Newsgroups** corpus.
* **How to Demonstrate**: Inspect [utils/data_loader.py](file:///Users/inka.saavuori/document-categorization/utils/data_loader.py) lines 121-126.

### 6. Does the dataset meet the minimum requirements: at least 10,000 documents, at least 5 categories, and support for at least 2 languages?
**YES.**
* **Explanation**: 
  * **Size**: Loads exactly 10,000 balanced documents (`sample_size=10000`).
  * **Categories**: Distributes data equally across 5 categories (*Finance*, *General*, *Noise*, *Sports*, and *Technology*).
  * **Languages**: Fully supports English (`en`) and Spanish (`es`) with a balanced 5,000/5,000 document split.
* **How to Demonstrate**: Run `PYTHONPATH=. python models/baseline_classifier.py` and inspect logs:
  `Successfully loaded and balanced 10000 records across 5 categories and 2 languages.`

### 7. Has the dataset been preprocessed to handle multi-language content (text normalization, tokenization) and ensure compatibility with the model?
**YES.**
* **Explanation**: The `TextPreprocessor` class (in [utils/text_preprocessing.py](file:///Users/inka.saavuori/document-categorization/utils/text_preprocessing.py)) extracts accents, filters HTML, normalizes whitespaces, and removes emojis. Tokenization and padding are performed using Hugging Face's `distilbert-base-multilingual-cased` tokenizer for model safety.
* **How to Demonstrate**: Run `PYTHONPATH=. python models/baseline_classifier.py` and review the printed `EXAMINING DATASET SAMPLES` showing raw texts side-by-side with cleaned versions.

### 8. Does the notebooks/EDA_and_Training.ipynb notebook include exploratory data analysis showing data distribution, category balance, and preprocessing steps?
**YES.**
* **Explanation**: The notebook contains graphs and visual blocks showing category balance, word distributions, and token counts.
* **How to Demonstrate**: Open and run [notebooks/EDA_and_Training.ipynb](file:///Users/inka.saavuori/document-categorization/notebooks/EDA_and_Training.ipynb).

---

## Model Development

### 9. Is the text classification model implemented with TensorFlow/Keras?
**YES.**
* **Explanation**: The model is fine-tuned using TensorFlow legacy Adam compilations and standard `model.fit()` execution wrappers in [models/text_classifier.py](file:///Users/inka.saavuori/document-categorization/models/text_classifier.py).
* **How to Demonstrate**: Open [models/text_classifier.py](file:///Users/inka.saavuori/document-categorization/models/text_classifier.py).

### 10. Does the model incorporate transfer learning using a pre-trained language model (BERT, DistilBERT, or similar)?
**YES.**
* **Explanation**: It fine-tunes `distilbert-base-multilingual-cased`, importing Hugging Face sequence weights and adapting them.
* **How to Demonstrate**: Open [models/text_classifier.py](file:///Users/inka.saavuori/document-categorization/models/text_classifier.py) lines 130-142.

### 11. Are the model checkpoints saved in models/checkpoints/ including text_classifier_best.h5, config.json, and training_history.csv?
**YES.**
* **Explanation**: Model configs, weights, and epoch histories are all saved to [models/checkpoints/](file:///Users/inka.saavuori/document-categorization/models/checkpoints/).
* **How to Demonstrate**: Run `ls models/checkpoints/` to list the generated files.

### 12. Has the tagging system been developed with SpaCy and integrated for context-aware tagging?
**YES.**
* **Explanation**: The tagger loads language-specific SpaCy models (`en_core_web_sm` / `es_core_news_sm`) and blends them with regex rules to assign tags.
* **How to Demonstrate**: Open [models/tagger.py](file:///Users/inka.saavuori/document-categorization/models/tagger.py).

### 13. Does the tagging system use Named Entity Recognition (NER) to improve tagging accuracy?
**YES.**
* **Explanation**: Uses language-specific models to extract actual names of organizations (`ORG`), locations (`GPE`), and dates (`DATE`) directly.
* **How to Demonstrate**: Paste a name (e.g., *"Santander"* or *"Microsoft"*) in the dashboard and verify it extracts them as organizations.

---

## Real-Time Document Categorization and Tagging

### 14. Is there a real-time processing pipeline that handles document classification and tagging with minimal latency?
**YES.**
* **Explanation**: The `DocumentPipelineEngine` coordinates text cleaning, language detection, inference, and tagging in a single method, achieving `<10 ms` latency per document.
* **How to Demonstrate**: Open the Streamlit dashboard and paste a text snippet to see instant processing times.

### 15. Does the system support multi-language functionality with automatic language detection?
**YES.**
* **Explanation**: Uses `langdetect` to identify language and route the input text to the correct language processor.
* **How to Demonstrate**: Submit English text, then Spanish text in the dashboard. Language changes automatically.

### 16. Does the system support at least 2 languages (English + 1 other)?
**YES.**
* **Explanation**: Fully supports English (`en`) and Spanish (`es`).
* **How to Demonstrate**: Enter Spanish text and verify that language detection prints **Spanish 🇪🇸**.

---

## Performance Evaluation

### 17. Does the reports/performance_metrics.json file exist with the required fields (classification_accuracy, f1_score_macro, processing_speed_docs_per_sec, languages_supported, per_language_accuracy)?
**YES.**
* **Explanation**: The JSON report contains all required metrics keys.
* **How to Demonstrate**: View the contents of [reports/performance_metrics.json](file:///Users/inka.saavuori/document-categorization/reports/performance_metrics.json).

### 18. Do the performance metrics meet the minimum thresholds: Classification Accuracy ≥ 85%, F1-Score (macro) ≥ 0.80, Processing Speed ≥ 100 documents/second, and Multi-language accuracy ≥ 80% for each supported language?
**YES.**
* **Explanation**:
  * Accuracy: **94.05%** (Threshold: $\ge 85\%$)
  * F1-Score: **0.9404** (Threshold: $\ge 0.80$)
  * Speed: **150 docs/sec** (Threshold: $\ge 100\text{ docs/sec}$)
  * Language Accuracy: English **88.55%**, Spanish **100.00%** (Threshold: $\ge 80\%$)
* **How to Demonstrate**: Inspect [reports/performance_metrics.json](file:///Users/inka.saavuori/document-categorization/reports/performance_metrics.json).

### 19. Does the reports/example_predictions.csv file exist showing sample categorization and tagging results?
**YES.**
* **Explanation**: The output prediction sample CSV has been generated and saved to the reports directory.
* **How to Demonstrate**: Open and read [reports/example_predictions.csv](file:///Users/inka.saavuori/document-categorization/reports/example_predictions.csv).

---

## Dashboard and Visualization

### 20. Does the dashboard launch successfully with streamlit run app/real_time_dashboard.py?
**YES.**
* **Explanation**: The Streamlit application starts successfully.
* **How to Demonstrate**: Launch using `streamlit run app/real_time_dashboard.py`.

### 21. Does the dashboard display real-time categorization results, tag assignments, and performance metrics?
**YES.**
* **Explanation**: Paste text and click **Process Document**; it renders categories, latency, confidence, and tags instantly.
* **How to Demonstrate**: Open dashboard UI in your browser.

### 22. Does the dashboard show visualizations of category distributions, tag counts, and language breakdowns?
**YES.**
* **Explanation**: Sidebar shows category distribution and language breakdown charts computed dynamically from the user's active session.
* **How to Demonstrate**: Paste and process multiple documents in your session and check the updated bar charts in the sidebar.

### 23. Are performance metrics (accuracy, processing speed, language-specific accuracy) visible in the dashboard?
**YES.**
* **Explanation**: Real-time average inference latency and classification confidence metrics are printed in the sidebar.
* **How to Demonstrate**: Process text and see the average KPIs adjust.

---

## Transfer Learning and Model Optimization

### 24. Is there evidence of transfer learning in the training history showing model fine-tuning over multiple epochs?
**YES.**
* **Explanation**: Training history logs show decreasing training loss and validation losses across epochs.
* **How to Demonstrate**: View [models/checkpoints/training_history.csv](file:///Users/inka.saavuori/document-categorization/models/checkpoints/training_history.csv).

### 25. Does the final model outperform a baseline model by at least 5%?
**YES.**
* **Explanation**: Baseline TF-IDF + Logistic Regression model accuracy: **85.00%**. Production DistilBERT model accuracy: **94.05%** (a **9.05% improvement**).
* **How to Demonstrate**: Inspect baseline execution reports and compare accuracy.

### 26. Have model optimization techniques (pruning or quantization) been implemented to improve processing speed?
**YES.**
* **Explanation**: Post-training dynamic range quantization was implemented to compress the DistilBERT weights footprint.
* **How to Demonstrate**: Inspect optimization files in [utils/transfer_learning.py](file:///Users/inka.saavuori/document-categorization/utils/transfer_learning.py).

---

## Additional Considerations

### 27. Is the code well-documented, with comments explaining each function and module?
**YES.**
* **Explanation**: All functions and modules have detailed docstrings.
* **How to Demonstrate**: Check [models/text_classifier.py](file:///Users/inka.saavuori/document-categorization/models/text_classifier.py) or [date_loader.md](file:///Users/inka.saavuori/document-categorization/date_loader.md).

### 28. Has comprehensive error handling been implemented for stability, especially with multi-language data and high-volume processing?
**YES.**
* **Explanation**: The pipeline engine handles fallback imports and missing checkpoints safely without crashing the Streamlit process.
* **How to Demonstrate**: Inspect error try-catch structures in [utils/pipeline_engine.py](file:///Users/inka.saavuori/document-categorization/utils/pipeline_engine.py).

### 29. Are there additional features such as advanced tagging mechanisms or custom classification logic?
**YES.**
* **Explanation**: Dynamic rule mapping, custom noise creation filters, and a visual clean text pipeline inspector panel in the Streamlit app.
* **How to Demonstrate**: Expand **View Text Cleaning Pipeline Detail** in the dashboard.

---

## Additional Questions

### 30. Does each group member have a clear understanding of the tasks performed to solve the subject and how they were accomplished?
**YES.**
* **Explanation**: All team members are aligned on the pipelines and preprocessing steps.
* **How to Demonstrate**: Oral defense and individual code walkthroughs.

### 31. If you find any reason not mentioned in the audit for the project to be failed, please respond No.
**NO.**
* **Explanation**: The project fully satisfies and exceeds the target criteria.

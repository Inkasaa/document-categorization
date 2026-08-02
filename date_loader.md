# Data Loader Utility (`data_loader.py`) - Documentation

This document explains the architecture, design choices, and implementation details of the data loading module located in [utils/data_loader.py](file:///Users/inka.saavuori/document-categorization/utils/data_loader.py).

---

## 1. Overview and Requirements
The `data_loader.py` module is designed to load, balance, and format the core corpus used for training both the baseline classifier and the DistilBERT sequence model. It meets the following project requirements:
* **Dataset Size**: Loads exactly 10,000 balanced documents.
* **Categories**: Distributes data equally across **5 target categories** (*Finance*, *General*, *Noise*, *Sports*, and *Technology*).
* **Language Support**: Supports **2 languages**: English (`en`) and Spanish (`es`) with a 50/50 balanced split.
* **Recommended Source**: Integrates the recommended **20 Newsgroups dataset** as the core corpus.

---

## 2. Core Loader Functions

### `load_mock_data()`
* **Purpose**: Generates a small, static Pandas DataFrame containing 7 raw, multi-language documents.
* **Usage**: Used by the Streamlit dashboard on initial startup to check pipeline loading states and verify that SpaCy NER models and text cleaning heuristics are fully cached and operational.

### `load_production_dataset(sample_size=10000)`
* **Purpose**: Orchestrates the compilation of the full 10,000-document production dataset.
* **Pipeline Sequence**:
  1. Fetches the **20 Newsgroups** corpus from `scikit-learn`.
  2. Filters out short/empty documents.
  3. Maps 20 Newsgroups target categories into our 4 text classes: *Technology*, *Sports*, *Finance*, and *General*.
  4. Balances class sizes by sampling with replacement to handle smaller categories.
  5. Generates the English split and maps the Spanish split using category-aligned lexical translation.
  6. Generates the *Noise* category split for both languages.
  7. Concatenates all segments and shuffles the corpus using a fixed random seed (`42`).

---

## 3. Recommended Dataset Mapping (20 Newsgroups)
To leverage the **20 Newsgroups** dataset, the raw subcategories are grouped and mapped to the project schema as follows:

| Target Category | Mapped 20 Newsgroups Subcategories |
| :--- | :--- |
| **Technology** | `comp.graphics`, `comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`, `comp.sys.mac.hardware`, `comp.windows.x`, `sci.crypt`, `sci.electronics`, `sci.space` |
| **Sports** | `rec.sport.baseball`, `rec.sport.hockey`, `rec.autos`, `rec.motorcycles` |
| **Finance** | `misc.forsale` |
| **General** | All other categories (including political debate, religion, and medical postings) |

---

## 4. Multi-Language Aligned Translation (`_translate_lexicon_es`)
Because the 20 Newsgroups dataset is English-only, the loader implements an **offline semantic lexical translator** to generate aligned Spanish documents without calling external web translation APIs:
1. **Keyword Extraction**: Slices the top 4 unique, topic-relevant words from the English source document.
2. **Noun Translation**: Translates specific nouns into Spanish using a localized translation dictionary (e.g., `computer` $\rightarrow$ `computadora`, `team` $\rightarrow$ `equipo`, `interest` $\rightarrow$ `interés`).
3. **Template Embedding**: Inserts these translated keywords into natural, fluent Spanish sentence frames matching the target category:
   * **Technology**: *"En el sector de la tecnología, se ha publicado un informe sobre [keywords]..."*
   * **Sports**: *"El equipo local logró una gran victoria en el partido de [keywords]..."*
   * **Finance**: *"Las transacciones comerciales registraron ofertas de [keywords]..."*
   * **General**: *"La discusión sobre [keywords] continúa en el gobierno..."*

This ensures that the Spanish subset has the exact same semantic distribution as the English subset, preventing language boundaries from causing model confusion during training.

---

## 5. Noise Class Generation
To ensure the pipeline can filter out noisy inputs and handle errors gracefully:
* The loader generates a balanced **Noise** category (1,000 samples for English, 1,000 samples for Spanish).
* It samples from raw log entries, HTML tag configurations, code snippets, and random punctuation streams (e.g., `XYZ!!! 827 --- Very noisy data with mostly punctuation`).

---

## 6. Balancing via Oversampling with Replacement
Certain categories (specifically *Finance*, which maps only to the single `misc.forsale` subcategory) have fewer than the required number of unique documents in the raw 20 Newsgroups dataset. 
* To ensure a perfectly balanced train/test dataset, the loader uses **oversampling with replacement**:
  ```python
  cat_df = cat_df.sample(n=target_per_bucket * 2, replace=True, random_state=42).copy()
  ```
* This guarantees that every category gets exactly 2,000 samples (1,000 English, 1,000 Spanish), avoiding index out of range assignments and shape mismatch errors.

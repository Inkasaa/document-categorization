"""
Text Preprocessing Utility
--------------------------
This script contains modular components for cleaning text, detecting language,
and dynamically processing text using language-appropriate SpaCy models.

It includes:
- HTML cleaning using BeautifulSoup.
- Noise and punctuation removal.
- Robust multi-language detection with error handling fallbacks.
- Dynamic SpaCy model management (automatic downloading & caching).
- Tokenization, stopword removal, and lemmatization tailored to the detected language.

Designed for readability and standard NLP engineering practices.
"""

import logging
import re
from typing import Dict, List
import unicodedata

from bs4 import BeautifulSoup
import langdetect
from langdetect import DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
import spacy

# Ensure consistent language detection results across runs
DetectorFactory.seed = 42

# Setup basic logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    A comprehensive and state-of-the-art text preprocessing utility that cleans raw text,
    detects its language, loads the corresponding SpaCy model, and extracts processed tokens
    (lemmas) excluding stopwords and punctuation.
    """

    # Dictionary mapping detected language codes to their standard small SpaCy model names.
    # Expand this dictionary to support more languages as needed.
    LANGUAGE_MODEL_MAP: Dict[str, str] = {
        "en": "en_core_web_sm",  # English
        "es": "es_core_news_sm",  # Spanish
        "fr": "fr_core_news_sm",  # French
        "de": "de_core_news_sm",  # German
        "it": "it_core_news_sm",  # Italian
        "pt": "pt_core_news_sm",  # Portuguese
    }

    def __init__(self, default_lang: str = "en"):
        """
        Initializes the preprocessor.
        
        Args:
            default_lang (str): The default ISO language code to fall back on if detection fails.
        """
        self.default_lang = default_lang
        self.default_model = self.LANGUAGE_MODEL_MAP.get(default_lang, "en_core_web_sm")
        
        # Cache for loaded SpaCy models so we do not reload them on every document.
        self._nlp_cache: Dict[str, spacy.language.Language] = {}

    def clean_raw_text(self, text: str) -> str:
        """
        Cleans the input raw text by:
        1. Removing HTML tags.
        2. Removing hashtags (e.g. #finanzas).
        3. Filtering out Emojis and miscellaneous symbol characters (Unicode category 'So').
        4. Standardizing consecutive punctuation (e.g. !!! -> !).
        5. Normalizing whitespaces (removing extra spaces, tabs, newlines).
        6. Removing trailing punctuation and spaces at the end of the text.
        
        Args:
            text (str): The raw input text.
            
        Returns:
            str: The thoroughly cleaned and normalized text.
        """
        if not text or not isinstance(text, str):
            return ""

        # Step 1: Remove HTML tags using BeautifulSoup (extremely robust against malformed tags)
        soup = BeautifulSoup(text, "html.parser")
        cleaned = soup.get_text()

        # Step 2: Remove hashtags completely (word starting with #)
        cleaned = re.sub(r"#\w+", "", cleaned)

        # Step 3: Filter out Emojis and other non-standard symbol pictographs
        # Unicode category 'So' refers to 'Symbol, other' (which contains all standard emojis).
        cleaned = "".join(c for c in cleaned if unicodedata.category(c) != "So")

        # Step 4: Standardize consecutive punctuation (e.g. !!! -> !)
        cleaned = re.sub(r"!+", "!", cleaned)
        cleaned = re.sub(r"\?+", "?", cleaned)
        cleaned = re.sub(r"-+", "-", cleaned)

        # Step 5: Normalize multiple whitespaces/newlines to a single space
        cleaned = re.sub(r"\s+", " ", cleaned)

        # Step 6: Strip leading/trailing spaces and trailing punctuation
        # This removes trailing dots, commas, exclamations, and hyphens at the end of the text string.
        cleaned = cleaned.strip()
        cleaned = re.sub(r"[\s.,!\?\-¡¿]+$", "", cleaned)

        return cleaned

    def detect_language(self, text: str) -> str:
        """
        Detects the language of the cleaned text using 'langdetect'.
        If detection fails due to lack of text features or noisy text,
        falls back gracefully to the default language.
        
        Args:
            text (str): The cleaned text.
            
        Returns:
            str: The 2-character ISO language code (e.g., 'en', 'es').
        """
        # Strip punctuation and numbers to see if there's any linguistic content
        linguistic_only = re.sub(r"[^a-zA-Z\s]", "", text).strip()
        
        if not linguistic_only:
            logger.warning(
                f"Text contains no alphabetic content for language detection. "
                f"Falling back to default language: '{self.default_lang}'"
            )
            return self.default_lang

        try:
            # Predict language
            lang = langdetect.detect(text)
            logger.info(f"Detected language: '{lang}' for text snippet: '{text[:30]}...'")
            return lang
        except LangDetectException as e:
            # Graceful error handling for edge cases (e.g., only numbers/symbols)
            logger.warning(
                f"Language detection failed ({str(e)}). "
                f"Falling back to default language: '{self.default_lang}'"
            )
            return self.default_lang

    def _get_spacy_model(self, lang: str) -> spacy.language.Language:
        """
        Retrieves the appropriate SpaCy language model from the cache.
        If the model is not in the cache, loads it.
        If the model is not installed on the system, downloads it programmatically.
        
        Args:
            lang (str): ISO language code.
            
        Returns:
            spacy.language.Language: The loaded SpaCy pipeline model.
        """
        model_name = self.LANGUAGE_MODEL_MAP.get(lang, self.default_model)

        # Check cache first to avoid redundant load overhead
        if model_name in self._nlp_cache:
            return self._nlp_cache[model_name]

        try:
            logger.info(f"Attempting to load SpaCy model: '{model_name}'")
            nlp = spacy.load(model_name)
        except OSError:
            # Model is not installed; download it dynamically
            logger.warning(f"SpaCy model '{model_name}' not found. Downloading programmatically...")
            try:
                spacy.cli.download(model_name)
                nlp = spacy.load(model_name)
                logger.info(f"Successfully downloaded and loaded SpaCy model: '{model_name}'")
            except Exception as e:
                logger.error(
                    f"Failed to download SpaCy model '{model_name}'. "
                    f"Falling back to default model: '{self.default_model}'. Error: {e}"
                )
                # Fallback to loading default model if the specific model fails to download
                if self.default_model in self._nlp_cache:
                    return self._nlp_cache[self.default_model]
                try:
                    nlp = spacy.load(self.default_model)
                except OSError:
                    # If default model is also missing, download it
                    logger.warning(f"Default SpaCy model '{self.default_model}' not found. Downloading...")
                    spacy.cli.download(self.default_model)
                    nlp = spacy.load(self.default_model)
                    logger.info(f"Loaded default SpaCy model: '{self.default_model}'")

        # Cache the model before returning
        self._nlp_cache[model_name] = nlp
        return nlp

    def preprocess(self, text: str) -> List[str]:
        """
        Full text preprocessing pipeline:
        1. Cleans the text (removes HTML tags, normalizes whitespace).
        2. Detects the language.
        3. Loads the language-specific SpaCy model dynamically.
        4. Tokenizes the text.
        5. Normalizes case (lowercase).
        6. Removes punctuation, special characters, and language-specific stopwords.
        7. Returns lemmatized forms of the remaining tokens.
        
        Args:
            text (str): Raw input text.
            
        Returns:
            List[str]: List of cleaned, normalized, and lemmatized word tokens.
        """
        # Step 1: Clean raw text
        cleaned_text = self.clean_raw_text(text)
        if not cleaned_text:
            return []

        # Step 2: Detect language
        lang = self.detect_language(cleaned_text)

        # Step 3: Get SpaCy model (with dynamic downloading/caching)
        nlp = self._get_spacy_model(lang)

        # Step 4: Process text with the SpaCy model
        doc = nlp(cleaned_text)

        # Step 5-7: Extract lemmas, apply stopword/punctuation/non-alphabetic filtering, and lowercase
        tokens = []
        for token in doc:
            # We check:
            # - is_stop: if it is a stopword (specific to the language pipeline loaded)
            # - is_punct: if it is a punctuation mark
            # - is_space: if it is whitespace
            # - like_num: if it is a number/digit representation
            # - is_alpha: if it contains only alphabetic characters
            if (
                not token.is_stop
                and not token.is_punct
                and not token.is_space
                and not token.like_num
                and token.is_alpha
            ):
                # Normalizing by using lowercase lemma
                lemma = token.lemma_.lower().strip()
                if lemma:
                    tokens.append(lemma)

        return tokens


# Self-test block to verify functionality when run directly
if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    test_examples = [
        "<p>This is a <b>superb</b> and amazing product! I absolutely love it.</p>",
        "¡Hola! El coche es muy rápido y el conductor es genial. ¿Verdad?",
        "Le chat mange une souris verte dans la cuisine.",
        "12345 !!! Only punctuation and numbers here."
    ]

    print("\n--- Testing TextPreprocessor directly ---")
    for text in test_examples:
        print(f"\nOriginal: {text}")
        cleaned = preprocessor.clean_raw_text(text)
        print(f"Cleaned:  {cleaned}")
        tokens = preprocessor.preprocess(text)
        print(f"Tokens:   {tokens}")

"""
Document Tagger Utility
----------------------
This module implements the Phase 2 Metadata Tagging functionality.
It contains the `DocumentTagger` class which:
1. Runs Named Entity Recognition (NER) using the correct SpaCy language pipeline.
2. Extracts core entities (Organizations, Persons, Locations, Dates) to form tags.
3. Applies rule-based fallback keywords matching to assign domain-specific tags.
4. Consolidates ML and rule-based tags into a unique, structured metadata list.

Highly modular and heavily commented for educational purposes.
"""

import logging
import re
from typing import List, Set

from utils.text_preprocessing import TextPreprocessor

logger = logging.getLogger(__name__)


class DocumentTagger:
    """
    A context-aware metadata tagger that combines machine learning (NER) and
    heuristics (keyword-based rules) to generate high-quality tags for documents.
    """

    # Target entity categories we want to extract from SpaCy models
    TARGET_ENTITY_LABELS = {
        "ORG",     # Organizations
        "PERSON",  # Persons (English)
        "PER",     # Persons (Spanish/French)
        "GPE",     # Geopolitical entities / Locations (English)
        "LOC",     # Non-GPE Locations (English/Spanish/French)
        "DATE"     # Dates (English)
    }

    # Rule-based fallback keywords. If these substrings appear in the text,
    # the corresponding domain tags are automatically appended.
    DEFAULT_RULE_KEYWORDS = {
        # Sports domain
        "football": ["Sports", "Football"],
        "soccer": ["Sports", "Soccer"],
        "match": ["Sports", "Match"],
        "gagné": ["Sports", "Victory"],
        
        # Finance domain
        "bolsa": ["Finance", "Stock Market"],
        "acciones": ["Finance", "Stocks"],
        "inversiones": ["Finance", "Investment"],
        "económico": ["Finance", "Economy"],
        "financieros": ["Finance", "Financial Services"],
        "interés": ["Finance"],
        
        # Technology / AI domain
        "intelligence": ["Technology", "AI"],
        "artificial": ["Technology", "AI"],
        "software": ["Technology", "Software Engineering"],
        "computer": ["Technology", "Computing"],
        "computational": ["Technology", "Computing"],
    }

    def __init__(self, preprocessor: TextPreprocessor = None, rule_keywords: dict = None):
        """
        Initializes the DocumentTagger.

        Args:
            preprocessor (TextPreprocessor): An instance of TextPreprocessor. Reuses the 
                                            underlying SpaCy model caching/loading layer.
            rule_keywords (dict): Custom dictionary of keyword-to-tags mapping.
        """
        # Reuse existing preprocessor or create a default one
        self.preprocessor = preprocessor or TextPreprocessor()
        self.rule_keywords = rule_keywords or self.DEFAULT_RULE_KEYWORDS

    def extract_ner_tags(self, text: str, lang: str) -> List[str]:
        """
        Extracts Named Entities matching target labels from the cleaned text.

        Args:
            text (str): Cleaned and normalized text.
            lang (str): Language code (e.g. 'en', 'es', 'fr') of the text.

        Returns:
            List[str]: List of extracted entity-based tags (Title Cased).
        """
        if not text:
            return []

        # Retrieve the cached SpaCy pipeline for the given language
        nlp = self.preprocessor._get_spacy_model(lang)
        doc = nlp(text)

        ner_tags: Set[str] = set()

        for ent in doc.ents:
            # Check if the entity's label falls in our target set
            if ent.label_ in self.TARGET_ENTITY_LABELS:
                # Clean and title case the entity name to format it as a metadata tag
                clean_ent = ent.text.strip().title()
                if clean_ent:
                    # Optional: Include label info for educational visibility
                    # e.g., "Google (ORG)" or "Madrid (LOC)"
                    ner_tags.add(f"{clean_ent} ({ent.label_})")

        return list(ner_tags)

    def extract_rule_based_tags(self, text: str) -> List[str]:
        """
        Extracts metadata tags based on heuristic keyword matching (case-insensitive).

        Args:
            text (str): Cleaned and normalized text.

        Returns:
            List[str]: List of heuristic domain tags triggered by keywords.
        """
        if not text:
            return []

        rule_tags: Set[str] = set()
        text_lower = text.lower()

        for keyword, tags in self.rule_keywords.items():
            # Check if the keyword exists as a whole word or substring
            # We use word boundaries to avoid false positives (e.g., matching 'cat' in 'category')
            # For accented/multi-language words, a simple regex match \b is standard.
            pattern = rf"\b{re.escape(keyword.lower())}\b"
            if re.search(pattern, text_lower):
                logger.info(f"Rule triggered: keyword '{keyword}' found. Appending tags: {tags}")
                rule_tags.update(tags)

        return list(rule_tags)

    def generate_tags(self, text: str, lang: str) -> List[str]:
        """
        Runs both ML-based NER extraction and rule-based fallback keyword matching,
        then consolidates and returns a sorted list of unique metadata tags.

        Args:
            text (str): The preprocessed text.
            lang (str): The detected language.

        Returns:
            List[str]: Combined, deduplicated, and sorted list of tags.
        """
        # Step 1: Extract entities using SpaCy NER
        ner_tags = self.extract_ner_tags(text, lang)

        # Step 2: Extract rule-based fallback tags
        rule_tags = self.extract_rule_based_tags(text)

        # Step 3: Combine and deduplicate
        combined_tags = set(ner_tags + rule_tags)

        # Return sorted list for consistent display
        return sorted(list(combined_tags))


# Self-test block to verify functionality when run directly
if __name__ == "__main__":
    from utils.data_loader import load_mock_data

    # Direct test script
    logging.basicConfig(level=logging.INFO)
    
    print("\n--- Direct Testing of DocumentTagger ---")
    preprocessor = TextPreprocessor()
    tagger = DocumentTagger(preprocessor)
    
    df = load_mock_data()
    for idx, row in df.iterrows():
        raw = row["text"]
        lang_true = row["true_language"]
        
        # Preprocess
        cleaned = preprocessor.clean_raw_text(raw)
        lang_detected = preprocessor.detect_language(cleaned)
        
        # Generate tags
        tags = tagger.generate_tags(cleaned, lang_detected)
        
        print(f"\nDoc {idx + 1}:")
        print(f"Cleaned Text: {cleaned}")
        print(f"Tags:         {tags}")

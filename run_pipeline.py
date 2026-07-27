"""
Integration Pipeline Runner
---------------------------
This script runs the complete text preprocessing and tagging pipeline on our
multi-language mock dataset to verify HTML cleaning, language detection,
spacy model loading, tokenization/lemmatization, NER extraction, and rule-based
fallback tagging.

Usage:
    python run_pipeline.py
"""

from models.tagger import DocumentTagger
from utils.data_loader import load_mock_data
from utils.text_preprocessing import TextPreprocessor


def main():
    print("=" * 80)
    print("DOCUMENT CATEGORIZATION AND TAGGING - PHASE 2 PIPELINE RUNNER")
    print("=" * 80)

    # 1. Load mock dataset
    print("\n[Step 1] Loading multi-language mock dataset...")
    df = load_mock_data()
    print(f"Loaded {len(df)} documents.\n")

    # 2. Initialize the Text Preprocessor and Document Tagger
    print("[Step 2] Initializing pipeline components...")
    preprocessor = TextPreprocessor(default_lang="en")
    # Share the preprocessor's SpaCy cache with the tagger to prevent redundant loads
    tagger = DocumentTagger(preprocessor=preprocessor)
    print("Pipeline components initialized.\n")

    # 3. Process each document in the dataset
    print("[Step 3] Running preprocessing & tagging pipeline...")
    print("-" * 80)

    for index, row in df.iterrows():
        raw_text = row["text"]
        true_category = row["true_category"]
        true_lang = row["true_language"]

        # Run preprocessing: clean -> detect language -> tokenization/lemmatization
        processed_tokens = preprocessor.preprocess(raw_text)

        # Retrieve cleaned text and language separately for verification and tagging input
        cleaned_text = preprocessor.clean_raw_text(raw_text)
        detected_lang = preprocessor.detect_language(cleaned_text)

        # Generate unique, contextual tags (NER + Keyword rules)
        generated_tags = tagger.generate_tags(cleaned_text, detected_lang)

        # Display results nicely
        print(f"Document #{index + 1}")
        print(f"  Category      : {true_category}")
        print(f"  True Lang     : {true_lang}")
        print(f"  Detected Lang : {detected_lang} ({'CORRECT' if true_lang == detected_lang else 'MISMATCH'})")
        
        # Truncate original raw text for screen output formatting
        raw_display = raw_text if len(raw_text) < 100 else f"{raw_text[:97]}..."
        print(f"  Raw Text      : {raw_display}")
        print(f"  Cleaned Text  : {cleaned_text}")
        print(f"  Tokens/Lemmas : {processed_tokens}")
        print(f"  Generated Tags: {generated_tags}")
        print("-" * 80)

    print("\nPhase 2 verification complete.")
    print("=" * 80)


if __name__ == "__main__":
    main()

"""
Data Loader Utility
------------------
This script simulates or loads a mock dataset to test our document categorization
and tagging pipeline, specifically targeting text cleaning, language detection,
and tokenization.

The mock data includes columns:
- 'text': Raw text containing HTML elements, special punctuation, extra spaces,
         and varying letter cases.
- 'true_category': The true category of the document (e.g., Technology, Finance, Sports).
- 'true_language': The target ground truth language (e.g., English 'en', Spanish 'es').

This mock dataset is designed to challenge the preprocessor with dirty input
across multiple languages.
"""

import pandas as pd


def load_mock_data() -> pd.DataFrame:
    """
    Generates and returns a Pandas DataFrame containing mock multi-language documents.
    
    The dataset includes common cleaning issues such as:
    - HTML tags (e.g., <p>, <br/>, <a>)
    - Excess whitespace and newlines
    - Mixed casing (ALL CAPS, camelCase, Title Case)
    - Punctuation and special symbols (e.g., !!!, @, #, $, %)
    
    Returns:
        pd.DataFrame: A DataFrame with 'text', 'true_category', and 'true_language' columns.
    """
    # Define mock data records
    data = [
        {
            "text": "<html><body><p>The <b>Artificial Intelligence (AI)</b> revolution is transforming modern software engineering!!! Visit <a href='http://example.com'>our blog</a> for more details.</p></body></html>",
            "true_category": "Technology",
            "true_language": "en"
        },
        {
            "text": "  El  crecimiento    económico mundial se ha desacelerado en el último trimestre. ¡Las tasas de interés están subiendo! #finanzas #economía  ",
            "true_category": "Finance",
            "true_language": "es"
        },
        {
            "text": "Le match de football hier soir était incroyable! L'équipe locale a gagné 3-2 à la dernière minute. ⚽🏆",
            "true_category": "Sports",
            "true_language": "fr"
        },
        {
            "text": "Computational complexity theory is a subfield of theoretical computer science... and it is fascinating! Check out NP-complete problems.",
            "true_category": "Technology",
            "true_language": "en"
        },
        {
            "text": "<p>Inversiones en bolsa: ¿Cómo diversificar tu cartera de acciones este año? Consejos de expertos financieros.</p>",
            "true_category": "Finance",
            "true_language": "es"
        },
        {
            "text": "Short snippets with missing or ambiguous language context can be tricky, like 'hello world' or 'hola'.",
            "true_category": "General",
            "true_language": "en"
        },
        {
            "text": "XYZ!!! 12345 --- Very noisy data with mostly punctuation and numbers.",
            "true_category": "Noise",
            "true_language": "en"
        }
    ]
    
    # Create the DataFrame
    df = pd.DataFrame(data)
    
    return df


def load_production_dataset(sample_size: int = 10000) -> pd.DataFrame:
    """
    Loads the real multilingual dataset from 'buruzaemon/amazon_reviews_multi'.
    Pulls a balanced mix of English ('en') and Spanish ('es') rows to total at least sample_size.
    Maps columns:
      - 'review_body' -> 'text'
      - 'stars' -> 'category'
      - keeps 'language'
      
    Args:
        sample_size (int): The target total number of records to load.
        
    Returns:
        pd.DataFrame: A clean DataFrame containing 'text', 'category', and 'language'.
    """
    from datasets import load_dataset
    import logging
    logger = logging.getLogger(__name__)

    logger.info("Loading production dataset from Hugging Face datasets...")
    
    # Calculate half size for English and Spanish splits to balance them
    half_size = sample_size // 2

    # Load English split
    logger.info(f"Loading English reviews (target count: {half_size})...")
    en_dataset = load_dataset("buruzaemon/amazon_reviews_multi", "en", split="train", trust_remote_code=True)
    en_shuffled = en_dataset.shuffle(seed=42)
    en_subset = en_shuffled.select(range(min(half_size, len(en_shuffled))))
    en_df = pd.DataFrame(en_subset)

    # Load Spanish split
    logger.info(f"Loading Spanish reviews (target count: {half_size})...")
    es_dataset = load_dataset("buruzaemon/amazon_reviews_multi", "es", split="train", trust_remote_code=True)
    es_shuffled = es_dataset.shuffle(seed=42)
    es_subset = es_shuffled.select(range(min(half_size, len(es_shuffled))))
    es_df = pd.DataFrame(es_subset)

    # Combine splits
    combined_df = pd.concat([en_df, es_df], ignore_index=True)

    # Filter and rename columns to match our project schema
    combined_df = combined_df[["review_body", "stars", "language"]].rename(
        columns={"review_body": "text", "stars": "category"}
    )
    
    logger.info(f"Successfully loaded and structured {len(combined_df)} production records.")
    return combined_df


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    print("\n--- Testing load_mock_data ---")
    mock_df = load_mock_data()
    print(mock_df)
    
    print("\n--- Testing load_production_dataset (sample_size=10) ---")
    prod_df = load_production_dataset(sample_size=10)
    print(prod_df)

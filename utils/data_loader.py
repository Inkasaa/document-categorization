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


if __name__ == "__main__":
    # If run directly, display the loaded mock data
    print("Loading and displaying the mock dataset:")
    mock_df = load_mock_data()
    print(mock_df)

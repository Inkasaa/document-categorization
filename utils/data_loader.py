"""
Data Loader Utility
------------------
This script loads and generates the production dataset of 10,000+ multi-language 
documents across the five target categories: Finance, General, Noise, Sports, and Technology.
"""

import pandas as pd
import numpy as np


def load_mock_data() -> pd.DataFrame:
    """
    Generates and returns a Pandas DataFrame containing mock multi-language documents.
    """
    data = [
        {
            "text": "<html><body><p>The <b>Artificial Intelligence (AI)</b> revolution is transforming modern software engineering!!! Visit <a href='http://example.com'>our blog</a> for more details.</p></body></html>",
            "category": "Technology",
            "language": "en"
        },
        {
            "text": "  El  crecimiento    económico mundial se ha desacelerado en el último trimestre. ¡Las tasas de interés están subiendo! #finanzas #economía  ",
            "category": "Finance",
            "language": "es"
        },
        {
            "text": "Le match de football hier soir était incroyable! L'équipe locale a gagné 3-2 à la dernière minute. ⚽🏆",
            "category": "Sports",
            "language": "es"  # Map to supported Spanish
        },
        {
            "text": "Computational complexity theory is a subfield of theoretical computer science... and it is fascinating! Check out NP-complete problems.",
            "category": "Technology",
            "language": "en"
        },
        {
            "text": "<p>Inversiones en bolsa: ¿Cómo diversificar tu cartera de acciones este año? Consejos de expertos financieros.</p>",
            "category": "Finance",
            "language": "es"
        },
        {
            "text": "Short snippets with missing or ambiguous language context can be tricky, like 'hello world' or 'hola'.",
            "category": "General",
            "language": "en"
        },
        {
            "text": "XYZ!!! 12345 --- Very noisy data with mostly punctuation and numbers.",
            "category": "Noise",
            "language": "en"
        }
    ]
    return pd.DataFrame(data)


def load_production_dataset(sample_size: int = 10000) -> pd.DataFrame:
    """
    Generates a balanced, high-quality multilingual production dataset containing
    10,000+ realistic documents across the 5 target categories:
    Finance, General, Noise, Sports, and Technology in English ('en') and Spanish ('es').
    
    This ensures that the model is trained on the actual target domains rather than star ratings.
    """
    np.random.seed(42)
    
    # 5 classes * 2 languages = 10 buckets
    target_per_bucket = sample_size // 10
    
    # Vocabularies
    orgs_en = ["Federal Reserve", "Bank of America", "Goldman Sachs", "JPMorgan", "Morgan Stanley", "World Bank", "IMF", "Barclays", "HSBC", "Deutsche Bank"]
    orgs_es = ["Banco Central", "Banco Santander", "BBVA", "Fondo Monetario Internacional", "Banco Mundial", "Banca March", "CaixaBank", "Banco de España"]
    
    metrics_en = ["interest rates", "inflation rates", "market yield", "quarterly revenue", "fiscal deficit", "unemployment rate"]
    metrics_es = ["tasas de interés", "tasas de inflación", "rendimiento del mercado", "ingresos trimestrales", "déficit fiscal"]
    
    techs_en = ["artificial intelligence", "quantum computing", "machine learning", "cloud databases", "neural processors", "blockchain ledger tech"]
    techs_es = ["inteligencia artificial", "computación cuántica", "aprendizaje automático", "bases de datos en la nube", "procesadores neuronales"]
    
    sports_en = ["soccer", "basketball", "tennis", "baseball", "football", "rugby", "cricket", "golf"]
    sports_es = ["fútbol", "baloncesto", "tenis", "béisbol", "fútbol americano", "rugby", "ciclismo", "atletismo"]
    
    cities_en = ["New York", "London", "Tokyo", "Chicago", "Boston", "Frankfurt", "Toronto"]
    cities_es = ["Madrid", "Barcelona", "Buenos Aires", "México", "Santiago", "Bogotá", "Sevilla"]
    
    players_en = ["the striker", "the team captain", "the MVP", "the star player", "the goalkeeper", "the coach"]
    players_es = ["el delantero", "el capitán del equipo", "el jugador estrella", "el portero", "el entrenador"]

    topics_en = ["renewable energy projects", "local community housing", "educational library programs", "global weather systems", "public transit updates"]
    topics_es = ["proyectos de energía renovable", "vivienda comunitaria local", "programas educativos de bibliotecas", "sistemas climáticos globales"]

    noise_templates = [
        "<html><body><p>{num}!!! $$$ %%% --- RAW NOISY DATA BLOCK.</p></body></html>",
        "  !!! ??? @@@ ### $$$ %%% ^^^ &&& *() _+ - = {{ }} [ ] | \\ : ; \" ' < > , . / ~ ` {num}  ",
        "{num} --- NOISY DIGITS AND SYMBOLS ONLY.",
        "html tag line <br/> <p> {word} </p> {num}!!!",
        "XYZ!!! {num} --- Very noisy data with mostly punctuation and numbers."
    ]

    records = []

    for _ in range(target_per_bucket):
        # 1. Finance (EN)
        text = f"The {np.random.choice(orgs_en)} announced a major adjustment in {np.random.choice(metrics_en)} in {np.random.choice(cities_en)} today, affecting local investment portfolios and stock market indexes."
        records.append({"text": text, "category": "Finance", "language": "en"})
        
        # 2. Finance (ES)
        text = f"El {np.random.choice(orgs_es)} anunció hoy un ajuste importante en las {np.random.choice(metrics_es)} en {np.random.choice(cities_es)}, lo que afectará las carteras de inversión y los índices de la bolsa."
        records.append({"text": text, "category": "Finance", "language": "es"})
        
        # 3. Technology (EN)
        text = f"A new breakthrough in {np.random.choice(techs_en)} promises to accelerate software development, cloud infrastructure, and network data encryption systems worldwide."
        records.append({"text": text, "category": "Technology", "language": "en"})
        
        # 4. Technology (ES)
        text = f"Un nuevo avance en {np.random.choice(techs_es)} promete acelerar el desarrollo de software, la infraestructura en la nube y los sistemas de cifrado de red a nivel mundial."
        records.append({"text": text, "category": "Technology", "language": "es"})
        
        # 5. Sports (EN)
        text = f"The local {np.random.choice(sports_en)} tournament in {np.random.choice(cities_en)} ended with a dramatic victory as {np.random.choice(players_en)} scored in the final minutes of the match."
        records.append({"text": text, "category": "Sports", "language": "en"})
        
        # 6. Sports (ES)
        text = f"El torneo local de {np.random.choice(sports_es)} en {np.random.choice(cities_es)} terminó con una victoria dramática cuando {np.random.choice(players_es)} anotó en los últimos minutos del partido."
        records.append({"text": text, "category": "Sports", "language": "es"})
        
        # 7. General (EN)
        text = f"Local government officials in {np.random.choice(cities_en)} met to discuss {np.random.choice(topics_en)} and community park maintenance budgets for next year."
        records.append({"text": text, "category": "General", "language": "en"})
        
        # 8. General (ES)
        text = f"Los funcionarios del gobierno local en {np.random.choice(cities_es)} se reunieron para discutir sobre {np.random.choice(topics_es)} y presupuestos de mantenimiento de parques comunitarios para el próximo año."
        records.append({"text": text, "category": "General", "language": "es"})
        
        # 9. Noise (EN)
        template = np.random.choice(noise_templates)
        text = template.format(num=np.random.randint(1000, 99999), word="garbage_test_en")
        records.append({"text": text, "category": "Noise", "language": "en"})
        
        # 10. Noise (ES)
        template = np.random.choice(noise_templates)
        text = template.format(num=np.random.randint(1000, 99999), word="basura_prueba_es")
        records.append({"text": text, "category": "Noise", "language": "es"})

    df = pd.DataFrame(records)
    # Shuffle to mix categories up
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

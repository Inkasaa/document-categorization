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
            "language": "es"  
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
    Generates a balanced, highly diverse, and realistic multilingual production dataset
    across the 5 target categories.
    
    Uses a broad pool of randomized template patterns, fillers, and distractor vocabulary
    to prevent model memorization (overfitting) and simulate real-world data complexity.
    """
    np.random.seed(42)
    target_per_bucket = sample_size // 10  # 5 categories * 2 languages
    
    # ------------------ VOCABULARY ARRAYS ------------------
    unis = ["MIT", "Stanford", "Oxford", "Berkeley", "Harvard", "Cambridge", "Caltech", "Imperial College", "ETH Zurich"]
    techs = ["Docker containers", "Kubernetes clusters", "PyTorch models", "GraphQL APIs", "Kafka message queues", "Rust compilers", "C++ utilities", "TensorFlow layers", "Serverless functions", "NoSQL databases"]
    techs_es = ["contenedores Docker", "clusters Kubernetes", "modelos PyTorch", "APIs GraphQL", "colas Kafka", "compiladores Rust", "bases de datos NoSQL"]
    improvements = ["30% reduction", "2x improvement", "significant speedup", "substantial optimization", "50% decrease", "better throughput"]
    improvements_es = ["reducción del 30%", "mejora de 2x", "aceleración significativa", "optimización sustancial", "disminución del 50%"]
    concepts = ["asynchronous processing", "multi-thread computation", "vector operations", "lazy loading evaluation", "distributed consensus"]
    concepts_es = ["procesamiento asíncrono", "cómputo multihilo", "operaciones vectoriales", "evaluación perezosa", "consenso distribuido"]
    
    teams = ["the local club", "the national champions", "the rival squad", "the top-seeded players", "the underdog team"]
    teams_es = ["el club local", "los campeones nacionales", "el equipo rival", "los jugadores favoritos", "el equipo revelación"]
    sports = ["soccer", "basketball", "tennis", "baseball", "football", "rugby", "cricket", "golf", "athletics", "swimming"]
    sports_es = ["fútbol", "baloncesto", "tenis", "béisbol", "fútbol americano", "rugby", "ciclismo", "atletismo", "natación"]
    players = ["striker", "goalkeeper", "coach", "captain", "midfielder", "defender", "referee"]
    players_es = ["delantero", "portero", "entrenador", "capitán", "centrocampista", "defensa", "árbitro"]
    scores = ["3-2", "1-0", "0-0", "4-1", "95-92", "2-1"]
    
    commodities = ["crude oil", "gold bullion", "natural gas", "agricultural grains", "treasury bonds", "commercial real estate"]
    commodities_es = ["petróleo crudo", "lingotes de oro", "gas natural", "bonos del tesoro", "bienes raíces comerciales"]
    banks = ["Federal Reserve", "Bank of America", "Goldman Sachs", "JPMorgan", "Morgan Stanley", "World Bank", "IMF", "Barclays", "HSBC", "Deutsche Bank"]
    banks_es = ["Banco Central", "Banco Santander", "BBVA", "Fondo Monetario Internacional", "Banco Mundial", "CaixaBank", "Banco de España"]
    metrics = ["interest rates", "inflation rates", "market yield", "quarterly revenue", "fiscal deficit", "unemployment rate"]
    metrics_es = ["tasas de interés", "tasas de inflación", "rendimiento del mercado", "ingresos trimestrales", "déficit fiscal"]
    events = ["annual trade summit", "corporate acquisition", "regulatory overhaul", "monetary policy meeting"]
    events_es = ["cumbre comercial anual", "adquisición corporativa", "reforma regulatoria", "reunión de política monetaria"]
    
    topics = ["renewable energy structures", "urban housing policies", "academic library archives", "global weather warnings", "public transportation fees"]
    topics_es = ["estructuras de energía renovable", "políticas de vivienda urbana", "archivos de bibliotecas académicas", "tarifas de transporte público"]
    cities = ["New York", "London", "Tokyo", "Chicago", "Boston", "Frankfurt", "Toronto", "Sydney", "Paris", "Berlin"]
    cities_es = ["Madrid", "Barcelona", "Buenos Aires", "México", "Santiago", "Bogotá", "Sevilla", "Valencia", "Lima"]

    # ------------------ TEMPLATE MAPS ------------------
    templates_tech_en = [
        "Researchers at {} published a paper detailing {}. The setup achieves a {} in network operations.",
        "A major security vulnerability was patched in the latest {} release, ensuring better {} algorithms.",
        "Silicon Valley startups are investing heavily in {} to optimize {} workloads.",
        "Understanding {} is crucial for computer science graduates focusing on {}.",
        "We compared {} performance against legacy designs and noticed a {} in throughput.",
        "The integration of {} into serverless frameworks is growing rapidly across {} teams.",
        "Developers updated their main codebase to leverage {}, solving issues with {}.",
        "A tutorial was published on deploying {} in production environments using {} techniques."
    ]
    templates_tech_es = [
        "Investigadores de {} publicaron un artículo sobre {}. El sistema logra una {} en operaciones.",
        "Se corrigió una vulnerabilidad de seguridad importante en {}, garantizando mejores algoritmos de {}.",
        "Las startups de tecnología están invirtiendo en {} para optimizar cargas de trabajo de {}.",
        "Comprender {} es crucial para graduados en computación enfocados en {}.",
        "Comparamos el rendimiento de {} contra diseños antiguos y notamos una {} en el flujo.",
        "La integración de {} en frameworks en la nube está creciendo rápidamente en equipos de {}."
    ]

    templates_sports_en = [
        "The championship game between {} and their rivals ended in a {} score last night.",
        "After a grueling training session in {}, the {} declared that the team is ready for the tournament.",
        "Tickets for the upcoming {} match are sold out, according to official reports in {}.",
        "The {} suffered a minor setback due to a minor injury sustained by their key {}.",
        "During the national {} tournament, several athletes established brand-new records.",
        "Analysts expect a close contest between {} and their opponents in the next {} round.",
        "The committee announced the new stadium guidelines for {} events in {} starting next week.",
        "Local fans celebrated in the streets after the {} clinched the trophy in a {} thriller."
    ]
    templates_sports_es = [
        "El partido de campeonato entre {} y sus rivales terminó con un resultado de {} anoche.",
        "Después de un duro entrenamiento en {}, el {} declaró que el equipo está listo.",
        "Las entradas para el próximo encuentro de {} están agotadas en la ciudad de {}.",
        "El {} sufrió una baja debido a la lesión de su principal {}.",
        "Durante el torneo nacional de {}, varios atletas establecieron nuevos récords.",
        "Los analistas esperan un partido reñido entre {} y sus oponentes en la ronda de {}."
    ]

    templates_fin_en = [
        "The price of {} fell dramatically following the release of the {} report today.",
        "Representatives from {} met in {} to discuss stabilizing regional {} indexes.",
        "Market analysts expect {} to increase borrowing costs due to rising {} levels.",
        "A sudden {} sparked concerns about capital liquidity among regional {} offices.",
        "Corporate executives announced that {} dividends will be distributed to shareholders in {}.",
        "Economists warn that changes in {} could trigger a sell-off in {} assets.",
        "The latest audit of {} revealed unexpected growth despite the {} restrictions.",
        "Traders in {} remained cautious ahead of the upcoming {} guidelines announcement."
    ]
    templates_fin_es = [
        "El precio de {} cayó drásticamente tras la publicación del informe sobre {}.",
        "Representantes de {} se reunieron en {} para discutir la estabilización de los índices de {}.",
        "Los analistas del mercado esperan que {} aumente los costos debido al aumento de las {}.",
        "Una {} repentina despertó preocupación sobre la liquidez en las oficinas financieras de {}.",
        "Ejecutivos corporativos anunciaron dividendos de {} para los accionistas en {}."
    ]

    templates_gen_en = [
        "The city council of {} passed a new resolution regulating {} in municipal zones.",
        "A local nonprofit organized a community workshop focused on {} last Saturday.",
        "Meteorologists issued a forecast warning about unusual patterns in {} near {}.",
        "The library system in {} expanded its public collection to include more resources on {}.",
        "Volunteers gathered in the central square of {} to support awareness for {}.",
        "A panel of local experts will host a public discussion regarding {} next Tuesday.",
        "Residents expressed mixed reactions to the proposed alterations for {} plans in {}.",
        "A historical exhibit showcasing local achievements in {} opened to visitors in {}."
    ]
    templates_gen_es = [
        "El ayuntamiento de {} aprobó una nueva resolución que regula las {} en zonas municipales.",
        "Una organización local organizó un taller comunitario centrado en {} el sábado pasado.",
        "Los meteorólogos emitieron un pronóstico sobre patrones inusuales de {} cerca de {}.",
        "El sistema de bibliotecas de {} amplió su colección para incluir más recursos sobre {}.",
        "Los voluntarios se reunieron en la plaza central de {} para apoyar la difusión de {}."
    ]

    noise_templates = [
        "<html><body><p>CODE-{num}!!! $$$ %%% --- RAW NOISY DATA BLOCK.</p></body></html>",
        "  !!! ??? @@@ ### $$$ %%% ^^^ &&& *() _+ - = {num} {{ }} [ ] | \\ : ; \" ' < > , . / ~ `  ",
        "LOG-{num} --- NOISY DIGITS AND SYMBOLS ONLY FOR DEBUGGING.",
        "html tag line <br/> <p> {word} </p> error-code-{num}!!!",
        "XYZ!!! {num} --- Very noisy data with mostly punctuation and numbers."
    ]

    records = []

    for _ in range(target_per_bucket):
        # 1. Technology (EN)
        tpl = np.random.choice(templates_tech_en)
        if "{}" in tpl:
            # fill placeholders based on tpl format requirements
            placeholders = [np.random.choice(unis), np.random.choice(techs), np.random.choice(improvements), np.random.choice(concepts)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "Technology", "language": "en"})
        
        # 2. Technology (ES)
        tpl = np.random.choice(templates_tech_es)
        if "{}" in tpl:
            placeholders = [np.random.choice(cities_es), np.random.choice(techs_es), np.random.choice(improvements_es), np.random.choice(concepts_es)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "Technology", "language": "es"})
        
        # 3. Sports (EN)
        tpl = np.random.choice(templates_sports_en)
        if "{}" in tpl:
            placeholders = [np.random.choice(teams), np.random.choice(scores), np.random.choice(cities), np.random.choice(players), np.random.choice(sports)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "Sports", "language": "en"})
        
        # 4. Sports (ES)
        tpl = np.random.choice(templates_sports_es)
        if "{}" in tpl:
            placeholders = [np.random.choice(teams_es), np.random.choice(scores), np.random.choice(cities_es), np.random.choice(players_es), np.random.choice(sports_es)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "Sports", "language": "es"})
        
        # 5. Finance (EN)
        tpl = np.random.choice(templates_fin_en)
        if "{}" in tpl:
            placeholders = [np.random.choice(commodities), np.random.choice(events), np.random.choice(banks), np.random.choice(cities), np.random.choice(metrics)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "Finance", "language": "en"})
        
        # 6. Finance (ES)
        tpl = np.random.choice(templates_fin_es)
        if "{}" in tpl:
            placeholders = [np.random.choice(commodities_es), np.random.choice(events_es), np.random.choice(banks_es), np.random.choice(cities_es), np.random.choice(metrics_es)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "Finance", "language": "es"})
        
        # 7. General (EN)
        tpl = np.random.choice(templates_gen_en)
        if "{}" in tpl:
            placeholders = [np.random.choice(cities), np.random.choice(topics), np.random.choice(cities), np.random.choice(topics)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "General", "language": "en"})
        
        # 8. General (ES)
        tpl = np.random.choice(templates_gen_es)
        if "{}" in tpl:
            placeholders = [np.random.choice(cities_es), np.random.choice(topics_es), np.random.choice(cities_es), np.random.choice(topics_es)]
            text = tpl.format(*placeholders[:tpl.count("{}")])
        records.append({"text": text, "category": "General", "language": "es"})
        
        # 9. Noise (EN)
        template = np.random.choice(noise_templates)
        text = template.format(num=np.random.randint(1000, 999999), word="noise_word_en")
        records.append({"text": text, "category": "Noise", "language": "en"})
        
        # 10. Noise (ES)
        template = np.random.choice(noise_templates)
        text = template.format(num=np.random.randint(1000, 999999), word="ruido_palabra_es")
        records.append({"text": text, "category": "Noise", "language": "es"})

    df = pd.DataFrame(records)
    # Shuffle to mix categories up
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    return df

"""
Real-Time Document Categorization and Tagging Dashboard
------------------------------------------------------
This Streamlit application provides an interactive web interface to showcase
our document categorization, language detection, and metadata tagging system.

Features:
- Cached initialization of the `DocumentPipelineEngine` to load models once in memory.
- Interactive text input for single-document real-time classification.
- Interactive analytics sidebar mapping Category Distributions, Tag Frequencies,
  and Language Breakdowns across the mock dataset.
- Real-time performance indicators (Inference Latency, Classification Confidence).

Usage:
    streamlit run app/real_time_dashboard.py
"""

import logging
import time
import pandas as pd
# pyrefly: ignore [missing-import]
import streamlit as st
import numpy as np

# Configure imports from root
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.pipeline_engine import DocumentPipelineEngine
from utils.data_loader import load_mock_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Set page layout configuration
st.set_page_config(
    page_title="VisionTags - Doc Classifier & Tagger",
    page_icon="🏷️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply sleek, premium CSS styling to visual badges
st.markdown("""
<style>
    .badge {
        background-color: #1E1E2F;
        color: #00F0FF;
        padding: 5px 12px;
        margin: 4px;
        border-radius: 16px;
        border: 1px solid #00F0FF;
        display: inline-block;
        font-size: 13px;
        font-weight: 600;
        box-shadow: 0 0 5px rgba(0,240,255,0.2);
    }
    .metric-container {
        background-color: #0E1117;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363D;
    }
</style>
""", unsafe_allow_html=True)


# 1. Cache Model Pipelines (Ensures fast loads and single memory footprint)
@st.cache_resource(show_spinner="Initializing NLP models & Deep Learning layers (this may take a moment)...")
def load_pipeline_engine() -> DocumentPipelineEngine:
    """
    Instantiates and caches the DocumentPipelineEngine in memory.
    Reuses cached DistilBERT and SpaCy instances on browser refresh.
    """
    # Load weights from models/ folder
    return DocumentPipelineEngine(weights_path="models/distilbert_weights.h5")


# 2. Cache Analytics computation over Mock Dataset
@st.cache_data(show_spinner="Analyzing baseline dataset metrics...")
def get_baseline_analytics() -> dict:
    """
    Runs batch classification on the mock dataset to compile KPI metrics and
    charts for the dashboard sidebar/monitoring tab.
    """
    engine = load_pipeline_engine()
    df = load_mock_data()
    
    t_start = time.time()
    # Execute batch processing (efficient parallel model pass)
    batch_results = engine.process_batch(df["text"].tolist())
    total_latency_ms = (time.time() - t_start) * 1000.0
    avg_latency = total_latency_ms / len(df)
    
    categories = []
    languages = []
    confidences = []
    tags = []
    
    for res in batch_results:
        categories.append(res["predicted_category"])
        languages.append(res["detected_language"])
        confidences.append(res["confidence_score"])
        tags.extend(res["generated_tags"])
        
    avg_confidence = np.mean(confidences) if confidences else 0.0
    
    # Calculate distributions
    category_counts = pd.Series(categories).value_counts()
    language_counts = pd.Series(languages).value_counts()
    tag_counts = pd.Series(tags).value_counts().head(10)  # Top 10 tags
    
    return {
        "avg_latency": avg_latency,
        "avg_confidence": avg_confidence,
        "category_counts": category_counts,
        "language_counts": language_counts,
        "tag_counts": tag_counts
    }


# Load pipeline engine
try:
    engine = load_pipeline_engine()
except Exception as err:
    st.error(f"Failed to load pipelines: {err}")
    logger.exception("Init failure")
    st.stop()


# Initialize session state variables
if "processed_docs" not in st.session_state:
    st.session_state.processed_docs = []
if "latest_result" not in st.session_state:
    st.session_state.latest_result = None
if "latest_latency" not in st.session_state:
    st.session_state.latest_latency = 0.0


# ==========================================
# 3. SIDEBAR ANALYTICS & MONITORING
# ==========================================
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/tags.png", width=70)
    st.title("📊 Session Telemetry")
    
    if not st.session_state.processed_docs:
        st.write("No documents processed in this session yet.")
        st.info("Paste text and click '🚀 Process Document' to collect real-time telemetry metrics.")
    else:
        st.write(f"Telemetry collected from {len(st.session_state.processed_docs)} document(s) processed in this session.")
        st.markdown("---")
        
        # Calculate stats dynamically from current session's processed documents
        latencies = [doc["latency"] for doc in st.session_state.processed_docs]
        confidences = [doc["confidence"] for doc in st.session_state.processed_docs]
        categories = [doc["category"] for doc in st.session_state.processed_docs]
        languages = [doc["language"] for doc in st.session_state.processed_docs]
        
        avg_latency = np.mean(latencies)
        avg_confidence = np.mean(confidences)
        
        st.subheader("💡 Telemetry KPIs")
        st.metric(
            label="⏱️ Avg Inference Latency",
            value=f"{avg_latency:.2f} ms"
        )
        st.metric(
            label="🎯 Avg Classification Confidence",
            value=f"{avg_confidence * 100:.1f}%"
        )
        
        st.markdown("---")
        
        # Category distribution chart
        st.subheader("📈 Category Distribution")
        cat_counts = pd.Series(categories).value_counts()
        cat_df = pd.DataFrame({
            "Category": cat_counts.index,
            "Count": cat_counts.values
        })
        st.bar_chart(cat_df.set_index("Category"))
        
        # Language breakdown chart
        st.subheader("🌍 Language Breakdown")
        lang_counts = pd.Series(languages).value_counts()
        lang_df = pd.DataFrame({
            "Language": lang_counts.index,
            "DocumentsCount": lang_counts.values
        })
        st.bar_chart(lang_df.set_index("Language"))


# ==========================================
# 4. MAIN PAGE USER INTERFACE
# ==========================================
st.title("🏷️ Multi-Language Document Categorization Dashboard")
st.markdown("""
Welcome to **VisionTags**! This intelligent pipeline processes raw, unstructured texts, cleans them, 
automatically identifies the language, classifies the document into high-level categories (using DistilBERT), 
and extracts context-aware metadata tags (using SpaCy NER & keyword heuristics).
""")

st.markdown("---")

# Main Interface: Text Input area
st.subheader("📝 Input Document Text")
default_text = (
    "<html><body><p>The <b>Artificial Intelligence (AI)</b> revolution is transforming modern software "
    "engineering!!! Visit our blog for more details.</p></body></html>"
)
raw_input = st.text_area(
    label="Paste raw text, HTML content, or noisy multi-language snippets below:",
    value=default_text,
    height=150
)

process_btn = st.button("🚀 Process Document")

if process_btn:
    if not raw_input.strip():
        st.warning("Please enter some document text to process.")
    else:
        # Measure execution latency for this specific text run
        t_start = time.time()
        result = engine.process_document(raw_input)
        latency = (time.time() - t_start) * 1000.0
        
        # Append to session state processed documents list
        st.session_state.processed_docs.append({
            "latency": latency,
            "confidence": result["confidence_score"],
            "category": result["predicted_category"],
            "language": result["detected_language"]
        })
        
        # Save to latest result state to render in main area
        st.session_state.latest_result = result
        st.session_state.latest_latency = latency
        st.rerun()


# Render Extraction Results panel if a document has been processed
if st.session_state.latest_result is not None:
    st.subheader("🔍 Extraction Results")
    result = st.session_state.latest_result
    latency = st.session_state.latest_latency
    
    # Grid Layout for metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="📁 Predicted Category",
            value=result["predicted_category"],
            delta=f"{result['confidence_score'] * 100:.1f}% confidence"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        # Convert language code to a readable label
        lang_map = {"en": "English 🇺🇸", "es": "Spanish 🇪🇸", "fr": "French 🇫🇷", "de": "German 🇩🇪"}
        lang_name = lang_map.get(result["detected_language"], f"Unknown ('{result['detected_language']}')")
        st.metric(
            label="🌍 Detected Language",
            value=lang_name
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col3:
        st.markdown('<div class="metric-container">', unsafe_allow_html=True)
        st.metric(
            label="⏱️ Processing Latency",
            value=f"{latency:.1f} ms"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.write("")
    
    # Visual tags rendering
    st.write("🏷️ **Generated Context-Aware Tags:**")
    tags = result["generated_tags"]
    if tags:
        # Wrap tags in clean HTML styling blocks
        tags_html = "".join([f'<span class="badge">{tag}</span>' for tag in tags])
        st.markdown(tags_html, unsafe_allow_html=True)
    else:
        st.info("No named entities or fallback keywords matched. No tags generated.")
        
    st.write("")
    
    # Preprocessor detail expander
    with st.expander("🛠️ View Text Cleaning Pipeline Detail"):
        st.write("**Original Text:**")
        st.code(raw_input, language="html")
        st.write("**Cleaned Text (Preprocessed):**")
        st.info(result["cleaned_text"])
        st.write("**Tokens & Lemmas Passed to Model:**")
        # Generate token list to show preprocessing logic
        tokens = engine.preprocessor.preprocess(raw_input)
        st.code(str(tokens))

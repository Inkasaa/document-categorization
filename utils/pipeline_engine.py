"""
Pipeline Integration Engine
---------------------------
This module implements the Phase 4 Pipeline Integration & Optimization.
It contains the `DocumentPipelineEngine` class which acts as the unified orchestrator:
1. Sequentially chains raw text cleaning, language detection, DistilBERT sequence
   classification, and SpaCy NER/rule-based tagging.
2. Implements a highly efficient `process_batch()` method that processes multiple
   documents by tokenizing and executing model inference in a single parallel batch,
   dramatically improving inference throughput and reducing overhead.
3. Implements weight loading error handling, falling back gracefully to the
   un-fine-tuned base multilingual model if saved checkpoints are not found.
4. Caches preprocessor, tagger, classifier, and label encoder instances in memory.

Modular, heavily commented, and self-contained for educational walkthrough.
"""

import logging
import os
from typing import Dict, List, Union
import numpy as np
import tensorflow as tf

from models.tagger import DocumentTagger
from models.text_classifier import DistilBertClassifier
from utils.text_preprocessing import TextPreprocessor
from utils.transfer_learning import LabelEncoder, tokenize_texts

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class DocumentPipelineEngine:
    """
    Unified Orchestrator and Caching Engine for document preprocessing,
    classification, and metadata tagging.
    """

    def __init__(self, weights_path: str = "models/distilbert_weights.h5"):
        """
        Initializes and caches all core NLP models and ML pipelines.
        
        Args:
            weights_path (str): Relative or absolute path to the fine-tuned classifier weights.
        """
        logger.info("Initializing unified DocumentPipelineEngine...")

        # 1. Instantiate the preprocessor (shared cache resource)
        self.preprocessor = TextPreprocessor(default_lang="en")

        # 2. Instantiate the metadata tagger (reusing the preprocessor's loaded SpaCy pipeline cache)
        self.tagger = DocumentTagger(preprocessor=self.preprocessor)

        # 3. Fit the LabelEncoder with our standard categories
        self.label_encoder = LabelEncoder()
        # Default categories mapped during project design
        self.label_encoder.fit(["Technology", "Finance", "Sports", "General", "Noise"])
        num_classes = len(self.label_encoder.classes_)

        # 4. Instantiate the deep learning sequence classifier
        self.classifier = DistilBertClassifier(num_classes=num_classes)

        # 5. Robust weight-loading error handling
        if weights_path and os.path.exists(weights_path):
            try:
                self.classifier.load_weights(weights_path)
                logger.info(f"Successfully loaded fine-tuned weights from '{weights_path}'")
            except Exception as e:
                logger.error(
                    f"Error loading weights from '{weights_path}': {e}. "
                    f"Gracefully falling back to the un-fine-tuned base multilingual model."
                )
        else:
            logger.warning(
                f"Weights file '{weights_path}' not found. "
                f"Gracefully falling back to the un-fine-tuned base multilingual model."
            )

        logger.info("DocumentPipelineEngine initialization complete.")

    def process_document(self, raw_text: str) -> Dict[str, Union[str, float, List[str]]]:
        """
        Runs the complete text classification and tagging pipeline for a single document.

        Args:
            raw_text (str): Raw input document string.

        Returns:
            Dict: Output fields containing:
                  - 'cleaned_text': Stripped and normalized text.
                  - 'detected_language': ISO 2-letter language code.
                  - 'predicted_category': Classification label.
                  - 'confidence_score': Float between 0.0 and 1.0.
                  - 'generated_tags': List of extracted metadata tags.
        """
        if not raw_text or not isinstance(raw_text, str):
            return {
                "cleaned_text": "",
                "detected_language": self.preprocessor.default_lang,
                "predicted_category": "Unknown",
                "confidence_score": 0.0,
                "generated_tags": []
            }

        # Step A: Preprocess text (clean + language detection)
        cleaned_text = self.preprocessor.clean_raw_text(raw_text)
        detected_lang = self.preprocessor.detect_language(cleaned_text)

        # Step B: Classify text using DistilBERT model
        classification = self.classifier.predict(
            cleaned_text,
            preprocessor=self.preprocessor,
            label_encoder=self.label_encoder
        )

        # Step C: Tag text using SpaCy NER and keyword rules
        tags = self.tagger.generate_tags(cleaned_text, detected_lang)

        return {
            "cleaned_text": cleaned_text,
            "detected_language": detected_lang,
            "predicted_category": classification["category"],
            "confidence_score": classification["confidence"],
            "generated_tags": tags
        }

    def process_batch(self, list_of_texts: List[str]) -> List[Dict[str, Union[str, float, List[str]]]]:
        """
        Processes a list of raw document texts in a single optimized parallel batch.
        Tokenizes and executes the DistilBERT neural network forward pass in one step,
        preventing loop overhead and maximizing GPU/CPU vectorization.

        Args:
            list_of_texts (List[str]): List of raw document strings.

        Returns:
            List[Dict]: List of output metadata dictionaries for each text.
        """
        if not list_of_texts:
            return []

        logger.info(f"Starting batch processing of {len(list_of_texts)} documents...")

        cleaned_texts: List[str] = []
        languages: List[str] = []

        # Step A: Perform preprocessing (CPU-bound string parsing and language detection)
        for raw in list_of_texts:
            cleaned = self.preprocessor.clean_raw_text(raw)
            cleaned_texts.append(cleaned)
            
            # Grabbing language code
            if cleaned:
                lang = self.preprocessor.detect_language(cleaned)
            else:
                lang = self.preprocessor.default_lang
            languages.append(lang)

        # Step B: Batch Tokenize all cleaned texts simultaneously (maximizes tokenizer throughput)
        tokenized_batch = tokenize_texts(cleaned_texts, model_name=self.classifier.model_name)
        model_inputs = {
            "input_ids": tokenized_batch["input_ids"],
            "attention_mask": tokenized_batch["attention_mask"]
        }

        # Step C: Run single forward pass on the DistilBERT batch inputs (parallel matrix multiplication)
        logger.info("Executing batch inference on DistilBERT model...")
        outputs = self.classifier.model(model_inputs, training=False)
        logits = outputs.logits
        # Convert raw logits to probabilities using Softmax activation
        batch_probs = tf.nn.softmax(logits, axis=-1).numpy()

        results: List[Dict[str, Union[str, float, List[str]]]] = []

        # Step D: Extract predicted category index and generate metadata tags for each text in loop
        for idx in range(len(list_of_texts)):
            cleaned = cleaned_texts[idx]
            lang = languages[idx]

            # Graceful safety check if text was empty
            if not cleaned:
                pred_category = "Unknown"
                confidence = 0.0
                tags = []
            else:
                # Extract predicted label and confidence score
                probs = batch_probs[idx]
                pred_idx = int(np.argmax(probs))
                confidence = float(probs[pred_idx])
                pred_category = self.label_encoder.inverse_transform([pred_idx])[0]

                # Generate tags using the SpaCy-based DocumentTagger (cached SpaCy model)
                tags = self.tagger.generate_tags(cleaned, lang)

            results.append({
                "cleaned_text": cleaned,
                "detected_language": lang,
                "predicted_category": pred_category,
                "confidence_score": confidence,
                "generated_tags": tags
            })

        logger.info("Batch processing complete.")
        return results


# Self-test block using mock dataset to verify pipeline execution
if __name__ == "__main__":
    from utils.data_loader import load_mock_data

    print("\n" + "=" * 80)
    print("PHASE 4: PIPELINE ENGINE END-TO-END VERIFICATION")
    print("=" * 80)

    # 1. Load mock data
    print("\n[Step 1] Loading mock dataset...")
    df = load_mock_data()
    raw_texts = df["text"].tolist()

    # 2. Test Weight Loading Error Fallback (load with invalid weights path)
    print("\n[Step 2] Testing weight loading error fallback...")
    invalid_engine = DocumentPipelineEngine(weights_path="models/invalid_path_weights.h5")

    # 3. Initialize Engine with Valid Weights (or fallback to base model)
    print("\n[Step 3] Initializing DocumentPipelineEngine with standard weights...")
    engine = DocumentPipelineEngine(weights_path="models/distilbert_weights.h5")

    # 4. Run Batch Processing
    print("\n[Step 4] Running batch processing over mock dataset...")
    batch_results = engine.process_batch(raw_texts)

    # 5. Output Batch Results
    print("\n[Step 5] Displaying processed batch results:")
    for idx, res in enumerate(batch_results):
        print("-" * 80)
        print(f"Document #{idx + 1}")
        print(f"  Detected Lang : {res['detected_language']}")
        print(f"  Category      : {res['predicted_category']} (Conf: {res['confidence_score']:.4f})")
        print(f"  Cleaned Text  : {res['cleaned_text']}")
        print(f"  Generated Tags: {res['generated_tags']}")
    print("-" * 80)

    print("\nVerification pipeline completed successfully!")
    print("=" * 80)

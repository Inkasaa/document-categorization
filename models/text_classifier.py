"""
Text Classifier Model (DistilBERT)
----------------------------------
This module implements the Phase 3 Sequence Classification pipeline using TensorFlow/Keras
and Hugging Face's DistilBERT transformer.

It includes:
1. `DistilBertClassifier` class wrapper for sequence classification tasks.
2. Fine-tuning method compiled with Adam optimizer and Sparse Categorical Crossentropy.
3. Inference method returning categorical predictions and confidence scores.
4. Self-testing execution block simulating Phase 3 workflow.

Highly modular and heavily commented for educational walkthrough.
"""

import logging
import os
from typing import Dict, List, Union
import numpy as np
import tensorflow as tf

# pyrefly: ignore [missing-import]
from transformers import TFDistilBertForSequenceClassification

from utils.text_preprocessing import TextPreprocessor
from utils.transfer_learning import LabelEncoder, tokenize_texts

logger = logging.getLogger(__name__)


class DistilBertClassifier:
    """
    A deep learning text classifier that wraps TFDistilBertForSequenceClassification.
    Provides standard high-level APIs for fine-tuning and inference.
    """

    def __init__(self, num_classes: int, model_name: str = "distilbert-base-multilingual-cased"):
        """
        Initializes the classifier and loads the pretrained DistilBERT sequence classification model.

        Args:
            num_classes (int): Number of target categorization classes.
            model_name (str): Hugging Face pre-trained model identifier.
        """
        self.num_classes = num_classes
        self.model_name = model_name

        logger.info(f"Loading pretrained TensorFlow model '{model_name}' for {num_classes} classes...")
        # Load sequence classification model with the corresponding number of output labels
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            use_safetensors=False
        )
        logger.info("Model loaded successfully.")

    def compile_model(self, learning_rate: float = 2e-5):
        """
        Compiles the model with the Adam optimizer and Sparse Categorical Crossentropy loss.
        
        Args:
            learning_rate (float): Small learning rate typical for fine-tuning transformer weights.
        """
        logger.info(f"Compiling Keras model with Adam optimizer (learning_rate={learning_rate})...")
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # We use from_logits=True because TFDistilBertForSequenceClassification outputs raw logits
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        logger.info("Model compilation complete.")

    def train(
        self,
        train_inputs: Dict[str, np.ndarray],
        train_labels: np.ndarray,
        epochs: int = 2,
        batch_size: int = 2,
        save_dir: str = "models"
    ):
        """
        Trains the classifier on tokenized inputs and labels, then saves model weights.

        Args:
            train_inputs (Dict[str, np.ndarray]): Dict containing 'input_ids' and 'attention_mask'.
            train_labels (np.ndarray): Target category indices (NumPy array).
            epochs (int): Number of training iterations.
            batch_size (int): Batch size used for gradient steps.
            save_dir (str): Relative directory path to save model weights.
        """
        logger.info(f"Starting fine-tuning for {epochs} epochs (batch_size={batch_size})...")
        
        # Prepare inputs as a dictionary of TF Tensors or Keras-friendly NumPy arrays
        inputs = {
            "input_ids": train_inputs["input_ids"],
            "attention_mask": train_inputs["attention_mask"]
        }

        # Execute training loop via standard Keras model.fit()
        self.model.fit(
            inputs,
            train_labels,
            epochs=epochs,
            batch_size=batch_size,
            verbose=1
        )
        logger.info("Training complete.")

        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        weights_path = os.path.join(save_dir, "distilbert_weights.h5")
        
        logger.info(f"Saving model weights to '{weights_path}'...")
        # Note: save_weights saves only the weight layers, which is highly space-efficient.
        # Alternatively, self.model.save_pretrained(save_dir) can be used to save Hugging Face format.
        self.model.save_weights(weights_path)
        logger.info("Weights saved successfully.")

    def load_weights(self, weights_path: str):
        """
        Loads pre-trained fine-tuned weights back into the model structure.

        Args:
            weights_path (str): File path to weights file (e.g. .h5 file).
        """
        logger.info(f"Loading weights from '{weights_path}'...")
        self.model.load_weights(weights_path)
        logger.info("Weights loaded successfully.")

    def predict(
        self,
        text: str,
        preprocessor: TextPreprocessor,
        label_encoder: LabelEncoder,
        max_length: int = 128
    ) -> Dict[str, Union[str, float]]:
        """
        Predicts the categorization class and confidence score for a single raw text string.

        Args:
            text (str): Raw input document.
            preprocessor (TextPreprocessor): Text cleaning utility.
            label_encoder (LabelEncoder): Label mapper to decode index back to category name.
            max_length (int): Tokenizer max length limit.

        Returns:
            Dict[str, Union[str, float]]: Prediction outputs with category and confidence fields.
        """
        # Step 1: Clean raw input text
        cleaned_text = preprocessor.clean_raw_text(text)
        if not cleaned_text:
            return {"category": "Unknown", "confidence": 0.0}

        # Step 2: Tokenize clean text
        tokenized = tokenize_texts([cleaned_text], model_name=self.model_name, max_length=max_length)
        inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

        # Step 3: Run inference
        # We wrap in tf.device or run directly. TensorFlow handles inference on CPU/GPU automatically.
        outputs = self.model(inputs, training=False)
        logits = outputs.logits

        # Step 4: Convert outputs to probabilities using Softmax activation
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        # Step 5: Extract argmax index and decode class label name
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_category = label_encoder.inverse_transform([pred_idx])[0]

        return {
            "category": predicted_category,
            "confidence": confidence
        }


# Self-test block using mock dataset to verify execution without crashes
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    from utils.data_loader import load_mock_data

    print("\n" + "=" * 80)
    print("PHASE 3 SELF-TEST & VERIFICATION RUN")
    print("=" * 80)

    # 1. Load mock data
    print("\n[Step 1] Loading dataset...")
    df = load_mock_data()
    
    # 2. Preprocess texts
    print("\n[Step 2] Cleaning texts using TextPreprocessor...")
    preprocessor = TextPreprocessor()
    df["cleaned_text"] = df["text"].apply(preprocessor.clean_raw_text)
    print(df[["cleaned_text", "true_category"]])

    # 3. Label encode target category
    print("\n[Step 3] Fitting LabelEncoder...")
    encoder = LabelEncoder()
    labels = df["true_category"].tolist()
    encoded_labels = encoder.fit_transform(labels)
    num_classes = len(encoder.classes_)

    # 4. Tokenize texts
    print("\n[Step 4] Tokenizing clean texts...")
    tokenized_data = tokenize_texts(df["cleaned_text"].tolist(), max_length=64)

    # 5. Initialize and compile DistilBertClassifier
    print("\n[Step 5] Initializing DistilBertClassifier...")
    classifier = DistilBertClassifier(num_classes=num_classes)
    classifier.compile_model(learning_rate=3e-5)

    # 6. Run training for 1-2 epochs (small test run)
    print("\n[Step 6] Running verification training loop (2 epochs, batch_size=2)...")
    classifier.train(
        train_inputs=tokenized_data,
        train_labels=encoded_labels,
        epochs=2,
        batch_size=2
    )

    # 7. Test single inference prediction
    print("\n[Step 7] Testing inference predict() method...")
    test_text = "<html><body><p>Highly advanced AI and software engineering concepts are discussed here!</p></body></html>"
    pred_result = classifier.predict(test_text, preprocessor, encoder)
    print(f"Input Text:  {test_text}")
    print(f"Prediction:  {pred_result}")
    print("\nVerification pipeline completed successfully!")
    print("=" * 80)

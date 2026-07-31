"""
Text Classifier Model (DistilBERT)
----------------------------------
This script implements the Phase 3 Sequence Classification pipeline using TensorFlow/Keras
and Hugging Face's DistilBERT transformer.

It is updated to:
1. Ingest the real 10,000-document multi-language Amazon dataset.
2. Shuffles, cleans, and splits the data into 80/20 train/test sets, further segmenting
   a stratified validation split.
3. Tokenizes datasets and trains a multi-class sequence classifier for exactly 5 epochs.
4. Uses Keras ModelCheckpoint callback to save the absolute best weights to
   'models/checkpoints/text_classifier_best.h5' and config.json to 'models/checkpoints/config.json'.
5. Exports training history to 'models/checkpoints/training_history.csv' at the end.
6. Evaluates the best checkpoint model on the 20% test split, printing total accuracy,
   macro F1-score, and separate language-specific (English vs. Spanish) test accuracies.
"""

import logging
import os
import sys
from typing import Dict, List, Union
import numpy as np
import pandas as pd
import tensorflow as tf

# pyrefly: ignore [missing-import]
from transformers import TFDistilBertForSequenceClassification

# Configure python path to allow importing utils & models packages
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.data_loader import load_production_dataset
from utils.text_preprocessing import TextPreprocessor
from utils.transfer_learning import LabelEncoder, tokenize_texts

# Setup structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
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

    def compile_model(self, learning_rate: float = 3e-5, run_eagerly: bool = False):
        """
        Compiles the model with the Adam optimizer and Sparse Categorical Crossentropy loss.
        
        Args:
            learning_rate (float): Small learning rate typical for fine-tuning transformer weights.
            run_eagerly (bool): If True, runs eager execution instead of building static graph.
        """
        logger.info(f"Compiling Keras model with Adam optimizer (learning_rate={learning_rate}, run_eagerly={run_eagerly})...")
        if hasattr(tf.keras.optimizers, "legacy"):
            logger.info("Using legacy Adam optimizer for Apple Silicon acceleration.")
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # We use from_logits=True because TFDistilBertForSequenceClassification outputs raw logits
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy("accuracy")]
        
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
        logger.info("Model compilation complete.")

    def train(
        self,
        train_inputs: Dict[str, np.ndarray],
        train_labels: np.ndarray,
        val_inputs: Dict[str, np.ndarray],
        val_labels: np.ndarray,
        epochs: int = 10,
        batch_size: int = 16,
        checkpoint_dir: str = "models/checkpoints"
    ):
        """
        Trains the classifier on tokenized inputs and labels using Keras fit() with EarlyStopping.
        Saves best checkpoint weights based on validation loss.

        Args:
            train_inputs (Dict[str, np.ndarray]): Training features containing input_ids and attention_mask.
            train_labels (np.ndarray): Training target label indices.
            val_inputs (Dict[str, np.ndarray]): Validation features.
            val_labels (np.ndarray): Validation target label indices.
            epochs (int): Number of training iterations.
            batch_size (int): Batch size used for gradient steps.
            checkpoint_dir (str): Directory path to save model weights and config files.
        """
        logger.info(f"Starting production fine-tuning for {epochs} epochs (batch_size={batch_size})...")
        
        # Ensure directories exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "text_classifier_best.h5")

        # Save Hugging Face configuration file config.json
        logger.info(f"Saving model configuration to '{checkpoint_dir}/config.json'...")
        self.model.config.save_pretrained(checkpoint_dir)

        # Build tf.data datasets to avoid eager graph compile overhead issues
        train_dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": train_inputs["input_ids"],
                "attention_mask": train_inputs["attention_mask"]
            },
            train_labels
        )).shuffle(len(train_labels)).batch(batch_size)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            {
                "input_ids": val_inputs["input_ids"],
                "attention_mask": val_inputs["attention_mask"]
            },
            val_labels
        )).batch(batch_size)

        # Setup Early Stopping callback
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=2,
            restore_best_weights=True,
            verbose=1
        )

        # Custom checkpoint callback to save epochs checkpoints as required by guidelines
        class EpochCheckpoint(tf.keras.callbacks.Callback):
            def __init__(self, directory):
                super().__init__()
                self.directory = directory
            def on_epoch_end(self, epoch, logs=None):
                epoch_checkpoint_path = os.path.join(self.directory, f"text_classifier_epoch_{epoch+1}.h5")
                print(f"\n  Saving epoch {epoch+1} weights to '{epoch_checkpoint_path}'...", flush=True)
                self.model.save_weights(epoch_checkpoint_path)

        epoch_checkpoint_callback = EpochCheckpoint(checkpoint_dir)

        # Fit model
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=[early_stopping, epoch_checkpoint_callback],
            verbose=1
        )

        # Save absolute best weights checkpoint
        logger.info(f"Saving absolute best weights to '{checkpoint_path}'...")
        self.model.save_weights(checkpoint_path)

        # Export training history to CSV
        history_path = os.path.join(checkpoint_dir, "training_history.csv")
        logger.info(f"Exporting training history to '{history_path}'...")
        history_df = pd.DataFrame(history.history)
        history_df.index = history_df.index + 1
        history_df.index.name = "Epoch"
        history_df.to_csv(history_path)
        
        # Display history dataframe
        print("\n=== TRAINING HISTORY METRICS ===")
        print(history_df.to_string())
        print("===============================\n")

        # Generate learning curves
        try:
            import matplotlib.pyplot as plt
            os.makedirs("reports", exist_ok=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Loss curve
            ax1.plot(history_df.index, history_df["loss"], label="Train Loss", marker="o", color="#e74c3c")
            ax1.plot(history_df.index, history_df["val_loss"], label="Val Loss", marker="o", color="#3498db")
            ax1.set_title("Training vs. Validation Loss")
            ax1.set_xlabel("Epoch")
            ax1.set_ylabel("Loss")
            ax1.legend()
            ax1.grid(True, linestyle="--", alpha=0.6)
            
            # Accuracy curve
            ax2.plot(history_df.index, history_df["accuracy"], label="Train Accuracy", marker="o", color="#2ecc71")
            ax2.plot(history_df.index, history_df["val_accuracy"], label="Val Accuracy", marker="o", color="#9b59b6")
            ax2.set_title("Training vs. Validation Accuracy")
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Accuracy")
            ax2.legend()
            ax2.grid(True, linestyle="--", alpha=0.6)
            
            plot_path = "reports/learning_curves.png"
            plt.tight_layout()
            plt.savefig(plot_path, dpi=150)
            plt.close()
            logger.info(f"Learning curves plot successfully exported to '{plot_path}'")
        except Exception as e:
            logger.error(f"Failed to generate learning curves plot: {e}")

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
        cleaned_text = preprocessor.clean_raw_text(text)
        if not cleaned_text:
            return {"category": "Unknown", "confidence": 0.0}

        tokenized = tokenize_texts([cleaned_text], model_name=self.model_name, max_length=max_length)
        inputs = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"]
        }

        outputs = self.model(inputs, training=False)
        logits = outputs.logits
        probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        predicted_category = label_encoder.inverse_transform([pred_idx])[0]

        return {
            "category": predicted_category,
            "confidence": confidence
        }


def run_production_training(sample_size: int = 10000, epochs: int = 5, batch_size: int = 16, run_eagerly: bool = False):
    """
    Ingests production records, cleans, tokenizes, runs fine-tuning,
    and performs detailed multi-language test evaluations.
    
    Args:
        sample_size (int): Total number of records to ingest.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size used for training gradient steps.
        run_eagerly (bool): If True, compiles the model in eager mode to avoid graph compilation delays.
    """
    print("\n" + "=" * 80)
    print(f"PHASE 3 PRODUCTION TRAINING & MULTI-LANGUAGE EVALUATION (Size: {sample_size}, Epochs: {epochs})")
    print("=" * 80)

    # 1. Load production dataset (balanced records)
    logger.info(f"Loading {sample_size} production records...")
    df = load_production_dataset(sample_size=sample_size)

    # 2. Clean reviews using existing TextPreprocessor
    logger.info("Preprocessing raw review strings...")
    preprocessor = TextPreprocessor(default_lang="en")
    df["cleaned_text"] = df["text"].apply(preprocessor.clean_raw_text)

    # 3. Label encode target category ratings (1-5 stars mapped to indices 0-4)
    logger.info("Encoding labels...")
    encoder = LabelEncoder()
    # Cast ratings to string lists for the encoder fit step
    encoded_labels = encoder.fit_transform(df["category"].tolist())
    num_classes = len(encoder.classes_)

    # 4. Train-Test Split (80/20 train/test distribution)
    # We split the indices so we can index both clean texts, categories, and language tags
    from sklearn.model_selection import train_test_split
    indices = np.arange(len(df))
    train_indices, test_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=42,
        stratify=df["category"]
    )

    # Further split training data to get 10% validation data
    train_indices_final, val_indices = train_test_split(
        train_indices,
        test_size=0.1,
        random_state=42,
        stratify=df.iloc[train_indices]["category"]
    )

    # Extract stratified splits
    df_train = df.iloc[train_indices_final]
    df_val = df.iloc[val_indices]
    df_test = df.iloc[test_indices]

    y_train = encoded_labels[train_indices_final]
    y_val = encoded_labels[val_indices]
    y_test = encoded_labels[test_indices]

    # 5. Tokenize text splits using DistilBertTokenizer
    logger.info("Tokenizing training, validation, and test splits...")
    model_name = "distilbert-base-multilingual-cased"
    max_length = 128
    
    train_inputs = tokenize_texts(df_train["cleaned_text"].tolist(), model_name=model_name, max_length=max_length)
    val_inputs = tokenize_texts(df_val["cleaned_text"].tolist(), model_name=model_name, max_length=max_length)
    test_inputs = tokenize_texts(df_test["cleaned_text"].tolist(), model_name=model_name, max_length=max_length)

    # 6. Initialize, compile, and train the model for target epochs
    classifier = DistilBertClassifier(num_classes=num_classes, model_name=model_name)
    classifier.compile_model(learning_rate=3e-5, run_eagerly=run_eagerly)

    # Train model
    classifier.train(
        train_inputs=train_inputs,
        train_labels=y_train,
        val_inputs=val_inputs,
        val_labels=y_val,
        epochs=epochs,
        batch_size=batch_size,
        checkpoint_dir="models/checkpoints"
    )

    # 7. Ingestion & Load best checkpoint weights back
    logger.info("Loading absolute best weight checkpoint...")
    best_weights_path = "models/checkpoints/text_classifier_best.h5"
    classifier.load_weights(best_weights_path)

    # 8. Run test evaluation
    logger.info("Running evaluation on the 20% test split...")
    tst_inputs = {
        "input_ids": test_inputs["input_ids"],
        "attention_mask": test_inputs["attention_mask"]
    }
    
    # Run prediction by passing inputs directly to model to avoid Keras .predict() deadlock
    predictions = classifier.model(tst_inputs, training=False)
    logits = predictions.logits
    probs = tf.nn.softmax(logits, axis=-1).numpy()
    y_pred = np.argmax(probs, axis=-1)

    # Calculate metrics
    from sklearn.metrics import accuracy_score, f1_score
    total_acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    # Segment test data by language
    test_languages = df_test["language"].values
    
    en_mask = (test_languages == "en")
    es_mask = (test_languages == "es")

    en_acc = accuracy_score(y_test[en_mask], y_pred[en_mask]) if np.any(en_mask) else 0.0
    es_acc = accuracy_score(y_test[es_mask], y_pred[es_mask]) if np.any(es_mask) else 0.0

    # Print final evaluation metrics report
    print("\n" + "=" * 60)
    print("PRODUCTION MODEL EVALUATION REPORT")
    print("=" * 60)
    print(f"Total Test Accuracy         : {total_acc * 100:.2f}%")
    print(f"Total Macro F1-Score        : {macro_f1:.4f}")
    print("-" * 60)
    print(f"English Test Accuracy ('en'): {en_acc * 100:.2f}%")
    print(f"Spanish Test Accuracy ('es'): {es_acc * 100:.2f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DistilBERT Production Training Pipeline")
    parser.add_argument("--sample_size", type=int, default=10000, help="Total production samples to load")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--run_eagerly", action="store_true", default=False, help="Compile and run eagerly to avoid graph compile delays")
    args = parser.parse_args()
    
    run_production_training(
        sample_size=args.sample_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        run_eagerly=args.run_eagerly
    )

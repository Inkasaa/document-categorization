"""
Transfer Learning Utilities
--------------------------
This module handles data preparation for Hugging Face transformer models.
It includes:
1. Multi-language tokenization using Hugging Face's `DistilBertTokenizer`.
2. A custom categorical `LabelEncoder` for target categories (e.g. Technology, Finance).

Designed to bridge raw preprocessed texts with TensorFlow deep learning models.
"""

import logging
from typing import Dict, List, Union
import numpy as np

# pyrefly: ignore [missing-import]
from transformers import DistilBertTokenizer

logger = logging.getLogger(__name__)


def tokenize_texts(
    texts: List[str],
    model_name: str = "distilbert-base-multilingual-cased",
    max_length: int = 128
) -> Dict[str, np.ndarray]:
    """
    Tokenizes a list of cleaned text strings into input IDs and attention masks
    suitable for training or inference with a DistilBERT model.

    Args:
        texts (List[str]): List of clean input text documents.
        model_name (str): The pretrained model identifier on Hugging Face.
                          Defaults to 'distilbert-base-multilingual-cased' for multi-language support.
        max_length (int): The maximum token length for padding and truncation.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing keys:
                               - 'input_ids': NumPy array of shape (num_texts, max_length)
                               - 'attention_mask': NumPy array of shape (num_texts, max_length)
    """
    logger.info(f"Loading tokenizer '{model_name}'...")
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)

    logger.info(f"Tokenizing {len(texts)} document(s) with max_length={max_length}...")
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="np"  # Returns NumPy arrays immediately, perfect for Keras
    )

    # Convert BatchEncoding to a standard dict of NumPy arrays
    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"]
    }


class LabelEncoder:
    """
    A simple, modular utility class for converting textual categories (e.g. 'Finance', 'Sports')
    into numerical label indices, and mapping model prediction outputs back to the label strings.
    """

    def __init__(self):
        """Initializes empty maps for labels."""
        self.classes_: List[str] = []
        self.class_to_idx: Dict[str, int] = {}
        self.idx_to_class: Dict[int, str] = {}

    def fit(self, labels: List[Union[str, int]]) -> "LabelEncoder":
        """
        Learns the unique set of categories in the provided labels.

        Args:
            labels (List[Union[str, int]]): List of categorical or numeric target labels.

        Returns:
            LabelEncoder: The fitted encoder instance.
        """
        # Convert all labels to string to ensure consistent sorting and mapping
        string_labels = [str(lbl) for lbl in labels]
        self.classes_ = sorted(list(set(string_labels)))
        self.class_to_idx = {name: idx for idx, name in enumerate(self.classes_)}
        self.idx_to_class = {idx: name for idx, name in enumerate(self.classes_)}
        logger.info(f"Fitted LabelEncoder with classes: {self.class_to_idx}")
        return self

    def transform(self, labels: List[Union[str, int]]) -> np.ndarray:
        """
        Converts category labels (strings or ints) to integer label IDs.

        Args:
            labels (List[Union[str, int]]): Target label values.

        Returns:
            np.ndarray: Integer array of encoded label values.
        """
        if not self.class_to_idx:
            raise ValueError("LabelEncoder must be fitted before calling transform.")

        encoded = []
        for label in labels:
            label_str = str(label)
            if label_str not in self.class_to_idx:
                # Handle unseen labels by mapping them to an out-of-bounds warning
                logger.warning(f"Unseen label '{label_str}' encountered. Defaulting to 0.")
                encoded.append(0)
            else:
                encoded.append(self.class_to_idx[label_str])

        return np.array(encoded, dtype=np.int32)

    def fit_transform(self, labels: List[str]) -> np.ndarray:
        """
        Fits on the data and transforms it in a single step.

        Args:
            labels (List[str]): List of categorical target labels.

        Returns:
            np.ndarray: Integer array of encoded labels.
        """
        return self.fit(labels).transform(labels)

    def inverse_transform(self, indices: Union[List[int], np.ndarray]) -> List[str]:
        """
        Converts numerical label indices back to their original textual categories.

        Args:
            indices (Union[List[int], np.ndarray]): Encoded label indices.

        Returns:
            List[str]: Original class names.
        """
        if not self.idx_to_class:
            raise ValueError("LabelEncoder must be fitted before calling inverse_transform.")

        decoded = []
        for idx in indices:
            idx_int = int(idx)
            if idx_int not in self.idx_to_class:
                logger.warning(f"Index {idx_int} not found in classes mapping. Defaulting to 'Unknown'.")
                decoded.append("Unknown")
            else:
                decoded.append(self.idx_to_class[idx_int])

        return decoded


# Self-test block to verify functionality when run directly
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # 1. Test tokenization
    sample_texts = [
        "Welcome to the natural language processing tutorial.",
        "El aprendizaje de transferencia es muy útil."
    ]
    tokens = tokenize_texts(sample_texts, max_length=16)
    print("Tokenized input_ids shape:", tokens["input_ids"].shape)
    print("Tokenized attention_mask shape:", tokens["attention_mask"].shape)

    # 2. Test label encoding
    sample_labels = ["Tech", "Finance", "Sports", "Tech", "Finance"]
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(sample_labels)
    print("Encoded Labels:", encoded)
    decoded = encoder.inverse_transform(encoded)
    print("Decoded Labels:", decoded)

import torch
import spacy
import pickle
import os
import numpy as np
from torch.amp import autocast
import logging  # Import logging

from src.train.ner_train import config
from src.train.ner_train import utils
from src.train.ner_train.model import NERModel
from src.train.ner_train.data_utils import align_tokens_and_embeddings_modified  # Reutilizamos esta función

# --- Import Ragged and cupy (needed for _extract_spacy_embeddings) ---
try:
    from thinc.types import Ragged
except ImportError:
    Ragged = None
try:
    import cupy
except ImportError:
    cupy = None

logger = logging.getLogger(__name__)  # Get logger instance


class PipelinePredictor:
    """
    Class to perform NER predictions using the trained model and format
    the output for a pipeline including SA and Alert Generation placeholders.
    """

    def __init__(self, model_path=config.MODEL_SAVE_PATH, vocab_cache_dir=".cache"):
        """
        Initializes the predictor by loading the spaCy model, vocabularies,
        and the trained NER model. Also includes placeholders for SA/Alert models.
        """
        logger.info("Initializing PipelinePredictor...")
        self.device = config.DEVICE
        self.max_seq_len = config.MAX_SEQ_LEN
        self.max_word_len = config.MAX_WORD_LEN

        # --- Load spaCy Model ---
        logger.info(f"Loading spaCy model: {config.SPACY_MODEL}")
        if self.device.type == "cuda":
            try:
                spacy.require_gpu()
            except Exception as e:
                logger.warning(f"Failed to require spaCy GPU: {e}")
        try:
            self.nlp = spacy.load(config.SPACY_MODEL, exclude=["lemmatizer", "attribute_ruler"])
            logger.info(f"SpaCy model loaded. Using GPU: {spacy.prefer_gpu()}")
        except Exception as e:
            logger.error(f"Failed to load spaCy model '{config.SPACY_MODEL}': {e}", exc_info=True)
            raise

        # --- Load Vocabularies ---
        vocab_file = os.path.join(vocab_cache_dir, f"{config.DATASET_NAME}_vocabs.pkl")
        logger.info(f"Loading vocabs from: {vocab_file}")
        if not os.path.exists(vocab_file):
            logger.error(f"Vocab file not found at {vocab_file}. Please run train.py first.")
            raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
        with open(vocab_file, 'rb') as f:
            vocabs = pickle.load(f)
            self.pos_vocab = vocabs['pos']
            self.dep_vocab = vocabs['dep']
            self.ner_vocab = vocabs['ner']
            self.char_vocab = vocabs['char']
        logger.info("Vocabs loaded successfully.")
        self.id_to_ner = {v: k for k, v in self.ner_vocab.items()}

        # --- Load Trained NER Model ---
        logger.info(f"Loading trained NER model from: {model_path}")
        self.ner_model = NERModel(
            num_ner_tags=len(self.ner_vocab),
            pos_vocab_size=len(self.pos_vocab),
            dep_vocab_size=len(self.dep_vocab),
            char_vocab_size=len(self.char_vocab)
        )
        if not os.path.exists(model_path):
            logger.error(f"Trained model file not found at {model_path}.")
            raise FileNotFoundError(f"Trained model not found: {model_path}")
        self.ner_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.ner_model.to(self.device)
        self.ner_model.eval()  # Set to evaluation mode
        logger.info("NER model loaded and set to evaluation mode.")

        # --- Placeholder for SA and Alert Generation Models ---
        # Load your actual SA and Alert Generation models here
        # self.sa_model = load_sa_model()
        # self.alert_model = load_alert_model()
        logger.info("Placeholder: SA and Alert Generation models would be loaded here.")

        logger.info("PipelinePredictor initialized successfully.")

    def _extract_spacy_embeddings(self, doc):
        """Extract embeddings from spaCy doc, handling Ragged/cupy."""
        embeddings_batch = None
        possible_attr_names = ['last_hidden_layer_state', 'last_hidden_state', 'outputs', 'tensors']

        if hasattr(doc, '_') and hasattr(doc._, 'trf_data') and doc._.trf_data:
            for attr_name in possible_attr_names:
                if hasattr(doc._.trf_data, attr_name):
                    attr_value = getattr(doc._.trf_data, attr_name)
                    tensor_candidate = None

                    if attr_value is not None:
                        if Ragged is not None and isinstance(attr_value, Ragged):
                            if hasattr(attr_value, 'data'):
                                ragged_data = attr_value.data
                                is_cupy_array = cupy is not None and isinstance(ragged_data, cupy.ndarray)
                                is_numpy_array = isinstance(ragged_data, np.ndarray)
                                if is_cupy_array or is_numpy_array:
                                    try:
                                        tensor_candidate = torch.as_tensor(ragged_data, device=self.device if is_cupy_array else None)
                                    except Exception as e:
                                        logger.warning(f"Failed to convert Ragged.data to tensor: {e}")
                        elif torch.is_tensor(attr_value):
                            tensor_candidate = attr_value
                        elif isinstance(attr_value, list) and len(attr_value) > 0 and torch.is_tensor(attr_value[-1]):
                            tensor_candidate = attr_value[-1]

                        if tensor_candidate is not None and tensor_candidate.numel() > 0:
                            embeddings_batch = tensor_candidate.to(self.device)
                            break

        if embeddings_batch is None:
            raise ValueError("Could not extract valid transformer embeddings from spaCy doc.")

        if embeddings_batch.dim() == 3 and embeddings_batch.shape[0] == 1:
            embeddings_batch = embeddings_batch.squeeze(0)
        elif embeddings_batch.dim() != 2:
            raise ValueError(f"Unexpected spaCy embedding shape: {embeddings_batch.shape}")

        return embeddings_batch

    def _predict_ner(self, text: str) -> dict[str, str]:
        """
        Performs NER prediction on the input text.

        Args:
            text: The input text string.

        Returns:
            A dictionary mapping tokens to their predicted NER tags.
        """
        if not text.strip():
            return {}

        # 1. Process with spaCy
        doc = self.nlp(text)
        spacy_tokens = [token for token in doc]
        original_tokens_text = [token.text for token in spacy_tokens]

        # 2. Extract spaCy features (Embeddings, POS, DEP)
        spacy_embeddings_tensor = self._extract_spacy_embeddings(doc)
        pos_tags = [token.tag_ for token in spacy_tokens]
        dep_tags = [token.dep_ for token in spacy_tokens]

        # 3. Align Embeddings
        aligned_embeddings, attention_mask_bool = align_tokens_and_embeddings_modified(
            original_tokens_text, doc, spacy_embeddings_tensor
        )
        actual_len = int(attention_mask_bool.sum().item())

        # 4. Convert features to IDs and apply Padding/Truncation
        pos_ids = utils.tags_to_ids(pos_tags[:actual_len], self.pos_vocab)
        dep_ids = utils.tags_to_ids(dep_tags[:actual_len], self.dep_vocab)
        char_ids_list = [utils.chars_to_ids(token, self.char_vocab, self.max_word_len) for token in original_tokens_text[:actual_len]]

        if actual_len < self.max_seq_len:
            padding_len = self.max_seq_len - actual_len
            pos_ids.extend([self.pos_vocab['<PAD>']] * padding_len)
            dep_ids.extend([self.dep_vocab['<PAD>']] * padding_len)
            pad_char_ids = [self.char_vocab['<PAD>']] * self.max_word_len
            char_ids_list.extend([pad_char_ids] * padding_len)

        # 5. Convert to Tensors and add Batch dimension
        embeddings_batch = aligned_embeddings.unsqueeze(0).to(self.device)
        pos_ids_batch = torch.tensor(pos_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        dep_ids_batch = torch.tensor(dep_ids, dtype=torch.long).unsqueeze(0).to(self.device)
        char_ids_batch = torch.tensor(char_ids_list, dtype=torch.long).unsqueeze(0).to(self.device)
        attention_mask_batch = attention_mask_bool.unsqueeze(0).to(self.device)

        # 6. NER Inference
        with torch.no_grad():
            # Usar autocast con el parámetro 'cuda' en lugar de enabled=True
            if self.device.type == "cuda" and config.USE_AMP:
                with autocast('cuda'):
                    predictions = self.ner_model(
                        embeddings=embeddings_batch,
                        pos_ids=pos_ids_batch,
                        dep_ids=dep_ids_batch,
                        char_ids=char_ids_batch,
                        attention_mask=attention_mask_batch
                    )
            else:
                predictions = self.ner_model(
                    embeddings=embeddings_batch,
                    pos_ids=pos_ids_batch,
                    dep_ids=dep_ids_batch,
                    char_ids=char_ids_batch,
                    attention_mask=attention_mask_batch
                )

        # 7. Post-process NER results
        predicted_ids = predictions[0]
        predicted_tags = [self.id_to_ner.get(pid, "<UNK>") for pid in predicted_ids]

        ner_result_dict = {token: tag for token, tag in zip(original_tokens_text[:actual_len], predicted_tags[:actual_len])}

        return ner_result_dict

    def _predict_sentiment(self, text: str) -> str:
        """
        Placeholder for sentiment analysis prediction.
        Replace this with your actual SA model inference call.
        """
        if "crisis" in text.lower() or "rejects" in text.lower():
            return "negative"
        elif "approves" in text.lower() or "landmark" in text.lower():
            return "positive"
        else:
            return "neutral"

    def _generate_alert(self, text: str, ner_result: dict, sentiment: str) -> str:
        """
        Placeholder for alert generation.
        Replace this with your actual Alert Generation model inference call.
        """
        if sentiment == "negative" and any(tag.endswith("-LOC") or tag.endswith("-ORG") for tag in ner_result.values()):
            return f"Potential issue involving entities in: {text[:50]}..."
        elif sentiment == "positive" and "AI" in ner_result:
            return f"Positive development regarding AI: {text[:50]}..."
        elif "B-MISC" in ner_result.values():
            return f"Miscellaneous event detected: {text[:50]}..."
        else:
            return "No specific alert generated."

    def predict(self, text: str) -> dict:
        """
        Runs the full  pipeline (NER, SA placeholder, Alert placeholder)
        for the input text and returns the combined results.

        Args:
            text: The input text string.

        Returns:
            A dictionary containing the original text, NER results,
            sentiment (placeholder), and alert (placeholder).
        """
        logger.info(f"Processing text: '{text[:70]}...'")

        # 1. Predict NER
        ner_result = self._predict_ner(text)

        # 2. Predict Sentiment (using placeholder)
        sentiment = self._predict_sentiment(text)

        # 3. Generate Alert (using placeholder)
        alert = self._generate_alert(text, ner_result, sentiment)

        # 4. Format output dictionary
        output = {
            "text": text,
            "ner_result": ner_result,
            "sentiment": sentiment,
            "alert": alert
        }
        logger.info(f"Finished processing text. Alert: '{alert}'")
        return output


# --- Example Usage ---
if __name__ == "__main__":
    model_file = config.MODEL_SAVE_PATH
    vocab_file_path = os.path.join(".cache", f"{config.DATASET_NAME}_vocabs.pkl")

    if not os.path.exists(model_file) or not os.path.exists(vocab_file_path):
        logger.error(f"Error: Model ({model_file}) or vocab file ({vocab_file_path}) not found.")
        logger.error("Please run train.py first.")
    else:
        predictor = PipelinePredictor(model_path=model_file)

        text1 = "EU rejects German call to boycott British lamb."
        text2 = "Peter Blackburn reports on the latest developments in the EU."
        text3 = "The crisis in Venezuela continues to worsen according to the UN."
        text4 = "European Parliament approves landmark AI regulation framework."

        result1 = predictor.predict(text1)
        logger.info(f"Final Output 1:\n{result1}")

        result2 = predictor.predict(text2)
        logger.info(f"Final Output 2:\n{result2}")

        result3 = predictor.predict(text3)
        logger.info(f"Final Output 3:\n{result3}")

        result4 = predictor.predict(text4)
        logger.info(f"Final Output 4:\n{result4}")


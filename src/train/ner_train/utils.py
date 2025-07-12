import torch
from seqeval.metrics import classification_report, f1_score
import numpy as np
import random
from src.train.ner_train import config

# Use the PrintLogger from config instead of logging
logger = config.get_logger(__name__)

def set_seed(seed):
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Settings for ensuring determinism (may affect performance)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    print(f"[INFO] - {__name__} - Random seed set to {seed} for reproducibility")

def build_vocab(tags_list):
    """Builds a vocabulary from a list of tag sequences."""
    vocab = {"<PAD>": 0} # Padding token is always 0
    idx = 1
    for tags in tags_list:
        for tag in tags:
            if tag not in vocab:
                vocab[tag] = idx
                idx += 1
    print(f"[INFO] - {__name__} - Built vocabulary with {len(vocab)} tags")
    return vocab

def build_char_vocab(tokens_list):
    """Builds a character vocabulary from a list of token sequences."""
    vocab = {"<PAD>": 0, "<UNK>": 1} # Padding and Unknown characters
    idx = 2
    for tokens in tokens_list:
        for token in tokens:
            for char in token:
                if char not in vocab:
                    vocab[char] = idx
                    idx += 1
    print(f"[INFO] - {__name__} - Built character vocabulary with {len(vocab)} characters")
    return vocab

def tags_to_ids(tags, vocab):
    """Converts a sequence of tags to their corresponding IDs."""
    # Use 0 (<PAD>) if tag is not found (should not happen with CoNLL if vocab is built correctly)
    return [vocab.get(tag, 0) for tag in tags]

def chars_to_ids(token, char_vocab, max_word_len):
    """Converts characters in a token to IDs, with padding/truncation."""
    # Map chars to IDs, using <UNK> for unknown chars
    char_ids = [char_vocab.get(c, char_vocab["<UNK>"]) for c in token]
    # Truncate if longer than max_word_len
    if len(char_ids) > max_word_len:
        char_ids = char_ids[:max_word_len]
    # Pad if shorter than max_word_len
    else:
        char_ids.extend([char_vocab["<PAD>"]] * (max_word_len - len(char_ids)))
    return char_ids

def calculate_metrics(y_true, y_pred, id_to_ner):
    """
    Calculates NER metrics using seqeval, ignoring padding.

    Args:
        y_true (list[list[int]]): List of true tag ID sequences.
        y_pred (list[list[int]]): List of predicted tag ID sequences.
        id_to_ner (dict): Mapping from tag IDs to tag names.

    Returns:
        dict: Dictionary containing precision, recall, f1, and classification report.
    """
    # Convert IDs to NER tag strings
    true_sequences = []
    pred_sequences = []
    pad_id = config.ner_vocab.get("<PAD>", -1) # Get PAD ID safely

    for true_ids, pred_ids in zip(y_true, y_pred):
        # Convert only non-PAD tokens using true_ids as reference length
        true_seq = [id_to_ner[tid] for tid in true_ids if tid != pad_id]
        # Ensure pred_seq aligns with true_seq length and ignores padding
        pred_seq = [id_to_ner.get(pid, "<UNK>") for pid, tid in zip(pred_ids, true_ids) if tid != pad_id]

        # Ensure sequences have the same length (important for seqeval)
        min_len = min(len(true_seq), len(pred_seq))
        true_sequences.append(true_seq[:min_len])
        pred_sequences.append(pred_seq[:min_len])

    # Filter out any potentially empty sequences after processing
    valid_indices = [i for i, seq in enumerate(true_sequences) if len(seq) > 0]
    true_sequences = [true_sequences[i] for i in valid_indices]
    pred_sequences = [pred_sequences[i] for i in valid_indices]

    if not true_sequences or not pred_sequences:
         print(f"[WARNING] - {__name__} - No valid sequences found for metric calculation.")
         return {"precision": 0, "recall": 0, "f1": 0, "report": "No valid sequences."}

    # Calculate metrics using seqeval
    try:
        report_dict = classification_report(true_sequences, pred_sequences, output_dict=True, zero_division=0)
        report_str = classification_report(true_sequences, pred_sequences, zero_division=0)
        # Use macro average F1 score
        f1 = f1_score(true_sequences, pred_sequences, average="macro", zero_division=0)
        
        print(f"[INFO] - {__name__} - Calculated metrics - Precision: {report_dict['macro avg']['precision']:.4f}, Recall: {report_dict['macro avg']['recall']:.4f}, F1: {f1:.4f}")
        
        return {
            "precision": report_dict["macro avg"]["precision"],
            "recall": report_dict["macro avg"]["recall"],
            "f1": f1,
            "report": report_str
        }
    except Exception as e:
        print(f"[ERROR] - {__name__} - Error calculating metrics with seqeval: {e}")
        print(f"[ERROR] - {__name__} - True sequences sample: {true_sequences[:2]}")
        print(f"[ERROR] - {__name__} - Pred sequences sample: {pred_sequences[:2]}")
        return {"precision": 0, "recall": 0, "f1": 0, "report": f"Error: {e}"}


def pad_sequence(sequences, batch_first=True, padding_value=0):
    """
    Manual padding function (similar to torch.nn.utils.rnn.pad_sequence).
    Assumes sequences is a list of Tensors.
    """
    max_len = max(len(s) for s in sequences)
    trailing_dims = sequences[0].size()[1:] if sequences[0].dim() > 1 else ()
    out_dims = (len(sequences), max_len) + trailing_dims if batch_first else (max_len, len(sequences)) + trailing_dims

    out_tensor = sequences[0].new_full(out_dims, padding_value)
    for i, tensor in enumerate(sequences):
        length = tensor.size(0)
        # use index notation to prevent duplicate references to the tensor
        if batch_first:
            out_tensor[i, :length, ...] = tensor
        else:
            out_tensor[:length, i, ...] = tensor
    return out_tensor

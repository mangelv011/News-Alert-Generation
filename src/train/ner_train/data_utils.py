import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import spacy
import numpy as np
from tqdm.auto import tqdm
import pickle
import os
import logging  # Import logging

# --- Import Ragged and cupy ---
try:
    from thinc.types import Ragged
except ImportError:
    Ragged = None  # Flag that Ragged is not available
try:
    import cupy
except ImportError:
    cupy = None  # Flag that cupy is not available

from src.train.ner_train import config
from src.train.ner_train import utils

logger = config.get_logger(__name__)  # Get logger instance

# --- spaCy GPU Configuration ---
if config.DEVICE.type == "cuda":
    if cupy:
        try:
            spacy.require_gpu()
            logger.info("spaCy GPU requirement set.")
        except Exception as e:
            logger.warning(f"Failed to require spaCy GPU. SpaCy will run on CPU. Error: {e}")
            logger.warning("Ensure 'cupy' compatible with your CUDA version is installed (e.g., pip install cupy-cudaXXX).")
    else:
        logger.warning("cupy not found. SpaCy cannot use GPU and will run on CPU.")
else:
    logger.info("Running spaCy on CPU.")

# Load spaCy model once (after potential require_gpu call)
logger.info(f"Loading spaCy model: {config.SPACY_MODEL}")
try:
    # Load the model after attempting GPU activation
    # Do not exclude 'parser' as it's needed for dependency tags
    nlp = spacy.load(config.SPACY_MODEL, exclude=["lemmatizer", "attribute_ruler"])
except OSError:
    logger.warning(f"SpaCy model '{config.SPACY_MODEL}' not found. Downloading...")
    spacy.cli.download(config.SPACY_MODEL.replace("-", "_"))  # e.g., spacy download en_core_web_trf
    nlp = spacy.load(config.SPACY_MODEL, exclude=["lemmatizer", "attribute_ruler"])
logger.info(f"SpaCy model loaded. Using GPU: {spacy.prefer_gpu()}")  # Verify if spaCy is actually using GPU

# --- Preprocessing Functions ---


def align_tokens_and_embeddings_modified(conll_tokens, spacy_doc, spacy_embeddings_tensor):
    """
    Aligns CoNLL tokens with pre-extracted spaCy (transformer) embeddings.
    Uses the first spaCy subtoken corresponding to a CoNLL token.
    Returns aligned embeddings and an attention mask (True for real tokens, False for padding).

    Args:
        conll_tokens (list): List of original CoNLL tokens (strings).
        spacy_doc (spacy.tokens.Doc): SpaCy processed document.
        spacy_embeddings_tensor (torch.Tensor): Embeddings tensor (num_spacy_tokens, embedding_dim).
    """
    aligned_embeddings = []
    token_indices = []  # Stores the index of the spaCy token corresponding to each CoNLL token

    spacy_token_idx = 0
    for conll_token in conll_tokens:
        current_spacy_text = ""
        start_idx = spacy_token_idx
        # Advance through spaCy tokens until the text matches (or exceeds) the CoNLL token
        while spacy_token_idx < len(spacy_doc) and len(current_spacy_text) < len(conll_token):
            if spacy_token_idx >= len(spacy_doc):
                break
            current_spacy_text += spacy_doc[spacy_token_idx].text_with_ws
            spacy_token_idx += 1

        if not current_spacy_text.startswith(conll_token):
            if start_idx < len(spacy_doc):
                token_indices.append(start_idx)
                spacy_token_idx = start_idx + 1
            else:
                token_indices.append(-1)
        else:
            token_indices.append(start_idx)

    for idx in token_indices:
        if idx != -1 and idx < spacy_embeddings_tensor.shape[0]:
            aligned_embeddings.append(spacy_embeddings_tensor[idx])
        else:
            aligned_embeddings.append(torch.zeros(config.SPACY_EMBEDDING_DIM, device=spacy_embeddings_tensor.device))

    if not aligned_embeddings:
        return torch.empty(0, config.SPACY_EMBEDDING_DIM), torch.empty(0, dtype=torch.bool)

    aligned_embeddings_tensor = torch.stack(aligned_embeddings)

    seq_len = aligned_embeddings_tensor.size(0)
    attention_mask = torch.ones(seq_len, dtype=torch.bool, device=aligned_embeddings_tensor.device)

    if seq_len > config.MAX_SEQ_LEN:
        aligned_embeddings_tensor = aligned_embeddings_tensor[:config.MAX_SEQ_LEN]
        attention_mask = attention_mask[:config.MAX_SEQ_LEN]
    elif seq_len < config.MAX_SEQ_LEN:
        padding_len = config.MAX_SEQ_LEN - seq_len
        padding_emb = torch.zeros(padding_len, config.SPACY_EMBEDDING_DIM, device=aligned_embeddings_tensor.device)
        padding_mask = torch.zeros(padding_len, dtype=torch.bool, device=attention_mask.device)
        aligned_embeddings_tensor = torch.cat([aligned_embeddings_tensor, padding_emb], dim=0)
        attention_mask = torch.cat([attention_mask, padding_mask], dim=0)

    return aligned_embeddings_tensor, attention_mask


def _extract_spacy_embeddings(doc, device):
    """Internal helper to extract embeddings from spaCy doc, handling Ragged/cupy."""
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
                                    tensor_candidate = torch.as_tensor(ragged_data, device=device if is_cupy_array else None)
                                except Exception as e:
                                    logger.warning(f"Failed to convert Ragged.data ({type(ragged_data)}) to tensor: {e}", exc_info=True)
                                    tensor_candidate = None

                    elif torch.is_tensor(attr_value):
                        tensor_candidate = attr_value

                    elif isinstance(attr_value, list) and len(attr_value) > 0 and torch.is_tensor(attr_value[-1]):
                        tensor_candidate = attr_value[-1]

                    if tensor_candidate is not None and tensor_candidate.numel() > 0:
                        embeddings_batch = tensor_candidate.to(device)
                        break

    if embeddings_batch is None:
        raise ValueError("Could not extract valid transformer embeddings from spaCy doc.")

    if embeddings_batch.dim() == 3 and embeddings_batch.shape[0] == 1:
        embeddings_batch = embeddings_batch.squeeze(0)
    elif embeddings_batch.dim() != 2:
        raise ValueError(f"Unexpected spaCy embedding shape after extraction: {embeddings_batch.shape}")

    return embeddings_batch


def preprocess_data(split='train', cache_dir=config.CACHE_DIR):
    """Loads, preprocesses, and caches the dataset."""
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{config.DATASET_NAME}_{split}_processed.pkl")
    vocab_file = os.path.join(cache_dir, f"{config.DATASET_NAME}_vocabs.pkl")
    
    # Asegurarse de que el directorio de modelos exista
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)

    if os.path.exists(vocab_file):
        try:
            with open(vocab_file, 'rb') as f:
                vocabs = pickle.load(f)
                config.pos_vocab = vocabs['pos']
                config.dep_vocab = vocabs['dep']
                config.ner_vocab = vocabs['ner']
                config.char_vocab = vocabs['char']
                logger.info("Loaded vocabs from cache.")
        except Exception as e:
            logger.error(f"Error loading vocab file {vocab_file}: {e}. Please delete it and re-run.", exc_info=True)
            raise
    elif split != 'train':
        raise FileNotFoundError(f"Vocab file {vocab_file} not found. Please process the 'train' split first.")

    if os.path.exists(cache_file):
        logger.info(f"Loading cached processed data for split '{split}'...")
        try:
            with open(cache_file, 'rb') as f:
                processed_data = pickle.load(f)
            if not processed_data and split == 'train':
                logger.warning("Loaded empty processed data from train cache. Reprocessing...")
                os.remove(cache_file)
                if os.path.exists(vocab_file):
                    os.remove(vocab_file)
                    config.pos_vocab, config.dep_vocab, config.ner_vocab, config.char_vocab = {}, {}, {}, {}
                processed_data = None
            elif not processed_data:
                logger.warning("Loaded empty processed data from cache.")
                return processed_data
            else:
                logger.info("Cached data loaded.")
                return processed_data
        except (pickle.UnpicklingError, EOFError, Exception) as e:
            logger.warning(f"Error loading cache file {cache_file}: {e}. Reprocessing...", exc_info=True)
            os.remove(cache_file)
            if split == 'train' and os.path.exists(vocab_file):
                os.remove(vocab_file)
                config.pos_vocab, config.dep_vocab, config.ner_vocab, config.char_vocab = {}, {}, {}, {}
            processed_data = None

    logger.info(f"Processing data for split '{split}'...")
    # Añadiendo trust_remote_code=True para permitir la ejecución de código personalizado
    dataset = load_dataset(config.DATASET_NAME, split=split, trust_remote_code=True)

    processed_examples = []
    all_pos_tags_for_vocab = []
    all_dep_tags_for_vocab = []
    all_tokens_for_char_vocab = []
    skipped_count = 0

    if split == 'train' and not config.ner_vocab:
        try:
            ner_feature = dataset.features['ner_tags'].feature
            config.ner_vocab = {tag: i for i, tag in enumerate(ner_feature.names)}
            if '<PAD>' not in config.ner_vocab:
                config.ner_vocab['<PAD>'] = len(config.ner_vocab)
            logger.info("Built NER vocab from dataset features.")
        except Exception as e:
            logger.error(f"Failed to build NER vocab from dataset features: {e}", exc_info=True)
            raise

    for i, example in enumerate(tqdm(dataset, desc=f"Processing {split}")):
        tokens = example['tokens']
        ner_tags_original_ids = example['ner_tags']

        try:
            ner_tags_str = [dataset.features['ner_tags'].feature.int2str(tag_id) for tag_id in ner_tags_original_ids]
        except Exception as e:
            logger.warning(f"Skipping example {i} due to error converting NER tags: {e}")
            skipped_count += 1
            continue

        text = " ".join(tokens)
        if not text.strip():
            skipped_count += 1
            continue
        try:
            doc = nlp(text)
        except Exception as e:
            logger.warning(f"Skipping example {i} due to spaCy processing error: {e}")
            skipped_count += 1
            continue

        try:
            spacy_embeddings_tensor = _extract_spacy_embeddings(doc, config.DEVICE)
        except ValueError as e:
            logger.warning(f"Skipping example {i}: {e}")
            skipped_count += 1
            continue
        except Exception as e:
            logger.warning(f"Skipping example {i} due to unexpected error extracting embeddings: {e}", exc_info=True)
            skipped_count += 1
            continue

        pos_tags = []
        dep_tags = []
        spacy_idx = 0
        for token_text in tokens:
            found = False
            temp_text = ""
            start_idx = spacy_idx
            while spacy_idx < len(doc):
                temp_text += doc[spacy_idx].text_with_ws
                spacy_idx += 1
                if temp_text.strip() == token_text:
                    pos_tags.append(doc[spacy_idx-1].tag_)
                    dep_tags.append(doc[spacy_idx-1].dep_)
                    found = True
                    break
                elif len(temp_text.strip()) > len(token_text):
                    if spacy_idx > start_idx + 1:
                        pos_tags.append(doc[spacy_idx-2].tag_)
                        dep_tags.append(doc[spacy_idx-2].dep_)
                        spacy_idx -= 1
                        found = True
                        break
                    else:
                        pos_tags.append(doc[start_idx].tag_)
                        dep_tags.append(doc[start_idx].dep_)
                        found = True
                        break

            if not found:
                if start_idx < len(doc):
                    pos_tags.append(doc[start_idx].tag_)
                    dep_tags.append(doc[start_idx].dep_)
                    spacy_idx = start_idx + 1
                else:
                    pos_tags.append("<UNK>")
                    dep_tags.append("<UNK>")

        min_len = min(len(tokens), len(pos_tags), len(dep_tags), len(ner_tags_str))
        if min_len == 0:
            skipped_count += 1
            continue
        tokens = tokens[:min_len]
        pos_tags = pos_tags[:min_len]
        dep_tags = dep_tags[:min_len]
        ner_tags_str = ner_tags_str[:min_len]

        try:
            aligned_embeddings, attention_mask = align_tokens_and_embeddings_modified(
                tokens, doc, spacy_embeddings_tensor
            )
        except Exception as e:
            logger.warning(f"Skipping example {i} due to embedding alignment error: {e}", exc_info=True)
            skipped_count += 1
            continue

        if aligned_embeddings.nelement() == 0 or attention_mask.nelement() == 0:
            logger.warning(f"Skipping example {i} due to empty aligned embeddings or mask.")
            skipped_count += 1
            continue

        current_seq_len = int(attention_mask.sum().item())
        if current_seq_len == 0:
            logger.warning(f"Skipping example {i} due to zero sequence length after alignment.")
            skipped_count += 1
            continue

        tokens = tokens[:current_seq_len]
        pos_tags = pos_tags[:current_seq_len]
        dep_tags = dep_tags[:current_seq_len]
        ner_tags_str = ner_tags_str[:current_seq_len]

        if split == 'train':
            all_pos_tags_for_vocab.append(pos_tags)
            all_dep_tags_for_vocab.append(dep_tags)
            all_tokens_for_char_vocab.append(tokens)

        processed_examples.append({
            'tokens': tokens,
            'embeddings': aligned_embeddings,
            'pos_tags': pos_tags,
            'dep_tags': dep_tags,
            'ner_tags': ner_tags_str,
            'attention_mask': attention_mask
        })

    logger.info(f"Finished initial processing for {split}. Total examples: {len(dataset)}, Skipped: {skipped_count}, Processed: {len(processed_examples)}")
    if skipped_count == len(dataset) and len(dataset) > 0:
        logger.error("CRITICAL ERROR: All examples were skipped during preprocessing!")
        logger.error("Check spaCy/transformer installation, compatibility, and data integrity.")

    if split == 'train':
        logger.info("Building vocabs...")
        config.pos_vocab = utils.build_vocab(all_pos_tags_for_vocab)
        config.dep_vocab = utils.build_vocab(all_dep_tags_for_vocab)
        config.char_vocab = utils.build_char_vocab(all_tokens_for_char_vocab)
        if '<PAD>' not in config.ner_vocab:
            config.ner_vocab['<PAD>'] = len(config.ner_vocab)

        vocabs_to_save = {
            'pos': config.pos_vocab,
            'dep': config.dep_vocab,
            'ner': config.ner_vocab,
            'char': config.char_vocab
        }
        try:
            with open(vocab_file, 'wb') as f:
                pickle.dump(vocabs_to_save, f)
            logger.info(f"Vocabs saved to {vocab_file}")
            logger.info(f"POS Vocab Size (using .tag_): {len(config.pos_vocab)}")
            logger.info(f"DEP Vocab Size: {len(config.dep_vocab)}")
            logger.info(f"NER Vocab Size: {len(config.ner_vocab)}")
            logger.info(f"Char Vocab Size: {len(config.char_vocab)}")
        except Exception as e:
            logger.error(f"Error saving vocab file {vocab_file}: {e}", exc_info=True)
            raise

    final_processed_data = []
    id_to_ner = get_id_to_ner()
    if not id_to_ner:
        raise RuntimeError("Failed to get id_to_ner mapping. Vocabularies might be missing or failed to load/build.")
    ner_pad_id = config.ner_vocab['<PAD>']
    pos_pad_id = config.pos_vocab['<PAD>']
    dep_pad_id = config.dep_vocab['<PAD>']
    char_pad_id = config.char_vocab['<PAD>']

    for ex in tqdm(processed_examples, desc=f"Converting tags and chars to IDs ({split})"):
        pos_ids = utils.tags_to_ids(ex['pos_tags'], config.pos_vocab)
        dep_ids = utils.tags_to_ids(ex['dep_tags'], config.dep_vocab)
        ner_ids = utils.tags_to_ids(ex['ner_tags'], config.ner_vocab)
        char_ids_list = [utils.chars_to_ids(token, config.char_vocab, config.MAX_WORD_LEN) for token in ex['tokens']]

        seq_len = len(pos_ids)
        if seq_len < config.MAX_SEQ_LEN:
            padding_len = config.MAX_SEQ_LEN - seq_len
            pos_ids.extend([pos_pad_id] * padding_len)
            dep_ids.extend([dep_pad_id] * padding_len)
            ner_ids.extend([ner_pad_id] * padding_len)
            pad_char_ids_for_token = [char_pad_id] * config.MAX_WORD_LEN
            char_ids_list.extend([pad_char_ids_for_token] * padding_len)

        try:
            final_processed_data.append({
                'embeddings': ex['embeddings'].cpu(),
                'pos_ids': torch.tensor(pos_ids, dtype=torch.long).cpu(),
                'dep_ids': torch.tensor(dep_ids, dtype=torch.long).cpu(),
                'char_ids': torch.tensor(char_ids_list, dtype=torch.long).cpu(),
                'ner_ids': torch.tensor(ner_ids, dtype=torch.long).cpu(),
                'attention_mask': ex['attention_mask'].cpu(),
                'original_tokens': ex['tokens'],
                'original_ner_tags': ex['ner_tags']
            })
        except Exception as e:
            logger.error(f"Error converting example to tensor: {e}. Example keys: {ex.keys()}", exc_info=True)
            raise

    logger.info(f"Saving processed data to cache: {cache_file}")
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(final_processed_data, f)
        logger.info("Processed data saved.")
    except Exception as e:
        logger.error(f"Error saving processed data cache file {cache_file}: {e}", exc_info=True)
        raise

    return final_processed_data


# --- Dataset Class ---


class NERDataset(Dataset):
    """PyTorch Dataset wrapper for the processed NER data."""

    def __init__(self, data):
        """
        Args:
            data (list[dict]): List of processed data dictionaries.
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# --- Collate Function ---


def collate_fn(batch):
    """
    Custom collate function. Assumes data is already padded/truncated.
    Stacks the tensors from the batch items.
    """
    embeddings = torch.stack([item['embeddings'] for item in batch])
    pos_ids = torch.stack([item['pos_ids'] for item in batch])
    dep_ids = torch.stack([item['dep_ids'] for item in batch])
    char_ids = torch.stack([item['char_ids'] for item in batch])
    ner_ids = torch.stack([item['ner_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    original_tokens = [item['original_tokens'] for item in batch]
    original_ner_tags = [item['original_ner_tags'] for item in batch]

    return {
        'embeddings': embeddings,
        'pos_ids': pos_ids,
        'dep_ids': dep_ids,
        'char_ids': char_ids,
        'ner_ids': ner_ids,
        'attention_mask': attention_mask,
        'original_tokens': original_tokens,
        'original_ner_tags': original_ner_tags
    }


# --- DataLoaders ---


def get_dataloaders(batch_size):
    """Creates DataLoaders for train, validation, and test splits."""
    logger.info("Creating DataLoaders...")
    train_data = preprocess_data('train')
    val_data = preprocess_data('validation')
    test_data = preprocess_data('test')

    train_dataset = NERDataset(train_data)
    val_dataset = NERDataset(val_data)
    test_dataset = NERDataset(test_data)

    num_workers = 0
    logger.info(f"Setting num_workers={num_workers} for DataLoaders.")

    pin_memory = (config.DEVICE.type == 'cuda' and num_workers > 0)
    persistent_workers = num_workers > 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers
    )
    logger.info("DataLoaders created.")
    return train_loader, val_loader, test_loader


# --- Helper to get inverse mapping ---


def get_id_to_ner():
    """Gets the mapping from NER tag IDs back to tag names."""
    if not config.ner_vocab:
        logger.warning("NER vocab not loaded. Attempting to load/build via preprocess_data('train')...")
        try:
            preprocess_data('train')
        except Exception as e:
            logger.error(f"Error loading/building vocabs needed for id_to_ner mapping: {e}", exc_info=True)
            return {}
    if config.ner_vocab:
        return {v: k for k, v in config.ner_vocab.items()}
    else:
        logger.error("NER vocab is still empty after attempting to load/build.")
        return {}

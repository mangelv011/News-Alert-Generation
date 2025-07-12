import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from collections import Counter, defaultdict
from datasets import load_from_disk
from transformers import pipeline
import numpy as np

# --- Constants ---
TEXT_PAD_TOKEN = '<PAD>'
TEXT_UNK_TOKEN = '<UNK>'
NER_PAD_TOKEN = '<PAD>'
NER_UNK_TOKEN = 'O'
SENTIMENT_PAD_TOKEN = '<PAD>'
SENTIMENT_UNK_TOKEN = 'NEU'

# --- Vocab Building ---
def build_vocab_from_sequences(sequences, pad_token=TEXT_PAD_TOKEN, unk_token=TEXT_UNK_TOKEN, min_freq=1):
    """Builds vocabulary from token sequences with frequency filtering."""
    counts = Counter(token.lower() for seq in sequences for token in seq)
    vocab = defaultdict(lambda: vocab[unk_token])
    vocab[pad_token] = 0
    vocab[unk_token] = 1
    idx = 2
    for token, count in sorted(counts.items()):
        if count >= min_freq and token not in vocab:
            vocab[token] = idx
            idx += 1
    print(f"Built Text Vocab with {len(vocab)} items")
    return vocab

def build_tag_vocab(tag_sequences_or_list, pad_token, unk_token=None, is_target=False):
    """Builds vocabulary for tag sequences (NER) or list of tags (sentiment)."""
    # Count tags appropriately based on input type
    if is_target:  # Flat list of sentiment labels
        counts = Counter(tag_sequences_or_list)
        items = counts.keys()
    else:  # Sequences of NER tags
        counts = Counter(tag for seq in tag_sequences_or_list for tag in seq)
        items = counts.keys()

    # Initialize vocab with special tokens
    vocab = defaultdict(int)
    if unk_token and not is_target:
        vocab[pad_token] = 0
        vocab[unk_token] = 1
        idx = 2
    else:
        vocab[pad_token] = 0
        idx = 1

    # Add regular tags
    for tag in sorted(items):
        if tag != pad_token and tag != unk_token:
            vocab[tag] = idx
            idx += 1

    # Set default factory function
    if unk_token and unk_token in vocab:
        vocab.default_factory = lambda: vocab[unk_token]
    else:
        vocab.default_factory = lambda: vocab[pad_token]

    print(f"Built Tag Vocab with {len(vocab)} items: {list(vocab.keys())}")
    return vocab

# --- Load Sentiment Pipeline (evaluation mode only) ---
print("Loading sentiment analysis pipeline...")
# Initialize pipeline in evaluation mode (no training)
with torch.no_grad():
    sentiment_pipeline = pipeline(
        "sentiment-analysis", 
        model="cardiffnlp/twitter-roberta-base-sentiment-latest", 
        device=0 if torch.cuda.is_available() else -1
    )
    # Ensure model is in evaluation mode
    sentiment_pipeline.model.eval()
print("Sentiment pipeline loaded (eval mode)")

# Map model outputs to our sentiment classes
SENTIMENT_LABEL_MAP = {
    'LABEL_0': 'NEG', 'LABEL_1': 'NEU', 'LABEL_2': 'POS',
    'negative': 'NEG', 'neutral': 'NEU', 'positive': 'POS'
}

def encode_twitter_example(example, text_vocab, ner_vocab, sentiment_vocab, max_len):
    """Encodes examples from Twitter dataset."""
    # 1. Text Encoding - Safely
    tokens = example['tokens'][:max_len]
    text_encoded = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in text_vocab:
            text_encoded.append(text_vocab[token_lower])
        else:
            text_encoded.append(text_vocab[TEXT_UNK_TOKEN])
    example['encoded_text'] = text_encoded

    # 2. NER Encoding - Safely
    ner_tags = example['ner_tags'][:max_len]
    ner_encoded = []
    for tag in ner_tags:
        if tag in ner_vocab:
            ner_encoded.append(ner_vocab[tag])
        else:
            ner_encoded.append(ner_vocab[NER_UNK_TOKEN])
    example['encoded_ner'] = ner_encoded

    # 3. Sentiment Encoding
    sentiment_label = example['sentiment_label']
    if sentiment_label in sentiment_vocab:
        example['encoded_sentiment'] = sentiment_vocab[sentiment_label]
    else:
        example['encoded_sentiment'] = sentiment_vocab[SENTIMENT_UNK_TOKEN]
    
    return example

# --- Dataset Functions ---
def collate_fn(batch_list, text_pad_idx, ner_pad_idx):
    """Collate function for creating batches with padding."""
    texts = [item['encoded_text'].clone().detach() if torch.is_tensor(item['encoded_text']) 
             else torch.tensor(item['encoded_text'], dtype=torch.long) for item in batch_list]
             
    ners = [item['encoded_ner'].clone().detach() if torch.is_tensor(item['encoded_ner']) 
            else torch.tensor(item['encoded_ner'], dtype=torch.long) for item in batch_list]
            
    sentiments = torch.tensor([item['encoded_sentiment'] for item in batch_list], dtype=torch.long)

    texts_padded = pad_sequence(texts, batch_first=True, padding_value=text_pad_idx)
    ners_padded = pad_sequence(ners, batch_first=True, padding_value=ner_pad_idx)

    return texts_padded, ners_padded, sentiments

# --- Main Data Loading Function ---
def load_data(batch_size=32, max_len=128, text_vocab=None, ner_vocab=None, sentiment_vocab=None):
    """
    Loads sentiment analysis data, builds/uses vocabs, and creates dataloaders.
    
    Args:
        batch_size: Batch size for dataloaders
        max_len: Maximum sequence length
        text_vocab: Optional pre-built text vocabulary
        ner_vocab: Optional pre-built NER vocabulary
        sentiment_vocab: Optional pre-built sentiment vocabulary
    """
    # Define project structure paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    data_dir = os.path.join(project_root, "src", "data", "sentiment_analysis_dataset", "joint_ner_sentiment_dataset")
    
    if not os.path.exists(data_dir):
        print("Sentiment analysis dataset not found. Running preparation script...")
        # Import and run the preparation script
        from prepare_twitter_dataset import prepare_twitter_sentiment_dataset
        data_dir = prepare_twitter_sentiment_dataset()
    
    print(f"Loading sentiment analysis dataset from {data_dir}...")
    dataset = load_from_disk(data_dir)
    
    # Get available NER tags
    ner_tag_names = list(set([tag for ex in dataset['train']['ner_tags'] for tag in ex]))
    print(f"Dataset loaded with NER tags: {ner_tag_names}")
    
    # 1. Process provided vocabs or build new ones
    if all([text_vocab, ner_vocab, sentiment_vocab]):
        print("Using provided vocabularies")
        print(f"Vocabulary sizes: Text={len(text_vocab)}, NER={len(ner_vocab)}, Sentiment={len(sentiment_vocab)}")
    else:
        # Build vocabularies
        if text_vocab is None:
            print("Building text vocabulary...")
            all_train_tokens = [tokens[:max_len] for tokens in dataset['train']['tokens']]
            text_vocab = build_vocab_from_sequences(all_train_tokens, min_freq=1)
        
        if ner_vocab is None:
            print("Building NER vocabulary...")
            all_train_ner_tags = [tags[:max_len] for tags in dataset['train']['ner_tags']]
            ner_vocab = build_tag_vocab(all_train_ner_tags, pad_token=NER_PAD_TOKEN, unk_token=NER_UNK_TOKEN)
        
        if sentiment_vocab is None:
            print("Building sentiment vocabulary...")
            all_train_sentiment_labels = dataset['train']['sentiment_label']
            sentiment_vocab = build_tag_vocab(all_train_sentiment_labels, pad_token=SENTIMENT_PAD_TOKEN, 
                                            unk_token=SENTIMENT_UNK_TOKEN, is_target=True)

    # 2. Get padding indices
    text_pad_idx = text_vocab[TEXT_PAD_TOKEN]
    ner_pad_idx = ner_vocab[NER_PAD_TOKEN]

    # 3. Encode the dataset
    print("Encoding dataset examples...")
    try:
        # Columns to remove
        columns_to_remove = ['original_tweet', 'original_entity', 'tokens', 'ner_tags', 'sentiment_label']
        columns_to_remove = [col for col in columns_to_remove if col in dataset['train'].column_names]
        
        print(f"Columns to remove: {columns_to_remove}")
        print(f"Current columns: {dataset['train'].column_names}")
        
        # Encode the dataset
        encode_fn = lambda ex: encode_twitter_example(ex, text_vocab, ner_vocab, sentiment_vocab, max_len)
        dataset = dataset.map(encode_fn, remove_columns=columns_to_remove)
    except Exception as e:
        print(f"Error during encoding: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 4. Setup for PyTorch
    dataset.set_format(type='torch', columns=['encoded_text', 'encoded_ner', 'encoded_sentiment'])

    # 5. Create DataLoaders
    train_loader = DataLoader(
        dataset['train'], 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, text_pad_idx, ner_pad_idx)
    )
    val_loader = DataLoader(
        dataset['validation'], 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, text_pad_idx, ner_pad_idx)
    )
    test_loader = DataLoader(
        dataset['test'], 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, text_pad_idx, ner_pad_idx)
    )

    print(f"Data loading complete. Train={len(dataset['train'])}, Val={len(dataset['validation'])}, Test={len(dataset['test'])}")
    print(f"Final vocabulary sizes: Text={len(text_vocab)}, NER={len(ner_vocab)}, Sentiment={len(sentiment_vocab)}")

    return train_loader, val_loader, test_loader, text_vocab, ner_vocab, sentiment_vocab
"""
Utility functions for the alert generation model.
"""
import random
import torch
import numpy as np
from . import config

def set_seed(seed=config.RANDOM_SEED):
    """
    Set random seeds for reproducibility
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def format_input(example):
    """
    Format the input for the model
    
    Args:
        example: Dictionary containing NER results, sentiment, and text
        
    Returns:
        Formatted input string
    """
    ner_parts = [f"{entity}:{label}" for entity, label in example['ner_result'].items() if label and entity]
    ner_text = ", ".join(ner_parts)
    return f"NER: {ner_text} | Sentiment: {example['sentiment']} | Text: {example['text']}" if config.INCLUDE_TEXT else f"NER: {ner_text} | Sentiment: {example['sentiment']}"


def compute_metrics(eval_preds, tokenizer):
    """
    Compute ROUGE metrics for model evaluation
    
    Args:
        eval_preds: Model predictions and labels
        tokenizer: Tokenizer for decoding predictions
        
    Returns:
        Dictionary of ROUGE metrics
    """
    import evaluate
    import numpy as np
    
    rouge = evaluate.load("rouge")
    preds, labels = eval_preds
    preds = np.where(preds >= 0, preds, tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(predictions=preds, references=labels, use_stemmer=True)
    return {k: round(v * 100, 2) for k, v in result.items()}
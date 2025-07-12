"""
Data utilities for the alert generation model.
"""
import json
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer
from . import config
from .utils import format_input

def load_dataset_from_jsonl(file_path=config.DATA_PATH):
    """
    Load and parse dataset from a JSONL file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON examples with non-empty NER results
    """
    data = []
    skipped_count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            try:
                example = json.loads(line.strip())
                if 'ner_result' in example and example['ner_result']:
                    data.append(example)
                else:
                    skipped_count += 1
            except json.JSONDecodeError:
                print(f"Error parsing line {line_number}: {line}")
                continue
    
    print(f"Total examples loaded: {len(data)}")
    print(f"Examples skipped due to empty ner_result: {skipped_count}")
    
    return data


def prepare_datasets(data):
    """
    Create structured datasets and split them into train, validation, and test sets
    
    Args:
        data: List of parsed examples
        
    Returns:
        Dictionary containing train, validation, and test datasets
    """
    df = pd.DataFrame(data)
    dataset = Dataset.from_pandas(df)
    dataset = dataset.shuffle(seed=config.RANDOM_SEED)
    train_test = dataset.train_test_split(test_size=0.2, seed=config.RANDOM_SEED)
    test_valid = train_test['test'].train_test_split(test_size=0.5, seed=config.RANDOM_SEED)
    return {
        'train': train_test['train'],
        'validation': test_valid['train'],
        'test': test_valid['test']
    }


def preprocess_function(examples, tokenizer):
    """
    Preprocess examples for the model
    
    Args:
        examples: Batch of examples
        tokenizer: Tokenizer for the model
        
    Returns:
        Processed inputs with tokenized text and labels
    """
    inputs = [format_input({
        'ner_result': examples['ner_result'][i],
        'sentiment': examples['sentiment'][i],
        'text': examples['text'][i]
    }) for i in range(len(examples['ner_result']))]
    
    targets = examples['alert']
    model_inputs = tokenizer(inputs, max_length=config.MAX_INPUT_LENGTH, truncation=True, padding="max_length")
    labels = tokenizer(targets, max_length=config.MAX_TARGET_LENGTH, truncation=True, padding="max_length")
    labels["input_ids"] = [[(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_and_prepare_data():
    """
    Load, prepare, and preprocess the dataset
    
    Returns:
        Tuple of (tokenized_datasets, raw_datasets, tokenizer)
    """
    print(f"Loading data from {config.DATA_PATH}")
    raw_data = load_dataset_from_jsonl()
    raw_datasets = prepare_datasets(raw_data)
    
    print(f"Loading tokenizer {config.MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    print("Preprocessing datasets")
    tokenized_datasets = {
        split: dataset.map(
            lambda examples: preprocess_function(examples, tokenizer), 
            batched=True, 
            remove_columns=dataset.column_names
        )
        for split, dataset in raw_datasets.items()
    }
    
    return tokenized_datasets, raw_datasets, tokenizer
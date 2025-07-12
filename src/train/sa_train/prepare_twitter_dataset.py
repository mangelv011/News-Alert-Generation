import os
import torch
import pandas as pd
import kagglehub
from datasets import Dataset, DatasetDict
from transformers import pipeline
from collections import Counter
from tqdm import tqdm
import numpy as np
import glob

def download_twitter_dataset():
    """Downloads Twitter Sentiment Analysis dataset from Kaggle."""
    print("Downloading Twitter Entity Sentiment Analysis dataset from Kaggle...")
    path = kagglehub.dataset_download("jp797498e/twitter-entity-sentiment-analysis")
    print(f"Dataset downloaded to: {path}")
    
    # List all downloaded files for debugging
    print("\nFiles available in the downloaded directory:")
    for file_path in glob.glob(os.path.join(path, "*")):
        print(f"  - {os.path.basename(file_path)}")
    
    return path

def prepare_twitter_sentiment_dataset():
    """
    Prepares Twitter dataset for sentiment analysis with NER.
    This dataset already includes both sentiment and entities.
    """
    # Define project structure paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    output_dir = os.path.join(project_root, "src", "data", "sentiment_analysis_dataset")
    os.makedirs(output_dir, exist_ok=True)
    
    # Download or use existing dataset
    dataset_path = download_twitter_dataset()
    
    # Look for CSV files
    train_file = os.path.join(dataset_path, "twitter_training.csv")
    valid_file = os.path.join(dataset_path, "twitter_validation.csv")
    
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        # Look for CSV files if not found directly
        csv_files = sorted(glob.glob(os.path.join(dataset_path, "*.csv")))
        if len(csv_files) >= 2:
            train_file = csv_files[0]
            valid_file = csv_files[1]
            print(f"Using '{os.path.basename(train_file)}' for training")
            print(f"Using '{os.path.basename(valid_file)}' for validation/test")
        elif len(csv_files) == 1:
            train_file = valid_file = csv_files[0]
            print(f"Using a single file '{os.path.basename(train_file)}' split for train/val/test")
        else:
            raise FileNotFoundError(f"No CSV files found in {dataset_path}")
    
    # Load CSV files without headers and assign column names manually
    print("\nLoading CSV files...")
    try:
        # Load CSVs without header
        train_df = pd.read_csv(train_file, header=None)
        valid_df = pd.read_csv(valid_file, header=None)
        
        # Assign column names
        column_names = ['ID', 'Entity', 'Sentiment', 'Tweet']
        train_df.columns = column_names
        valid_df.columns = column_names
        
        print(f"Training dataset: {train_df.shape[0]} rows, columns: {train_df.columns.tolist()}")
        print(f"Validation dataset: {valid_df.shape[0]} rows, columns: {valid_df.columns.tolist()}")
        
        # Check first few rows
        print("\nFirst rows of training dataset:")
        print(train_df.head(3))
    except Exception as e:
        print(f"Error loading CSV files: {e}")
        # Try to visualize file content for debugging
        try:
            with open(train_file, 'r', encoding='utf-8') as f:
                first_lines = [next(f) for _ in range(5)]
            print("\nFirst lines of file (raw format):")
            for i, line in enumerate(first_lines):
                print(f"Line {i+1}: {line.strip()}")
        except Exception as inner_e:
            print(f"Could not inspect file: {inner_e}")
        raise
    
    # Analyze sentiment distribution
    print("\nSentiment distribution in training data:")
    sentiment_distribution = train_df["Sentiment"].value_counts()
    for sentiment, count in sentiment_distribution.items():
        percentage = (count / len(train_df)) * 100
        print(f"  {sentiment}: {count} ({percentage:.2f}%)")
    
    # Create separate test set
    # Split: 80% training, 10% validation, 10% test
    print("\nCreating train/val/test splits...")
    
    # Use original validation set as test set
    test_df = valid_df.copy()
    
    # Split original training into train/val
    train_val_df = train_df.copy()
    train_val_df = train_val_df.sample(frac=1, random_state=42)  # Shuffle data
    
    train_size = int(0.9 * len(train_val_df))
    train_df_split = train_val_df.iloc[:train_size]
    val_df = train_val_df.iloc[train_size:]
    
    print(f"Training split: {len(train_df_split)} examples")
    print(f"Validation split: {len(val_df)} examples")
    print(f"Test split: {len(test_df)} examples")
    
    # Function to process dataframe to appropriate format
    def process_dataframe(df):
        processed_data = []
        
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing data"):
            # Extract data from columns
            tweet = str(row["Tweet"]) if not pd.isna(row["Tweet"]) else ""
            entity = str(row["Entity"]) if not pd.isna(row["Entity"]) else ""
            sentiment = str(row["Sentiment"]) if not pd.isna(row["Sentiment"]) else ""
            
            # Tokenize tweet
            tokens = tweet.split()
            if not tokens:  # Skip empty tweets
                continue
            
            # Mark entities in text
            ner_tags = ["O"] * len(tokens)
            if entity and entity.lower() != "null" and entity.lower() != "irrelevant":
                entity_lower = entity.lower()
                for i, token in enumerate(tokens):
                    if entity_lower in token.lower():
                        ner_tags[i] = "B-ENT"
                        # If there are more tokens that seem part of the entity, mark them as I-ENT
                        if i < len(tokens) - 1 and entity_lower in " ".join(tokens[i:i+2]).lower():
                            ner_tags[i+1] = "I-ENT"
            
            # Map sentiment to consistent format
            sentiment = sentiment.lower()
            if sentiment in ["positive", "pos"]:
                sentiment_label = "POS"
            elif sentiment in ["negative", "neg"]:
                sentiment_label = "NEG"
            elif sentiment in ["neutral", "neu"]:
                sentiment_label = "NEU"
            else:
                # If 'irrelevant' or another unrecognized value, skip it
                continue
            
            # Save processed example
            processed_data.append({
                "tokens": tokens,
                "ner_tags": ner_tags,
                "sentiment_label": sentiment_label,
                "original_tweet": tweet,
                "original_entity": entity
            })
        
        return Dataset.from_list(processed_data)
    
    # Process datasets
    print("\nProcessing datasets...")
    train_dataset = process_dataframe(train_df_split)
    val_dataset = process_dataframe(val_df)
    test_dataset = process_dataframe(test_df)
    
    # Combine into a DatasetDict
    processed_dataset = DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset
    })
    
    # Analyze final distribution
    for split in processed_dataset:
        # Sentiment distribution
        sentiment_counter = Counter(processed_dataset[split]["sentiment_label"])
        total = len(processed_dataset[split])
        
        print(f"\n{split.upper()} - Sentiment distribution:")
        for label, count in sorted(sentiment_counter.items()):
            percentage = (count / total) * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
        
        # NER tag distribution
        ner_counter = Counter()
        for tags in processed_dataset[split]["ner_tags"]:
            ner_counter.update(tags)
        
        print(f"{split.upper()} - NER tags:")
        for tag, count in sorted(ner_counter.items()):
            percentage = (count / sum(ner_counter.values())) * 100
            print(f"  {tag}: {count} ({percentage:.2f}%)")
    
    # Save processed dataset
    output_path = os.path.join(output_dir, "joint_ner_sentiment_dataset")
    processed_dataset.save_to_disk(output_path)
    print(f"\nProcessed dataset saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    prepare_twitter_sentiment_dataset()

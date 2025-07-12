"""
Complete Inference Pipeline

This script loads the three models of the project (NER, SA, and Alert Generator)
and performs inference on a text file, displaying the results in an integrated manner.
"""

import os
import sys
import json
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from colorama import Fore, Style, init

# Initialize colorama
init()

# Add the path to the src directory to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train.ner_train.predictor import PipelinePredictor
from train.ner_train import config as ner_config
from train.sa_train.model import SentimentFromTextNerLSTM

def load_ner_model():
    """Loads the NER model and its configuration."""
    model_path = os.path.join(ner_config.MODEL_OUTPUT_DIR, "named_entity_recognition_model.pt")
    vocab_cache_dir = os.path.join(ner_config.DATA_DIR, "ner_cache")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: NER model not found at {model_path}")
    
    predictor = PipelinePredictor(model_path=model_path, vocab_cache_dir=vocab_cache_dir)
    print(f"[INFO]: NER model successfully loaded from {model_path}")
    
    return predictor

def load_vocabularies(vocab_dir):
    """
    Loads the SA model vocabularies from pickle files.
    Creates defaultdict objects with consistent behavior for unknown tokens.
    
    Args:
        vocab_dir: Directory where vocabulary files are located
        
    Returns:
        text_vocab: Vocabulary for text tokens
        ner_vocab: Vocabulary for NER tags
        sentiment_vocab: Vocabulary for sentiment labels
        sentiment_vocab_inverse: Mapping from indices to sentiment labels
    """
    import pickle
    from collections import defaultdict
    
    # Define file paths
    text_vocab_path = os.path.join(vocab_dir, "sentiment_text_vocabulary.pkl")
    ner_vocab_path = os.path.join(vocab_dir, "sentiment_ner_tags_vocabulary.pkl")
    sentiment_vocab_path = os.path.join(vocab_dir, "sentiment_labels_vocabulary.pkl")
    
    # Load text vocabulary
    with open(text_vocab_path, 'rb') as f:
        text_vocab_dict = pickle.load(f)
        text_vocab = defaultdict(lambda: text_vocab_dict.get('<UNK>', 1))
        for k, v in text_vocab_dict.items():
            text_vocab[k] = v
    
    # Load NER vocabulary
    with open(ner_vocab_path, 'rb') as f:
        ner_vocab_dict = pickle.load(f)
        ner_vocab = defaultdict(lambda: ner_vocab_dict.get('O', 1))
        for k, v in ner_vocab_dict.items():
            ner_vocab[k] = v
    
    # Load sentiment vocabulary
    with open(sentiment_vocab_path, 'rb') as f:
        sentiment_vocab_dict = pickle.load(f)
        sentiment_vocab = defaultdict(lambda: sentiment_vocab_dict.get('NEU', 1))
        for k, v in sentiment_vocab_dict.items():
            sentiment_vocab[k] = v
    
    # Create inverse mapping for sentiment labels
    sentiment_vocab_inverse = {v: k for k, v in sentiment_vocab.items()}
    
    print(f"[INFO]: Vocabularies successfully loaded from {vocab_dir}")
    print(f"[INFO]: Sizes: text={len(text_vocab)}, NER={len(ner_vocab)}, sentiment={len(sentiment_vocab)}")
    
    return text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse

def load_sa_model():
    """Loads the sentiment analysis model and its vocabularies."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_path = os.path.join(project_root, "src", "models", "sa_model", "sentiment_analysis_model.pth")
    vocab_dir = os.path.join(project_root, "src", "data", "sa_vocabs")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Error: SA model not found at {model_path}")
    
    # Load vocabularies using the helper function
    text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse = load_vocabularies(vocab_dir)
    
    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SentimentFromTextNerLSTM(
        text_vocab_size=len(text_vocab),
        text_embedding_dim=200,
        ner_vocab_size=len(ner_vocab),
        ner_embedding_dim=50,
        hidden_dim=256,
        sentiment_vocab_size=len(sentiment_vocab),
        n_layers=2,
        bidirectional=True,
        dropout=0.5
    )
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"[INFO]: SA model successfully loaded from {model_path}")
    
    return model, text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse, device

def load_alert_generator_model():
    """Loads the T5-based alert generator model."""
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    model_dir = os.path.join(project_root, "src", "models", "alert_generator_model")
    
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Error: Alert model directory not found at {model_dir}")
    
    # Load model and tokenizer
    try:
        model = T5ForConditionalGeneration.from_pretrained(model_dir)
        tokenizer = T5Tokenizer.from_pretrained(model_dir)
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        
        print(f"[INFO]: Alert generator model successfully loaded from {model_dir}")
        
        return model, tokenizer, device
    except Exception as e:
        print(f"[ERROR]: Error loading alert generator model: {e}")
        raise

def predict_sentiment(sa_model, text, ner_tags, text_vocab, ner_vocab, sentiment_vocab_inverse, device):
    """Predicts the sentiment of a text using the SA model."""
    # Tokenize the text
    tokens = text.lower().strip().split()
    
    # Encode tokens
    encoded_text = []
    for token in tokens:
        token_lower = token.lower()
        if token_lower in text_vocab:
            encoded_text.append(text_vocab[token_lower])
        else:
            encoded_text.append(text_vocab['<UNK>'])
    
    # Prepare NER tags
    # If NER tags are a dictionary (token -> tag), convert to list
    if isinstance(ner_tags, dict):
        encoded_ner = []
        for token in tokens:
            tag = ner_tags.get(token, "O")  # Use 'O' as default value
            encoded_ner.append(ner_vocab[tag])
    else:
        # If already a list, use directly
        encoded_ner = [ner_vocab.get(tag, ner_vocab['O']) for tag in ner_tags]
    
    # Truncate if necessary
    max_len = 128
    if len(encoded_text) > max_len:
        encoded_text = encoded_text[:max_len]
        encoded_ner = encoded_ner[:max_len]
    
    # Convert to tensors
    text_tensor = torch.tensor([encoded_text], dtype=torch.long).to(device)
    ner_tensor = torch.tensor([encoded_ner], dtype=torch.long).to(device)
    
    # Perform prediction
    with torch.no_grad():
        output = sa_model(text_tensor, ner_tensor)
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        pred_class_idx = pred_class.item()
        confidence_val = confidence.item()
    
    # Get sentiment label
    sentiment = sentiment_vocab_inverse.get(pred_class_idx, "NEU")
    
    # Mapping of sentiment labels
    sentiment_mapping = {"NEG": "negative", "POS": "positive", "NEU": "neutral"}
    sentiment = sentiment_mapping.get(sentiment, "neutral")
    
    return sentiment, confidence_val

def generate_alert(alert_model, tokenizer, text, ner_result, sentiment, device):
    """Generates an alert using the generator model."""
    # Input format for the model
    ner_text = " ".join([f"{token}:{tag}" for token, tag in ner_result.items()])
    input_text = f"text: {text} ner: {ner_text} sentiment: {sentiment}"
    
    # Tokenize input
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
    
    # Generate alert
    with torch.no_grad():
        outputs = alert_model.generate(
            input_ids,
            max_length=100,
            num_beams=4,
            early_stopping=True,
            temperature=0.7,
            no_repeat_ngram_size=2,
            do_sample=True
        )
    
    # Decode result
    alert = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return alert

def print_colored_result(text, ner_result, sentiment, confidence, alert):
    """Prints results with formatting and colors for better visualization."""
    # Define colors based on sentiment
    sentiment_color = Fore.GREEN
    if sentiment == "negative":
        sentiment_color = Fore.RED
    elif sentiment == "neutral":
        sentiment_color = Fore.BLUE
    
    # Header
    print("\n" + "="*80)
    print(f"{Fore.CYAN}ANALYZED TEXT:{Style.RESET_ALL}")
    print(f"{text}")
    print("-"*80)
    
    # NER results
    print(f"{Fore.YELLOW}RECOGNIZED ENTITIES:{Style.RESET_ALL}")
    for token, tag in ner_result.items():
        tag_color = Fore.WHITE
        if "PER" in tag:
            tag_color = Fore.MAGENTA
        elif "LOC" in tag:
            tag_color = Fore.GREEN
        elif "ORG" in tag:
            tag_color = Fore.BLUE
        elif "MISC" in tag:
            tag_color = Fore.CYAN
        
        if tag != "O":
            print(f"  {token}: {tag_color}{tag}{Style.RESET_ALL}")
    
    # Sentiment results
    print(f"{Fore.YELLOW}SENTIMENT ANALYSIS:{Style.RESET_ALL}")
    print(f"  {sentiment_color}{sentiment}{Style.RESET_ALL} (Confidence: {confidence:.2f})")
    
    # Generated alert
    print(f"{Fore.YELLOW}GENERATED ALERT:{Style.RESET_ALL}")
    print(f"  {Fore.WHITE}{Style.BRIGHT}{alert}{Style.RESET_ALL}")
    
    print("="*80)

def main():
    """Main function to perform the complete inference pipeline."""
    try:
        # Load the three models
        print("[INFO]: Loading models...")
        ner_predictor = load_ner_model()
        sa_model, text_vocab, ner_vocab, sentiment_vocab, sentiment_vocab_inverse, sa_device = load_sa_model()
        alert_model, alert_tokenizer, alert_device = load_alert_generator_model()
        print("[INFO]: All models successfully loaded")
        
        # Load input data
        input_file = os.path.join(ner_config.DATA_DIR, "news_tweets.txt")
        if not os.path.exists(input_file):
            print(f"[ERROR]: Error: Input file not found at {input_file}")
            return
        
        # Read text file
        with open(input_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        print(f"[INFO]: Loaded {len(lines)} lines to process")
        
        # Process each line
        num_examples = len(lines)  
        for i, text in enumerate(lines[:num_examples], 1):
            print(f"[INFO]: Processing text {i}/{num_examples}: '{text[:50]}...'")
            
            # 1. Named Entity Recognition (NER)
            # Use the public prediction method instead of the private method
            ner_prediction = ner_predictor.predict(text)
            ner_result = ner_prediction["ner_result"]
            
            # 2. Sentiment Analysis (SA)
            sentiment, confidence = predict_sentiment(
                sa_model, text, ner_result, text_vocab, ner_vocab, 
                sentiment_vocab_inverse, sa_device
            )
            
            # 3. Alert Generation
            alert = generate_alert(
                alert_model, alert_tokenizer, text, ner_result, 
                sentiment, alert_device
            )
            
            # 4. Display formatted results
            print_colored_result(text, ner_result, sentiment, confidence, alert)
            
            # Result as JSON dictionary (optional for export)
            result = {
                "text": text,
                "ner_result": ner_result,
                "sentiment": {
                    "label": sentiment,
                    "confidence": confidence
                },
                "alert": alert
            }
        
        print("[INFO]: Processing completed")
    
    except FileNotFoundError as e:
        print(f"[ERROR]: Error: {e}")
    except Exception as e:
        print(f"[ERROR]: Error during inference: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
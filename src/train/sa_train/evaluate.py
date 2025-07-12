import os
import torch
from data import load_data, NER_UNK_TOKEN, sentiment_pipeline, SENTIMENT_LABEL_MAP
from train_functions import t_step
from utils import set_seed
from model import SentimentFromTextNerLSTM
import pickle
from collections import defaultdict, Counter
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

def load_vocab_with_default(path, unk_token=None):
    """Loads vocabulary from pickle file and sets up defaultdict behavior."""
    try:
        with open(path, 'rb') as f:
            vocab_dict = pickle.load(f)
            vocab_dict_copy = dict(vocab_dict)
            
            if unk_token and unk_token in vocab_dict_copy:
                unk_index = vocab_dict_copy[unk_token]
                vocab = defaultdict(lambda: unk_index, vocab_dict_copy)
            elif '<PAD>' in vocab_dict_copy:
                pad_index = vocab_dict_copy['<PAD>']
                vocab = defaultdict(lambda: pad_index, vocab_dict_copy)
            else:
                vocab = defaultdict(int, vocab_dict_copy)
                
            print(f"Loaded vocab from {path} with {len(vocab)} items")
            return vocab
    except FileNotFoundError:
        print(f"ERROR: Vocab file not found at {path}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to load vocab from {path}: {e}")
        return None

def predict_sentiment(model, sentence, text_vocab, ner_vocab, sentiment_vocab_inverse, device):
    """Predicts sentiment for a tweet or text sentence."""
    tokens = sentence.lower().strip().split()
    
    encoded_text = []
    for token in tokens:
        if token.lower() in text_vocab:
            encoded_text.append(text_vocab[token.lower()])
        else:
            encoded_text.append(text_vocab[NER_UNK_TOKEN])
    
    encoded_ner = [ner_vocab[NER_UNK_TOKEN]] * len(tokens)
    
    max_len = 128
    if len(encoded_text) > max_len:
        encoded_text = encoded_text[:max_len]
        encoded_ner = encoded_ner[:max_len]
    
    text_tensor = torch.tensor([encoded_text], dtype=torch.long).to(device)
    ner_tensor = torch.tensor([encoded_ner], dtype=torch.long).to(device)
    
    with torch.no_grad():
        prediction = model(text_tensor, ner_tensor)
        probs = torch.nn.functional.softmax(prediction, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)
        
        pred_class = pred_class.item()
        confidence = confidence.item()
        
    sentiment_label = sentiment_vocab_inverse.get(pred_class, "UNK")
    
    return (sentence, sentiment_label, confidence)

def calculate_class_metrics(model, test_loader, sentiment_vocab_inverse, device):
    """Calculates metrics by class on the test set."""
    model.eval()
    
    all_predictions = []
    all_labels = []
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    with torch.no_grad():
        for text_data, ner_data, target_sentiment_data in tqdm(test_loader, desc="Calculating class metrics"):
            text_data = text_data.to(device)
            ner_data = ner_data.to(device)
            target_sentiment_data = target_sentiment_data.to(device)
            
            predictions = model(text_data, ner_data)
            preds = torch.argmax(predictions, dim=1)
            
            all_predictions.extend(preds.cpu().numpy())
            all_labels.extend(target_sentiment_data.cpu().numpy())
            
            for true_label, pred_label in zip(target_sentiment_data, preds):
                true_class = true_label.item()
                pred_class = pred_label.item()
                
                if true_class == pred_class:
                    class_correct[true_class] += 1
                
                class_total[true_class] += 1
    
    # Show accuracy by class
    print("\nAccuracy by class:")
    for class_idx in sorted(class_total.keys()):
        if class_total[class_idx] > 0:
            accuracy = class_correct[class_idx] / class_total[class_idx]
            class_name = sentiment_vocab_inverse.get(class_idx, f"Class {class_idx}")
            print(f"  {class_name}: {accuracy:.4f} ({class_correct[class_idx]}/{class_total[class_idx]})")
    
    # Show class distribution
    print("\nClass distribution in test set:")
    total_examples = sum(class_total.values())
    for class_idx in sorted(class_total.keys()):
        percentage = (class_total[class_idx] / total_examples) * 100
        class_name = sentiment_vocab_inverse.get(class_idx, f"Class {class_idx}")
        print(f"  {class_name}: {class_total[class_idx]} ({percentage:.2f}%)")
    
    # Generate confusion matrix
    try:
        # Get class names for visualization
        class_names = [sentiment_vocab_inverse.get(i, f"Class {i}") 
                      for i in range(max(sentiment_vocab_inverse.keys()) + 1)]
        
        cm = confusion_matrix(all_labels, all_predictions)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        print("\nConfusion matrix (absolute values):")
        print(cm)
        print("\nConfusion matrix (normalized by true class):")
        print(cm_normalized)
        
        print("\nClassification report:")
        print(classification_report(all_labels, all_predictions, 
                                   target_names=[c for c in class_names if c != '<PAD>']))
        
        # Visualize confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Prediction')
        plt.ylabel('True Label')
        plt.title('Normalized Confusion Matrix')
        plt.tight_layout()
        
        os.makedirs('figures', exist_ok=True)
        plt.savefig('figures/confusion_matrix.png')
        print("\nConfusion matrix saved to figures/confusion_matrix.png")
        
    except Exception as e:
        print(f"Error generating confusion matrix: {e}")

def main():
    # --- Setup ---
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Paths ---
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
    MODEL_SAVE_DIR = os.path.join(PROJECT_ROOT, "src", "models", "sa_model")
    DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
    VOCAB_SAVE_DIR = os.path.join(DATA_DIR, "sa_vocabs")
    MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "sentiment_analysis_model.pth")
    TEXT_VOCAB_PATH = os.path.join(VOCAB_SAVE_DIR, "sentiment_text_vocabulary.pkl")
    NER_VOCAB_PATH = os.path.join(VOCAB_SAVE_DIR, "sentiment_ner_tags_vocabulary.pkl")
    SENTIMENT_VOCAB_PATH = os.path.join(VOCAB_SAVE_DIR, "sentiment_labels_vocabulary.pkl")

    # --- Hyperparameters ---
    BATCH_SIZE = 32
    MAX_LEN = 128
    TEXT_EMBEDDING_DIM = 200
    NER_EMBEDDING_DIM = 50
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5

    # --- Load Vocabularies ---
    print(f"Loading sentiment analysis vocabularies from {VOCAB_SAVE_DIR}...")
    text_vocab = load_vocab_with_default(TEXT_VOCAB_PATH, unk_token='<UNK>')
    ner_vocab = load_vocab_with_default(NER_VOCAB_PATH, unk_token='O')
    sentiment_vocab = load_vocab_with_default(SENTIMENT_VOCAB_PATH, unk_token='NEU')

    if not all([text_vocab, ner_vocab, sentiment_vocab]):
        print("Failed to load vocabularies. Exiting.")
        return
    
    print(f"Vocabulary sizes: Text={len(text_vocab)}, NER={len(ner_vocab)}, Sentiment={len(sentiment_vocab)}")

    # Create inverse sentiment vocabulary (index -> label)
    sentiment_vocab_inverse = {idx: label for label, idx in sentiment_vocab.items()}
    print(f"Sentiment classes: {sentiment_vocab_inverse}")

    # --- Load Test Data ---
    print("Loading sentiment analysis test data...")
    
    try:
        from prepare_twitter_dataset import prepare_twitter_sentiment_dataset
        if not os.path.exists(os.path.join(DATA_DIR, "sentiment_analysis_dataset/joint_ner_sentiment_dataset")):
            print("Sentiment analysis dataset not found, generating it first...")
            prepare_twitter_sentiment_dataset()
    except Exception as e:
        print(f"Warning: Could not prepare dataset: {e}")
    
    try:
        _, _, test_loader, _, _, _ = load_data(
            batch_size=BATCH_SIZE,
            max_len=MAX_LEN,
            text_vocab=text_vocab,
            ner_vocab=ner_vocab,
            sentiment_vocab=sentiment_vocab
        )
    except Exception as e:
        print(f"Error loading test data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Initialize Model ---
    print("Initializing model...")
    model = SentimentFromTextNerLSTM(
        text_vocab_size=len(text_vocab),
        text_embedding_dim=TEXT_EMBEDDING_DIM,
        ner_vocab_size=len(ner_vocab),
        ner_embedding_dim=NER_EMBEDDING_DIM,
        hidden_dim=HIDDEN_DIM,
        sentiment_vocab_size=len(sentiment_vocab),
        n_layers=N_LAYERS,
        bidirectional=BIDIRECTIONAL,
        dropout=DROPOUT
    )
    
    # --- Load Model Weights ---
    print(f"Loading model state from: {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model state file not found at {MODEL_PATH}")
        print("Please ensure the training script ran successfully.")
        return
    except Exception as e:
        print(f"ERROR: Failed to load model state: {e}")
        return

    model.to(device)
    model.eval()

    # --- Evaluate Model on Test Set ---
    print("Evaluating model on Twitter test set...")
    accuracy = t_step(model, test_loader, device)
    print(f"Test Accuracy: {accuracy*100:.2f}%")
    
    # --- Evaluate Model on Example Tweets ---
    print("\n--- Evaluating model on example tweets ---")
    
    example_tweets = [
        # Positive tweets
        "I absolutely love this new phone! Best purchase ever! #happy",
        "Just had an amazing dinner at @FancyRestaurant! Their service is top notch! 😍",
        "SpaceX launch was incredible to watch! Congratulations @elonmusk and team! #space",
        
        # Negative tweets
        "This airline lost my luggage again. Worst customer service ever. #angry",
        "Can't believe how bad the traffic is today. I'm going to be late for work. 😠",
        "The new update completely ruined the app. Going back to the old version.",
        
        # Neutral tweets
        "Just posted a new video about machine learning. Check it out! #AI #tech",
        "Reminder: the meeting has been moved to 3pm today. @TeamMembers",
        "Interesting article about climate change in the New York Times today."
    ]
    
    # Predict sentiment for each tweet with our model
    results = [predict_sentiment(model, sentence, text_vocab, ner_vocab, sentiment_vocab_inverse, device) 
               for sentence in example_tweets]
    
    # Predict sentiment with pretrained pipeline for comparison
    print("\nChecking pretrained sentiment pipeline...")
    pretrained_results = []
    with torch.no_grad():
        for sentence in example_tweets:
            try:
                result = sentiment_pipeline(sentence, truncation=True)
                label = result[0]['label']
                mapped_label = SENTIMENT_LABEL_MAP.get(label, "UNK")
                pretrained_results.append((sentence, mapped_label))
            except Exception as e:
                print(f"Error with pipeline on tweet '{sentence[:30]}...': {e}")
                pretrained_results.append((sentence, "ERROR"))
    
    # Show comparative results
    print("\nSentiment prediction comparison on tweets:")
    print("-" * 95)
    print(f"{'TWEET':<45} | {'MODEL':<10} | {'CONF.':<8} | {'PIPELINE':<10}")
    print("-" * 95)
    
    for i, (sentence, model_sentiment, confidence) in enumerate(results):
        display_sentence = sentence[:42] + "..." if len(sentence) > 42 else sentence
        pipeline_sentiment = pretrained_results[i][1]
        conf_str = f"{confidence:.2f}"
        print(f"{display_sentence:<45} | {model_sentiment:<10} | {conf_str:<8} | {pipeline_sentiment:<10}")
    
    print("-" * 95)
    
    # --- Confusion matrix for test set ---
    print("\n--- Confusion matrix on Twitter test set ---")
    calculate_class_metrics(model, test_loader, sentiment_vocab_inverse, device)
    
    # --- Evaluate on challenging tweets ---
    print("\n--- Evaluating model on challenging tweets ---")
    
    challenging_tweets = [
        # Tweets with mixed sentiment
        "The new iPhone camera is amazing but the battery life is terrible. #mixedfeelings",
        "I'm so happy about my promotion but sad to leave my team. 😊😢",
        "Conference was great overall, despite the horrible Wi-Fi and cold food.",
        
        # Tweets with sarcasm/irony
        "Sure, because waking up to no hot water is EXACTLY what I needed today. #sarcasm",
        "Wow, getting stuck in the rain without an umbrella. Lucky me!",
        "Nothing like spending 3 hours on hold with customer service. #bestdayever",
        
        # Tweets with entities and hashtags
        "Apple's new MacBook is super expensive but absolutely worth it. #Apple",
        "Cannot believe @Tesla stock dropped 10% today after their announcement!",
        "@BurgerKing's new plant-based burger tastes awful compared to @ImpossibleFoods"
    ]
    
    # Predict sentiment for each tweet
    challenge_results = [predict_sentiment(model, sentence, text_vocab, ner_vocab, sentiment_vocab_inverse, device) 
                         for sentence in challenging_tweets]
    
    # Show results
    print("\nResults on challenging tweets:")
    for sentence, sentiment, confidence in challenge_results:
        print(f"Tweet: {sentence}")
        print(f"  Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        print()

if __name__ == "__main__":
    main()


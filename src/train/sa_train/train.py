import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from src.train.sa_train.utils import set_seed
from src.train.sa_train.data import load_data
from src.train.sa_train.train_functions import train_step, val_step
from src.train.sa_train.model import SentimentFromTextNerLSTM
import pickle
from collections import Counter
import numpy as np
from src.train.sa_train.prepare_twitter_dataset import prepare_twitter_sentiment_dataset

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
    BEST_MODEL_PATH = os.path.join(MODEL_SAVE_DIR, "sentiment_analysis_model.pth")
    RUNS_DIR = os.path.join(SCRIPT_DIR, "./runs/sentiment_analysis_experiment")
    
    TEXT_VOCAB_PATH = os.path.join(VOCAB_SAVE_DIR, "sentiment_text_vocabulary.pkl")
    NER_VOCAB_PATH = os.path.join(VOCAB_SAVE_DIR, "sentiment_ner_tags_vocabulary.pkl")
    SENTIMENT_VOCAB_PATH = os.path.join(VOCAB_SAVE_DIR, "sentiment_labels_vocabulary.pkl")
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(VOCAB_SAVE_DIR, exist_ok=True)

    # --- Hyperparameters ---
    BATCH_SIZE = 32
    MAX_LEN = 128
    TEXT_EMBEDDING_DIM = 200
    NER_EMBEDDING_DIM = 50
    HIDDEN_DIM = 256
    N_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.5
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 3e-5
    EPOCHS = 70
    PATIENCE = 7

    # --- Prepare Twitter dataset ---
    prepare_twitter_sentiment_dataset()

    # --- Data Loading ---
    print("Loading sentiment analysis data...")
    train_loader, val_loader, test_loader, text_vocab, ner_vocab, sentiment_vocab = load_data(
        batch_size=BATCH_SIZE,
        max_len=MAX_LEN,
    )
    
    # --- Save Vocabularies ---
    print(f"Saving vocabularies to {VOCAB_SAVE_DIR}...")
    
    with open(TEXT_VOCAB_PATH, 'wb') as f:
        text_vocab_dict = dict(text_vocab)
        pickle.dump(text_vocab_dict, f)
        print(f"Saved text vocabulary with {len(text_vocab_dict)} items to {TEXT_VOCAB_PATH}")
    
    with open(NER_VOCAB_PATH, 'wb') as f:
        ner_vocab_dict = dict(ner_vocab)
        pickle.dump(ner_vocab_dict, f)
        print(f"Saved NER tags vocabulary with {len(ner_vocab_dict)} items to {NER_VOCAB_PATH}")
    
    with open(SENTIMENT_VOCAB_PATH, 'wb') as f:
        sentiment_vocab_dict = dict(sentiment_vocab)
        pickle.dump(sentiment_vocab_dict, f)
        print(f"Saved sentiment labels vocabulary with {len(sentiment_vocab_dict)} items to {SENTIMENT_VOCAB_PATH}")
    
    # --- Analysis of sentiment labels ---
    sentiment_vocab_inverse = {v: k for k, v in sentiment_vocab.items()}
    train_sentiments = [sentiment_vocab_inverse.get(idx, "UNK") for idx in range(len(sentiment_vocab))]
    print(f"Sentiment classes: {train_sentiments}")

    # Analyze class distribution in dataloaders
    def analyze_loader_distribution(loader, name):
        all_labels = []
        for _, _, sentiments in loader:
            all_labels.extend(sentiments.numpy())
        
        unique_labels = np.unique(all_labels)
        label_counts = Counter(all_labels)
        total = len(all_labels)
        
        print(f"\nClass distribution in {name}:")
        for label in unique_labels:
            count = label_counts[label]
            print(f"  Class {label} ({sentiment_vocab_inverse.get(label, 'UNK')}): {count} ({count/total*100:.2f}%)")
        
        return label_counts

    train_counts = analyze_loader_distribution(train_loader, "train_loader")
    val_counts = analyze_loader_distribution(val_loader, "val_loader")
    test_counts = analyze_loader_distribution(test_loader, "test_loader")

    # --- Model Initialization ---
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
    model.to(device)
    print(model)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # --- Training Setup ---
    class_weights = None
    if hasattr(train_loader.dataset, 'encoded_sentiment'):
        try:
            labels = train_loader.dataset['encoded_sentiment'].numpy()
            class_counts = np.bincount(labels)
            
            weight_per_class = len(labels) / (len(class_counts) * class_counts)
            class_weights = torch.tensor(weight_per_class, dtype=torch.float32).to(device)
            
            print(f"Class weights for balancing:")
            for i, weight in enumerate(class_weights):
                class_name = sentiment_vocab_inverse.get(i, f"Class {i}")
                print(f"  {class_name}: {weight.item():.4f}")
        except Exception as e:
            print(f"Error calculating class weights: {e}")
            class_weights = None
    
    if class_weights is not None:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        print("Using weighted Cross Entropy loss")
    else:
        criterion = nn.CrossEntropyLoss()
        print("Using standard Cross Entropy loss")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=2, verbose=True, min_lr=1e-6
    )
    writer = SummaryWriter(RUNS_DIR)
    
    best_val_acc = 0.0
    epochs_without_improvement = 0

    print(f"Training logs will be saved to: {RUNS_DIR}")
    print(f"Best model will be saved to: {BEST_MODEL_PATH}")

    # --- Training Loop ---
    print("Starting training with Twitter sentiment data...")
    for epoch in range(EPOCHS):
        train_step(model, train_loader, criterion, optimizer, writer, epoch, device)
        
        val_accuracy = val_step(model, val_loader, criterion, writer, epoch, device)

        scheduler.step(val_accuracy)

        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            epochs_without_improvement = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"--- Epoch {epoch+1}: New best model saved (val acc: {best_val_acc:.4f}) ---")
        else:
            epochs_without_improvement += 1
            print(f"--- Epoch {epoch+1}: No improvement. ({epochs_without_improvement}/{PATIENCE}) ---")

        if epochs_without_improvement >= PATIENCE:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"Training finished. Best validation accuracy: {best_val_acc:.4f}")
    writer.close()

if __name__ == "__main__":
    main()

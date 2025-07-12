import torch
import os

# --- General ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Removed device print statement, will be printed elsewhere if needed
SEED = 42

# --- Dataset ---
DATASET_NAME = "conll2003"
SPACY_MODEL = "en_core_web_trf"
MAX_SEQ_LEN = 128 # Max sequence length after spaCy/transformer tokenization. Adjust as needed.
MAX_WORD_LEN = 20 # Max word length for Char CNN

# --- Model ---
# Embeddings
SPACY_EMBEDDING_DIM = 768 # Depends on 'en_core_web_trf' model (RoBERTa-base)
POS_EMBEDDING_DIM = 50
DEP_EMBEDDING_DIM = 50
CHAR_EMBEDDING_DIM = 30
UNFREEZE_TRANSFORMER_LAYERS = 1 # Number of transformer layers to unfreeze (0 to freeze all)

# Char CNN
CHAR_CNN_FILTERS = 50 # Filters per kernel size
CHAR_CNN_KERNELS = [3, 4, 5] # Kernel sizes

# BiLSTM
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.33 # Recurrent dropout

# Attention (simple dot-product self-attention) - Currently unused
# ATTENTION_DIM = LSTM_HIDDEN_DIM * 2 # BiLSTM output dimension

# General Dropout (applied before the final layer)
DROPOUT_RATE = 0.5

# --- Training ---
BATCH_SIZE = 16 # Adjust based on GPU memory
EPOCHS = 70
LEARNING_RATE = 1e-4 # Initial learning rate for AdamW
TRANSFORMER_LEARNING_RATE = 2e-5 # Lower learning rate for transformer layers
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP_VAL = 1.0
EARLY_STOPPING_PATIENCE = 3
USE_AMP = True # Use Mixed Precision Training

# --- Paths ---
# Define project structure paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_DIR = os.path.join(PROJECT_ROOT, "src", "data")
MODEL_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "src", "models", "ner_model")
MODEL_SAVE_PATH = os.path.join(MODEL_OUTPUT_DIR, "ner_model_best.pt")
CACHE_DIR = os.path.join(DATA_DIR, "ner_cache")  # Changed to src/data/ner_cache
LOG_DIR = os.path.join(SCRIPT_DIR, "runs")  # Directory for TensorBoard logs

# --- Mappings (will be filled in data_utils) ---
pos_vocab = {}
dep_vocab = {}
ner_vocab = {}
char_vocab = {}

# Create directories if they don't exist
os.makedirs(MODEL_OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Print setup function for consistent output formatting
def get_logger(name):
    """Returns a print wrapper to replace the logger."""
    class PrintLogger:
        def __init__(self, prefix):
            self.prefix = prefix
        
        def info(self, message):
            print(f"[INFO] - {self.prefix} - {message}")
            
        def warning(self, message):
            print(f"[WARNING] - {self.prefix} - {message}")
            
        def error(self, message, exc_info=False):
            if exc_info:
                import traceback
                print(f"[ERROR] - {self.prefix} - {message}")
                traceback.print_exc()
            else:
                print(f"[ERROR] - {self.prefix} - {message}")
            
        def debug(self, message):
            # Debug messages can be enabled by uncommenting this
            # print(f"[DEBUG] - {self.prefix} - {message}")
            pass
    
    return PrintLogger(name)

# Print configuration summary
print(f"[CONFIG] - NER training configuration loaded")
print(f"[CONFIG] - Device: {DEVICE}")
print(f"[CONFIG] - Batch size: {BATCH_SIZE}")
print(f"[CONFIG] - Learning rate: {LEARNING_RATE}")
print(f"[CONFIG] - Model will be saved to: {MODEL_SAVE_PATH}")

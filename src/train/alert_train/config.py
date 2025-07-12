"""
Configuration parameters for the T5-based alert generator model.
"""
import os

# --- Model Configuration ---
MODEL_NAME = "google/flan-t5-small"
INCLUDE_TEXT = False
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 32

# --- Training Parameters ---
LEARNING_RATE = 5e-5
BATCH_SIZE = 8
NUM_EPOCHS = 150
WARMUP_STEPS = 50
WEIGHT_DECAY = 0.01
EVAL_STEPS = 20
SAVE_STEPS = 100

# --- Path Configuration ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(SCRIPT_DIR)))
DATA_PATH = os.path.join(PROJECT_ROOT, "src", "data", "alert_data.txt")
MODEL_DIR = os.path.join(PROJECT_ROOT, "src", "models", "alert_generator_model")
LOG_DIR = os.path.join(SCRIPT_DIR, "runs")

# --- Seed for reproducibility ---
RANDOM_SEED = 42
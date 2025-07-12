"""
Main training script for the alert generation model.
"""
import torch
import warnings
from . import config
from .utils import set_seed
from .data_utils import load_and_prepare_data
from .model_utils import (
    load_model,
    train_model,
    optimize_model_for_inference,
    run_inference
)

# Suppress warnings
warnings.filterwarnings("ignore")

def main():
    """Main function to run the training pipeline"""
    # Set random seed for reproducibility
    set_seed()
    
    # Check for available GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and prepare dataset
    tokenized_datasets, raw_datasets, tokenizer = load_and_prepare_data()
    
    # Load model
    model, _ = load_model(device)
    
    # Train model
    model, tokenizer = train_model(tokenized_datasets, tokenizer, model, config.MODEL_DIR)
    
    # Optimize model for inference
    inference_model, inference_tokenizer = optimize_model_for_inference()
    
    # Run inference with the optimized model
    print("\nRunning inference with optimized model...")
    inference_model = inference_model.to(device)
    run_inference(raw_datasets, inference_model, inference_tokenizer, device, "train", 3)
    run_inference(raw_datasets, inference_model, inference_tokenizer, device, "test", 3)


if __name__ == "__main__":
    main()
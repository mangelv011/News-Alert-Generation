"""
Model utilities for the alert generation model.
"""
import os
import torch
import shutil
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
)
from . import config
from .utils import compute_metrics

def load_model(device):
    """
    Load the pre-trained T5 model and move to device
    
    Args:
        device: Device to load the model on
    
    Returns:
        Loaded model and tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(config.MODEL_NAME).to(device)
    return model, tokenizer


def train_model(datasets, tokenizer, model, output_dir=config.MODEL_DIR):
    """
    Train the model with the given datasets
    
    Args:
        datasets: Dictionary containing tokenized datasets
        tokenizer: Tokenizer for the model
        model: Model to train
        output_dir: Directory to save the model
        
    Returns:
        Trained model and tokenizer
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "logs"), exist_ok=True)
    
    # Configure training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="steps",
        eval_steps=config.EVAL_STEPS,
        save_strategy="steps",
        save_steps=config.SAVE_STEPS,
        learning_rate=config.LEARNING_RATE,
        per_device_train_batch_size=config.BATCH_SIZE,
        per_device_eval_batch_size=config.BATCH_SIZE,
        weight_decay=config.WEIGHT_DECAY,
        num_train_epochs=config.NUM_EPOCHS,
        predict_with_generate=True,
        generation_max_length=config.MAX_TARGET_LENGTH,
        generation_num_beams=4,
        load_best_model_at_end=True,
        metric_for_best_model="rouge1",
        greater_is_better=True,
        warmup_steps=config.WARMUP_STEPS,
        logging_steps=10,
        logging_dir=os.path.join(output_dir, "logs"),
        report_to="tensorboard",
        fp16=False,
        gradient_accumulation_steps=4,
    )
    
    # Initialize trainer with a metric computation function that includes tokenizer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        tokenizer=tokenizer,
        compute_metrics=lambda eval_preds: compute_metrics(eval_preds, tokenizer),
    )
    
    # Train the model
    print("Starting model training...")
    trainer.train()
    
    # Save model and tokenizer
    print(f"Saving model to {output_dir}")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return model, tokenizer


def optimize_model_for_inference(model_dir=config.MODEL_DIR):
    """
    Optimize the model for inference, removing training artifacts
    and saving only essential components in the same folder.
    
    Args:
        model_dir: Model directory
        
    Returns:
        Optimized model and tokenizer
    """
    print(f"Optimizing model for inference in {model_dir}")
    
    # Create a temporary directory for optimization
    temp_dir = os.path.join(model_dir, "temp_optimization")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Load model and tokenizer
        model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        
        # Save model in optimized format in temporary directory
        print(f"Saving model in temporary directory: {temp_dir}")
        model.save_pretrained(
            temp_dir,
            is_main_process=True,
            save_function=torch.save,
            push_to_hub=False,
            safe_serialization=False  # Disable safetensors to avoid file access issues
        )
        
        # Save tokenizer in temporary directory
        tokenizer.save_pretrained(temp_dir)
        
        # Remove training artifacts from original directory
        training_artifacts = [
            "optimizer.pt", 
            "rng_state.pth", 
            "scheduler.pt", 
            "trainer_state.json", 
            "training_args.bin"
        ]
        
        for file in training_artifacts:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                print(f"Removing training artifact: {file_path}")
                os.remove(file_path)
        
        # Remove checkpoints from original directory
        for item in os.listdir(model_dir):
            if item.startswith("checkpoint-"):
                checkpoint_dir = os.path.join(model_dir, item)
                if os.path.isdir(checkpoint_dir):
                    print(f"Removing checkpoint: {checkpoint_dir}")
                    shutil.rmtree(checkpoint_dir)
        
        # Copy files from temporary directory to original directory
        print(f"Copying optimized files from {temp_dir} to {model_dir}")
        for item in os.listdir(temp_dir):
            source = os.path.join(temp_dir, item)
            destination = os.path.join(model_dir, item)
            
            if os.path.isdir(source):
                if os.path.exists(destination):
                    shutil.rmtree(destination)
                shutil.copytree(source, destination)
            else:
                shutil.copy2(source, destination)
        
        print("Model optimized and checkpoints successfully removed")
    except Exception as e:
        print(f"Error during model optimization: {str(e)}")
        raise
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            print(f"Removing temporary directory: {temp_dir}")
            shutil.rmtree(temp_dir)
    
    # Reload the optimized model
    model = AutoModelForSeq2SeqLM.from_pretrained(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    
    return model, tokenizer


def load_optimized_model_for_inference(model_dir=config.MODEL_DIR):
    """
    Load an optimized model for inference only
    
    Args:
        model_dir: Directory containing the optimized model
        
    Returns:
        Model and tokenizer ready for inference
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_dir,
        device_map="auto",        # Automatically choose best device
        torch_dtype=torch.float16  # Use half precision for faster inference
    )
    model.eval()  # Set model to evaluation mode
    
    # Apply memory optimization
    if hasattr(model, "config"):
        model.config.use_cache = True  # Enable caching for faster inference
    
    return model, tokenizer


def run_inference(datasets, model, tokenizer, device, dataset_type="test", num_examples=7):
    """
    Run inference on random examples from the dataset
    
    Args:
        datasets: Dictionary containing datasets
        model: Model for inference
        tokenizer: Tokenizer for the model
        device: Device to run inference on
        dataset_type: Type of dataset to use (train or test)
        num_examples: Number of examples to run inference on
    """
    import random
    from .utils import format_input
    
    print(f"\n=== INFERENCE ON {num_examples} RANDOM {dataset_type.upper()} EXAMPLES ===")
    sample_indices = random.sample(range(len(datasets[dataset_type])), num_examples)
    model.eval()

    for i, idx in enumerate(sample_indices):
        example = datasets[dataset_type][idx]
        input_text = format_input(example)
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=config.MAX_TARGET_LENGTH, 
                num_beams=4, 
                early_stopping=True,
                use_cache=True
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        reference = example["alert"]
        
        print(f"\nExample {i+1}:")
        print(f"  Input: {input_text}")
        print(f"  Reference: {reference}")
        print(f"  Prediction: {prediction}")
        print("-" * 50)
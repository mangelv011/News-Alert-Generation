import os
import re
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

@torch.enable_grad()
def train_step(model: torch.nn.Module,
               train_data: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               writer: SummaryWriter,
               epoch: int,
               device: torch.device) -> None:
    """
    Performs one epoch of training.
    
    Args:
        model: The neural network model
        train_data: DataLoader for training data
        loss_fn: Loss function
        optimizer: Optimizer for parameter updates
        writer: TensorBoard writer
        epoch: Current epoch number
        device: Device to run training on
    """
    model.train()
    losses = []
    correct_preds = 0
    total_preds = 0

    # Training loop
    for text_data, ner_data, target_sentiment_data in tqdm(train_data, desc=f"Epoch {epoch+1} [Train]"):
        # Move data to device
        text_data = text_data.to(device)
        ner_data = ner_data.to(device)
        target_sentiment_data = target_sentiment_data.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(text_data, ner_data)
        
        # Compute loss and update
        loss = loss_fn(predictions, target_sentiment_data)
        loss.backward()
        optimizer.step()
        
        # Track metrics
        losses.append(loss.item())
        preds = torch.argmax(predictions, dim=1)
        correct_preds += torch.sum(preds == target_sentiment_data).item()
        total_preds += target_sentiment_data.size(0)

    # Compute epoch metrics
    avg_loss = sum(losses) / len(losses) if losses else 0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    
    # Log metrics
    writer.add_scalar("train/loss", avg_loss, epoch)
    writer.add_scalar("train/accuracy", accuracy, epoch)
    print(f"Epoch {epoch+1} TRAIN -> Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")


@torch.no_grad()
def val_step(model: torch.nn.Module,
             val_data: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module,
             writer: SummaryWriter,
             epoch: int,
             device: torch.device) -> float:
    """
    Performs one epoch of validation.
    
    Args:
        model: The neural network model
        val_data: DataLoader for validation data
        loss_fn: Loss function
        writer: TensorBoard writer
        epoch: Current epoch number
        device: Device to run validation on
        
    Returns:
        float: Validation accuracy
    """
    model.eval()
    losses = []
    correct_preds = 0
    total_preds = 0

    # Validation loop
    for text_data, ner_data, target_sentiment_data in val_data:
        # Move data to device
        text_data = text_data.to(device)
        ner_data = ner_data.to(device)
        target_sentiment_data = target_sentiment_data.to(device)
        
        # Forward pass
        predictions = model(text_data, ner_data)
        
        # Compute loss
        loss = loss_fn(predictions, target_sentiment_data)
        
        # Track metrics
        losses.append(loss.item())
        preds = torch.argmax(predictions, dim=1)
        correct_preds += torch.sum(preds == target_sentiment_data).item()
        total_preds += target_sentiment_data.size(0)

    # Compute epoch metrics
    avg_loss = sum(losses) / len(losses) if losses else 0
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    
    # Log metrics
    writer.add_scalar("val/loss", avg_loss, epoch)
    writer.add_scalar("val/accuracy", accuracy, epoch)
    print(f"Epoch {epoch+1} VAL   -> Loss: {avg_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
    
    return accuracy


@torch.no_grad()
def t_step(model: torch.nn.Module,
           test_data: torch.utils.data.DataLoader,
           device: torch.device) -> float:
    """
    Evaluates model on test data.
    
    Args:
        model: The neural network model
        test_data: DataLoader for test data
        device: Device to run evaluation on
        
    Returns:
        float: Test accuracy
    """
    model.eval()
    correct_preds = 0
    total_preds = 0

    # Test loop
    for text_data, ner_data, target_sentiment_data in tqdm(test_data, desc="Testing"):
        # Move data to device
        text_data = text_data.to(device)
        ner_data = ner_data.to(device)
        target_sentiment_data = target_sentiment_data.to(device)
        
        # Forward pass
        predictions = model(text_data, ner_data)
        
        # Track metrics
        preds = torch.argmax(predictions, dim=1)
        correct_preds += torch.sum(preds == target_sentiment_data).item()
        total_preds += target_sentiment_data.size(0)

    # Compute accuracy
    accuracy = correct_preds / total_preds if total_preds > 0 else 0
    
    return accuracy
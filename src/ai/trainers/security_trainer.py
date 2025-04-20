"""
Security Model Trainer

This module provides training functionality for security models with a focus on
adversarial robustness and security-specific evaluation metrics.
"""

import os
import json
import logging
import time
from typing import Dict, List, Tuple, Union, Optional, Any, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, roc_auc_score
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup,
)

from src.ai.models.security_model import SecurityModel, RobustSecurityModel

logger = logging.getLogger(__name__)

class SecurityDataset(Dataset):
    """Dataset for security text classification."""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        """
        Initialize the dataset.
        
        Args:
            texts: List of text samples
            labels: List of label IDs
            tokenizer: Tokenizer to use for encoding
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )
        
        # Remove batch dimension
        encoding = {k: v.squeeze(0) for k, v in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        
        return encoding


class SecurityTrainer:
    """Trainer for security models with adversarial training capabilities."""
    
    def __init__(
        self,
        model: Union[SecurityModel, str],
        output_dir: str,
        security_training_config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: SecurityModel instance or path to pretrained model
            output_dir: Directory to save the model
            security_training_config: Configuration for security-specific training
        """
        # Set output directory
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize model
        if isinstance(model, str):
            self.model = SecurityModel.from_pretrained(model)
        else:
            self.model = model
            
        # Training configuration
        self.config = {
            # Default training parameters
            "learning_rate": 2e-5,
            "batch_size": 32,
            "num_epochs": 3,
            "weight_decay": 0.01,
            "warmup_ratio": 0.1,
            "max_grad_norm": 1.0,
            
            # Security-specific parameters
            "enable_adversarial_training": False,
            "adversarial_alpha": 0.3,
            "security_metrics": ["precision", "recall", "f1", "accuracy"],
            "adversarial_attack_types": ["fgsm"],
            "eps": 0.01,
        }
        
        # Update with provided configuration
        if security_training_config:
            self.config.update(security_training_config)
            
        # Set device
        self.device = next(self.model.model.parameters()).device
        
        logger.info(f"Initialized SecurityTrainer with output_dir={output_dir}")
        
        # Track best model
        self.best_model_path = None
        self.best_metric_value = -float("inf")
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
    def prepare_security_dataset(
        self,
        dataset: Union[Dict[str, Union[List[str], List[int]]], Tuple[List[str], List[int]]],
        val_split: float = 0.1,
    ):
        """
        Prepare dataset for security model training.
        
        Args:
            dataset: Dictionary with 'texts' and 'labels' keys or tuple of (texts, labels)
            val_split: Fraction of data to use for validation
        """
        # Extract texts and labels
        if isinstance(dataset, dict):
            texts = dataset["texts"]
            labels = dataset["labels"]
        else:
            texts, labels = dataset
            
        # Create dataset
        self.tokenizer = self.model.tokenizer
        self.full_dataset = SecurityDataset(
            texts=texts,
            labels=labels,
            tokenizer=self.tokenizer,
            max_length=self.model.max_length,
        )
        
        # Split into train and validation
        val_size = int(len(self.full_dataset) * val_split)
        train_size = len(self.full_dataset) - val_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.full_dataset, [train_size, val_size]
        )
        
        logger.info(f"Prepared dataset with {train_size} training and {val_size} validation samples")
        
    def train(self, train_dataset=None, val_dataset=None):
        """
        Train the security model.
        
        Args:
            train_dataset: Optional training dataset to use instead of self.train_dataset
            val_dataset: Optional validation dataset to use instead of self.val_dataset
        """
        if train_dataset is None:
            train_dataset = self.train_dataset
            
        if val_dataset is None:
            val_dataset = self.val_dataset
            
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        
        # Set up optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        
        # Set model to training mode
        self.model.model.train()
        
        # Training loop
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Starting epoch {epoch+1}/{self.config['num_epochs']}")
            
            # Track metrics
            epoch_loss = 0
            
            # Training step
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model.model(**batch)
                loss = outputs.loss
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), 
                    self.config["max_grad_norm"]
                )
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": loss.item()})
                
            # Calculate average loss
            avg_loss = epoch_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} - Average loss: {avg_loss:.4f}")
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
            
            # Log validation metrics
            log_str = f"Validation metrics - "
            for metric, value in val_metrics.items():
                log_str += f"{metric}: {value:.4f} "
            logger.info(log_str)
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.output_dir, f"checkpoint-epoch-{epoch+1}")
            self.save_checkpoint(checkpoint_path, val_metrics)
            
            # Track best model
            # Use F1 score as the primary metric for security models
            current_metric = val_metrics.get("f1", 0.0)
            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                self.best_model_path = checkpoint_path
                logger.info(f"New best model with F1: {current_metric:.4f}")
                
        # Load best model
        if self.best_model_path:
            logger.info(f"Loading best model from {self.best_model_path}")
            self.model = SecurityModel.from_pretrained(self.best_model_path)
            
        # Save final model
        final_path = os.path.join(self.output_dir, "final-model")
        self.model.save(final_path)
        logger.info(f"Training complete. Final model saved to {final_path}")
        
        return val_metrics
    
    def train_with_adversarial_examples(self, train_dataset=None, val_dataset=None):
        """
        Train the security model with adversarial examples.
        
        Args:
            train_dataset: Optional training dataset to use instead of self.train_dataset
            val_dataset: Optional validation dataset to use instead of self.val_dataset
        """
        if not self.config["enable_adversarial_training"]:
            logger.warning("Adversarial training not enabled. Using regular training.")
            return self.train(train_dataset, val_dataset)
            
        if train_dataset is None:
            train_dataset = self.train_dataset
            
        if val_dataset is None:
            val_dataset = self.val_dataset
            
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config["batch_size"],
            shuffle=False,
        )
        
        # Set up optimizer and scheduler
        optimizer = self._create_optimizer()
        scheduler = self._create_scheduler(optimizer, len(train_loader))
        
        # Set model to training mode
        self.model.model.train()
        
        # Training loop
        for epoch in range(self.config["num_epochs"]):
            logger.info(f"Starting epoch {epoch+1}/{self.config['num_epochs']} with adversarial training")
            
            # Track metrics
            epoch_loss = 0
            adv_loss = 0
            
            # Training step
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            for batch in progress_bar:
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Clear gradients
                optimizer.zero_grad()
                
                # Forward pass on clean data
                outputs = self.model.model(**batch)
                clean_loss = outputs.loss
                
                # Generate adversarial examples
                adv_batch = self._generate_adversarial_batch(batch)
                
                # Forward pass on adversarial examples
                adv_outputs = self.model.model(**adv_batch)
                current_adv_loss = adv_outputs.loss
                
                # Combined loss (weighted)
                alpha = self.config["adversarial_alpha"]
                loss = (1 - alpha) * clean_loss + alpha * current_adv_loss
                
                # Backward pass
                loss.backward()
                
                # Clip gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.model.parameters(), 
                    self.config["max_grad_norm"]
                )
                
                # Update parameters
                optimizer.step()
                scheduler.step()
                
                # Update metrics
                epoch_loss += clean_loss.item()
                adv_loss += current_adv_loss.item()
                progress_bar.set_postfix({
                    "clean_loss": clean_loss.item(),
                    "adv_loss": current_adv_loss.item()
                })
                
            # Calculate average losses
            avg_loss = epoch_loss / len(train_loader)
            avg_adv_loss = adv_loss / len(train_loader)
            logger.info(
                f"Epoch {epoch+1} - "
                f"Average clean loss: {avg_loss:.4f}, "
                f"Average adversarial loss: {avg_adv_loss:.4f}"
            )
            
            # Evaluate on validation set
            val_metrics = self.evaluate(val_loader)
            
            # Log validation metrics
            log_str = f"Validation metrics - "
            for metric, value in val_metrics.items():
                log_str += f"{metric}: {value:.4f} "
            logger.info(log_str)
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.output_dir, f"adv-checkpoint-epoch-{epoch+1}")
            self.save_checkpoint(checkpoint_path, val_metrics)
            
            # Track best model
            current_metric = val_metrics.get("f1", 0.0)
            if current_metric > self.best_metric_value:
                self.best_metric_value = current_metric
                self.best_model_path = checkpoint_path
                logger.info(f"New best model with F1: {current_metric:.4f}")
                
        # Load best model
        if self.best_model_path:
            logger.info(f"Loading best model from {self.best_model_path}")
            self.model = SecurityModel.from_pretrained(self.best_model_path)
            
        # Evaluate robustness on adversarial examples
        robustness_metrics = self.evaluate_security_performance(val_loader)
        
        # Save final model
        final_path = os.path.join(self.output_dir, "final-robust-model")
        
        # If the model isn't already a RobustSecurityModel, convert it
        if not isinstance(self.model, RobustSecurityModel):
            robust_model = RobustSecurityModel(
                model_name_or_path=self.best_model_path or self.output_dir,
                num_labels=self.model.num_labels,
                confidence_threshold=self.model.confidence_threshold,
                max_length=self.model.max_length,
                smoothing_noise=0.01,
                num_ensemble_samples=5,
            )
            robust_model.save(final_path)
        else:
            self.model.save(final_path)
            
        logger.info(f"Adversarial training complete. Robust model saved to {final_path}")
        
        return robustness_metrics
    
    def evaluate(self, val_loader=None):
        """
        Evaluate the model on validation data.
        
        Args:
            val_loader: Optional validation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        if val_loader is None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
            )
            
        # Set model to evaluation mode
        self.model.model.eval()
        
        # Track predictions and labels
        all_preds = []
        all_labels = []
        
        # Evaluation loop
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get labels
            labels = batch.pop("labels").cpu().numpy()
            
            # Forward pass
            with torch.no_grad():
                outputs = self.model.model(**batch)
                logits = outputs.logits
                
            # Get predictions
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Store predictions and labels
            all_preds.extend(preds)
            all_labels.extend(labels)
            
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        # Set model back to training mode
        self.model.model.train()
        
        return metrics
    
    def evaluate_security_performance(self, val_loader=None):
        """
        Evaluate the model's security performance, including robustness.
        
        Args:
            val_loader: Optional validation data loader
            
        Returns:
            Dictionary of security evaluation metrics
        """
        if val_loader is None:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.config["batch_size"],
                shuffle=False,
            )
            
        # Set model to evaluation mode
        self.model.model.eval()
        
        # Track predictions and labels
        clean_preds = []
        adv_preds = []
        all_labels = []
        
        # Evaluation loop
        for batch in val_loader:
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get labels
            labels = batch.pop("labels").cpu().numpy()
            
            # Forward pass on clean data
            with torch.no_grad():
                outputs = self.model.model(**batch)
                logits = outputs.logits
                
            # Get predictions on clean data
            clean_pred = torch.argmax(logits, dim=1).cpu().numpy()
            
            # Generate adversarial examples
            batch["labels"] = torch.tensor(labels).to(self.device)  # Add labels back for adversarial generation
            adv_batch = self._generate_adversarial_batch(batch)
            adv_batch.pop("labels")  # Remove labels for prediction
            
            # Forward pass on adversarial data
            with torch.no_grad():
                adv_outputs = self.model.model(**adv_batch)
                adv_logits = adv_outputs.logits
                
            # Get predictions on adversarial data
            adv_pred = torch.argmax(adv_logits, dim=1).cpu().numpy()
            
            # Store predictions and labels
            clean_preds.extend(clean_pred)
            adv_preds.extend(adv_pred)
            all_labels.extend(labels)
            
        # Calculate metrics
        clean_metrics = self._calculate_metrics(all_labels, clean_preds)
        adv_metrics = self._calculate_metrics(all_labels, adv_preds)
        
        # Calculate robustness
        robustness = {}
        for metric in clean_metrics:
            if clean_metrics[metric] > 0:
                robustness[f"robust_{metric}"] = adv_metrics[metric] / clean_metrics[metric]
            else:
                robustness[f"robust_{metric}"] = 0.0
                
        # Calculate attack success rate (when clean is correct but adversarial is wrong)
        clean_correct = np.array(clean_preds) == np.array(all_labels)
        adv_incorrect = np.array(adv_preds) != np.array(all_labels)
        attack_success = np.logical_and(clean_correct, adv_incorrect)
        attack_success_rate = attack_success.mean()
        
        # Combine metrics
        security_metrics = {
            **clean_metrics,
            **{f"adv_{k}": v for k, v in adv_metrics.items()},
            **robustness,
            "attack_success_rate": attack_success_rate,
        }
        
        # Set model back to training mode
        self.model.model.train()
        
        return security_metrics
    
    def save_checkpoint(self, path: str, metrics: Dict[str, float] = None):
        """
        Save a checkpoint of the model.
        
        Args:
            path: Path to save the checkpoint
            metrics: Optional metrics to save with the checkpoint
        """
        # Save model
        self.model.save(path)
        
        # Save metrics if provided
        if metrics:
            metrics_path = os.path.join(path, "metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(metrics, f, indent=2)
                
        logger.info(f"Checkpoint saved to {path}")
        
    def _create_optimizer(self):
        """Create optimizer for training."""
        # Get parameters
        model_params = self.model.model.parameters()
        
        # Create optimizer
        optimizer = AdamW(
            model_params,
            lr=self.config["learning_rate"],
            weight_decay=self.config["weight_decay"],
        )
        
        return optimizer
        
    def _create_scheduler(self, optimizer, num_training_steps):
        """Create learning rate scheduler."""
        warmup_steps = int(num_training_steps * self.config["warmup_ratio"])
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps * self.config["num_epochs"],
        )
        
        return scheduler
        
    def _calculate_metrics(self, labels, preds):
        """Calculate evaluation metrics."""
        # Calculate precision, recall, F1
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary" if self.model.num_labels == 2 else "macro"
        )
        
        # Calculate accuracy
        accuracy = accuracy_score(labels, preds)
        
        # Combine metrics
        metrics = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
        }
        
        # Filter metrics based on configuration
        return {k: metrics[k] for k in self.config["security_metrics"] if k in metrics}
        
    def _generate_adversarial_batch(self, batch):
        """Generate adversarial examples for the batch."""
        # Clone batch to avoid modifying original
        adv_batch = {k: v.clone().detach() for k, v in batch.items()}
        
        # Get embedding layer
        embedding_layer = self.model.model.get_input_embeddings()
        
        # Choose attack type
        attack_type = np.random.choice(self.config["adversarial_attack_types"])
        
        # Generate adversarial examples based on attack type
        if attack_type == "fgsm":
            # Fast Gradient Sign Method
            adv_batch = self._fgsm_attack(adv_batch, embedding_layer)
        elif attack_type == "pgd":
            # Projected Gradient Descent
            adv_batch = self._pgd_attack(adv_batch, embedding_layer)
            
        return adv_batch
        
    def _fgsm_attack(self, batch, embedding_layer):
        """
        Generate adversarial examples using Fast Gradient Sign Method.
        
        Args:
            batch: Input batch
            embedding_layer: Model's embedding layer
            
        Returns:
            Batch with adversarial examples
        """
        # Get input IDs and labels
        input_ids = batch["input_ids"]
        labels = batch.pop("labels")
        
        # Get embeddings
        embeddings = embedding_layer(input_ids)
        embeddings.requires_grad = True
        
        # Forward pass
        outputs = self.model.model(
            inputs_embeds=embeddings,
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        
        # Backward pass to get gradients
        loss.backward()
        
        # Create adversarial embeddings
        eps = self.config["eps"]
        adv_embeddings = embeddings + eps * torch.sign(embeddings.grad)
        
        # Replace batch with adversarial embeddings
        adv_batch = batch.copy()
        adv_batch["inputs_embeds"] = adv_embeddings.detach()
        adv_batch["labels"] = labels
        
        # Remove input_ids since we're using embeddings directly
        adv_batch.pop("input_ids", None)
        
        return adv_batch
        
    def _pgd_attack(self, batch, embedding_layer, num_steps=3):
        """
        Generate adversarial examples using Projected Gradient Descent.
        
        Args:
            batch: Input batch
            embedding_layer: Model's embedding layer
            num_steps: Number of PGD steps
            
        Returns:
            Batch with adversarial examples
        """
        # Get input IDs and labels
        input_ids = batch["input_ids"]
        labels = batch.pop("labels")
        
        # Get original embeddings
        original_embeddings = embedding_layer(input_ids)
        
        # Initialize adversarial embeddings
        adv_embeddings = original_embeddings.clone().detach()
        
        # PGD attack steps
        eps = self.config["eps"]
        alpha = eps / num_steps  # Step size
        
        for _ in range(num_steps):
            adv_embeddings.requires_grad = True
            
            # Forward pass
            outputs = self.model.model(
                inputs_embeds=adv_embeddings,
                attention_mask=batch["attention_mask"],
                labels=labels,
            )
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Update adversarial embeddings
            with torch.no_grad():
                adv_embeddings = adv_embeddings + alpha * torch.sign(adv_embeddings.grad)
                
                # Project back into epsilon ball
                delta = adv_embeddings - original_embeddings
                delta = torch.clamp(delta, -eps, eps)
                adv_embeddings = original_embeddings + delta
            
        # Replace batch with adversarial embeddings
        adv_batch = batch.copy()
        adv_batch["inputs_embeds"] = adv_embeddings.detach()
        adv_batch["labels"] = labels
        
        # Remove input_ids since we're using embeddings directly
        adv_batch.pop("input_ids", None)
        
        return adv_batch 
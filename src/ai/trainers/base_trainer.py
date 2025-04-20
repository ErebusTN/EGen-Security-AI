"""
Base Trainer for EGen Security AI.

This module provides the foundation for training transformer-based models
in the EGen Security AI system, with support for various training configurations,
logging, and checkpointing.
"""

import os
import time
import json
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import logging
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    Trainer,
    TrainingArguments,
    TrainerCallback,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from src.ai.models.base_model import BaseTransformerModel

class BaseTrainer:
    """Base class for training transformer models in EGen Security AI."""
    
    def __init__(
        self,
        model: BaseTransformerModel,
        output_dir: str,
        training_args: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
    ):
        """
        Initialize the trainer.
        
        Args:
            model: The BaseTransformerModel to train
            output_dir: Directory to save model checkpoints and logs
            training_args: Dictionary of training arguments
            callbacks: List of trainer callbacks
        """
        self.model = model
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing trainer for model: {model.model_name_or_path}")
        
        # Set default training arguments if not provided
        self.training_args = training_args or {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 8,
            "per_device_eval_batch_size": 8,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-5,
            "weight_decay": 0.01,
            "warmup_steps": 100,
            "evaluation_strategy": "steps",
            "logging_steps": 50,
            "save_steps": 500,
            "eval_steps": 500,
            "save_total_limit": 3,
            "fp16": model.precision == "fp16",
            "bf16": model.precision == "bf16",
            "report_to": "none",  # Disable default HF reporting
        }
        
        # Create the HuggingFace trainer config
        self.hf_training_args = TrainingArguments(
            output_dir=output_dir,
            **self.training_args
        )
        
        # Initialize callbacks
        self.callbacks = callbacks or []
        
        # Training metrics
        self.training_metrics = {
            "train_loss": [],
            "eval_loss": [],
            "learning_rate": [],
            "epoch": [],
            "step": [],
            "time_elapsed": [],
        }
        
        # Start time (to be set when training begins)
        self.start_time = None
        
        self.logger.info(f"Trainer initialized with output directory: {output_dir}")
        
    def prepare_dataset(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset] = None,
    ) -> None:
        """
        Prepare datasets for training.
        
        Args:
            train_dataset: Training dataset
            eval_dataset: Evaluation dataset (optional)
        """
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.logger.info(f"Prepared training dataset with {len(train_dataset)} samples")
        if eval_dataset:
            self.logger.info(f"Prepared evaluation dataset with {len(eval_dataset)} samples")
        
    def train(self) -> Dict[str, Any]:
        """
        Train the model using the prepared datasets.
        
        Returns:
            Dictionary with training metrics and results
        """
        if not hasattr(self, 'train_dataset'):
            raise ValueError("Training dataset not prepared. Call prepare_dataset first.")
        
        self.logger.info("Starting model training...")
        self.start_time = time.time()
        
        # Prepare model for training
        self.model.prepare_for_training()
        
        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self.model.model,
            args=self.hf_training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.model.tokenizer,
            callbacks=self.callbacks,
        )
        
        # Perform training
        train_result = trainer.train()
        
        # Save model
        trainer.save_model()
        self.model.tokenizer.save_pretrained(self.output_dir)
        
        # Log training metrics
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        # Update training metrics
        self.training_metrics.update(metrics)
        self.training_metrics["time_elapsed"] = time.time() - self.start_time
        
        # Save training metrics to JSON
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        self.logger.info(f"Model training completed in {self.training_metrics['time_elapsed']:.2f} seconds")
        self.logger.info(f"Training metrics saved to {metrics_path}")
        
        # Prepare model for inference again
        self.model.prepare_for_inference()
        
        return self.training_metrics
    
    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the evaluation dataset.
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not hasattr(self, 'eval_dataset') or self.eval_dataset is None:
            raise ValueError("Evaluation dataset not prepared. Call prepare_dataset with eval_dataset.")
        
        self.logger.info("Starting model evaluation...")
        
        # Create HuggingFace Trainer
        trainer = Trainer(
            model=self.model.model,
            args=self.hf_training_args,
            eval_dataset=self.eval_dataset,
            tokenizer=self.model.tokenizer,
        )
        
        # Perform evaluation
        eval_result = trainer.evaluate()
        
        # Log evaluation metrics
        trainer.log_metrics("eval", eval_result)
        trainer.save_metrics("eval", eval_result)
        
        self.logger.info(f"Model evaluation completed with loss: {eval_result.get('eval_loss', 'N/A')}")
        
        return eval_result
    
    def save_checkpoint(self, checkpoint_dir: Optional[str] = None) -> str:
        """
        Save a training checkpoint.
        
        Args:
            checkpoint_dir: Directory to save the checkpoint (defaults to timestamped directory in output_dir)
            
        Returns:
            Path to the saved checkpoint
        """
        if checkpoint_dir is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_dir = os.path.join(self.output_dir, f"checkpoint_{timestamp}")
        
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        self.model.save(checkpoint_dir)
        
        # Save training metrics
        metrics_path = os.path.join(checkpoint_dir, "training_metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(self.training_metrics, f, indent=2)
        
        self.logger.info(f"Model checkpoint saved to {checkpoint_dir}")
        
        return checkpoint_dir
    
    def get_training_config(self) -> Dict[str, Any]:
        """
        Get the current training configuration.
        
        Returns:
            Dictionary with current training configuration
        """
        return {
            "model_name": self.model.model_name_or_path,
            "output_dir": self.output_dir,
            "training_args": self.training_args,
            "num_callbacks": len(self.callbacks),
            "has_train_dataset": hasattr(self, 'train_dataset'),
            "has_eval_dataset": hasattr(self, 'eval_dataset') and self.eval_dataset is not None,
            "training_progress": {
                "current_epoch": self.training_metrics.get("epoch", [-1])[-1] if self.training_metrics.get("epoch") else -1,
                "current_step": self.training_metrics.get("step", [-1])[-1] if self.training_metrics.get("step") else -1,
                "latest_loss": self.training_metrics.get("train_loss", [-1])[-1] if self.training_metrics.get("train_loss") else -1,
            }
        }
    
    def add_callback(self, callback: TrainerCallback) -> None:
        """
        Add a trainer callback.
        
        Args:
            callback: TrainerCallback to add
        """
        self.callbacks.append(callback)
        self.logger.info(f"Added trainer callback: {callback.__class__.__name__}")
        
    def update_training_args(self, **kwargs) -> None:
        """
        Update training arguments.
        
        Args:
            **kwargs: Key-value pairs to update in training arguments
        """
        self.training_args.update(kwargs)
        self.hf_training_args = TrainingArguments(
            output_dir=self.output_dir,
            **self.training_args
        )
        self.logger.info(f"Updated training arguments: {kwargs}") 
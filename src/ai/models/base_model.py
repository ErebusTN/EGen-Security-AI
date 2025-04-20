"""
Base Transformer Model for EGen Security AI.

This module provides the foundation for all transformer-based models
in the EGen Security AI system, supporting loading pretrained models,
inference, and specialized security functions.
"""

import os
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
import traceback
from pathlib import Path

# Define custom exceptions
class ModelInitializationError(Exception):
    """Exception raised for errors during model initialization."""
    pass

class ModelLoadError(Exception):
    """Exception raised for errors when loading a model."""
    pass

class ModelGenerationError(Exception):
    """Exception raised for errors during text generation."""
    pass

class TokenizationError(Exception):
    """Exception raised for errors during tokenization."""
    pass

# Try to import PyTorch and Transformers, but don't fail if not installed
try:
    import torch
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    # Create dummy objects for type hints when libraries aren't available
    class DummyModule:
        pass
    PreTrainedModel = DummyModule
    PreTrainedTokenizer = DummyModule

from abc import ABC, abstractmethod

class BaseModel(ABC):
    """
    Abstract base class for all AI models.
    
    This class defines the interface that all model implementations must follow.
    """
    
    @abstractmethod
    def predict(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Make a prediction based on the input data.
        
        Args:
            input_data: The input data to make predictions on.
                        Can be a string or structured input as a dictionary.
        
        Returns:
            Dictionary containing prediction results.
        """
        pass
    
    def save(self, path: str) -> Dict[str, Any]:
        """
        Save the model to a specified path.
        
        Args:
            path: The path where the model should be saved.
            
        Returns:
            Dictionary with status of the save operation.
        """
        try:
            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            # Actual implementation should be in subclasses
            raise NotImplementedError("Save method not implemented")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to save model: {str(e)}"
            }
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load the model from a specified path.
        
        Args:
            path: The path from which the model should be loaded.
            
        Returns:
            Dictionary with status of the load operation.
        """
        try:
            if not os.path.exists(path):
                return {
                    "status": "error",
                    "message": f"Model path does not exist: {path}"
                }
            # Actual implementation should be in subclasses
            raise NotImplementedError("Load method not implemented")
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to load model: {str(e)}"
            }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model.
        
        Returns:
            Dictionary containing model metadata.
        """
        return {
            "model_type": self.__class__.__name__,
            "description": self.__doc__
        }
    
    def get_security_config(self) -> Dict[str, Any]:
        """
        Get the security configuration of the model.
        
        Returns:
            Dictionary containing security-related configuration settings.
        """
        return {
            "model_type": self.__class__.__name__
        }
        
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check if all dependencies required by this model are installed.
        
        Returns:
            Dictionary with dependency status information.
        """
        return {"dependencies_met": True}

class BaseTransformerModel:
    """Base class for transformer models in EGen Security AI."""
    
    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[str] = None,
        precision: str = "fp16",
        max_context_length: int = 4096,
        cache_dir: Optional[str] = None,
        load_in_8bit: bool = False,
        trust_remote_code: bool = False,
    ):
        """
        Initialize the transformer model.
        
        Args:
            model_name_or_path: HuggingFace model ID or path to local directory
            device: Computing device to use ('cuda', 'cpu', or specific GPU index)
            precision: Model precision ('fp16', 'bf16', 'fp32')
            max_context_length: Maximum token context length
            cache_dir: Directory to cache models
            load_in_8bit: Whether to load model in 8-bit precision
            trust_remote_code: Whether to trust remote code in model repositories
            
        Raises:
            ModelInitializationError: If the model cannot be initialized
        """
        if not TRANSFORMERS_AVAILABLE:
            raise ModelInitializationError(
                "Required dependencies not available. Please install: torch, transformers"
            )
        
        self.model_name_or_path = model_name_or_path
        self.max_context_length = max_context_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.precision = precision
        
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing model: {model_name_or_path}")
        
        # Initialize model and tokenizer
        try:
            self.tokenizer = self._load_tokenizer(
                model_name_or_path, cache_dir, trust_remote_code
            )
        except Exception as e:
            error_msg = f"Failed to load tokenizer: {str(e)}"
            self.logger.error(error_msg)
            raise ModelInitializationError(error_msg) from e
            
        try:
            self.model = self._load_model(
                model_name_or_path, cache_dir, trust_remote_code, load_in_8bit
            )
        except Exception as e:
            error_msg = f"Failed to load model: {str(e)}"
            self.logger.error(error_msg)
            raise ModelInitializationError(error_msg) from e
        
        # Set model to evaluation mode by default
        self.model.eval()
        
        # Log model information
        try:
            model_params = sum(p.numel() for p in self.model.parameters())
            self.logger.info(f"Model loaded with {model_params/1e9:.2f}B parameters")
            self.logger.info(f"Running on device: {self.device}, precision: {self.precision}")
        except Exception as e:
            self.logger.warning(f"Could not count model parameters: {str(e)}")
        
    def _load_tokenizer(
        self, 
        model_name_or_path: str, 
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
    ) -> PreTrainedTokenizer:
        """
        Load the tokenizer from the specified model path.
        
        Args:
            model_name_or_path: HuggingFace model ID or path
            cache_dir: Directory to cache models
            trust_remote_code: Whether to trust remote code
            
        Returns:
            The loaded tokenizer
            
        Raises:
            Exception: If tokenizer loading fails
        """
        try:
            return AutoTokenizer.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                padding_side="left",
                use_fast=True,
            )
        except Exception as e:
            self.logger.error(f"Error loading tokenizer: {str(e)}")
            # Try fallback options
            try:
                self.logger.info("Trying to load tokenizer with slower implementation")
                return AutoTokenizer.from_pretrained(
                    model_name_or_path,
                    cache_dir=cache_dir,
                    trust_remote_code=trust_remote_code,
                    use_fast=False,
                )
            except Exception as fallback_error:
                self.logger.error(f"Fallback tokenizer also failed: {str(fallback_error)}")
                raise
        
    def _load_model(
        self, 
        model_name_or_path: str, 
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        load_in_8bit: bool = False,
    ) -> PreTrainedModel:
        """
        Load the model from the specified path with appropriate settings.
        
        Args:
            model_name_or_path: HuggingFace model ID or path
            cache_dir: Directory to cache models
            trust_remote_code: Whether to trust remote code
            load_in_8bit: Whether to load in 8-bit precision
            
        Returns:
            The loaded model
            
        Raises:
            Exception: If model loading fails
        """
        try:
            # Set up quantization options based on precision
            quantization_config = None
            torch_dtype = torch.float32  # default
            
            if self.precision == "fp16":
                torch_dtype = torch.float16
            elif self.precision == "bf16":
                if not torch.cuda.is_bf16_supported():
                    self.logger.warning("BF16 not supported on this device, falling back to FP16")
                    torch_dtype = torch.float16
                else:
                    torch_dtype = torch.bfloat16
            
            # Check if running on CPU and need to adjust settings
            if self.device == "cpu" and (self.precision != "fp32" or load_in_8bit):
                self.logger.warning("Running on CPU - forcing FP32 precision and disabling 8-bit loading")
                torch_dtype = torch.float32
                load_in_8bit = False
            
            # Load the model with specified configuration
            model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                cache_dir=cache_dir,
                torch_dtype=torch_dtype,
                load_in_8bit=load_in_8bit,
                trust_remote_code=trust_remote_code,
                device_map=self.device if load_in_8bit else None,
            )
            
            # Move model to device if not using device_map
            if not load_in_8bit and self.device:
                model = model.to(self.device)
                
            return model
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise
        
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[str]:
        """
        Generate text response based on the prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling probability threshold
            top_k: Top-k sampling parameter
            num_return_sequences: Number of sequences to return
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated text responses
            
        Raises:
            ModelGenerationError: If text generation fails
        """
        if not prompt:
            raise ValueError("Prompt cannot be empty")
            
        try:
            # Encode the input prompt
            inputs = self.tokenize(prompt, max_length=self.max_context_length - max_new_tokens)
            
            # Generate text
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    num_return_sequences=num_return_sequences,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **kwargs,
                )
            
            # Decode the generated text
            generated_texts = []
            for output in outputs:
                generated_text = self.tokenizer.decode(output, skip_special_tokens=True)
                # Remove the input prompt from the output
                if generated_text.startswith(prompt):
                    generated_text = generated_text[len(prompt):]
                generated_texts.append(generated_text)
                
            return generated_texts
        except Exception as e:
            error_msg = f"Error generating text: {str(e)}"
            self.logger.error(error_msg)
            self.logger.debug(traceback.format_exc())
            raise ModelGenerationError(error_msg) from e
        
    def tokenize(self, text: str, max_length: Optional[int] = None) -> Dict:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize
            max_length: Maximum length for truncation
            
        Returns:
            Dictionary containing tokenized inputs
            
        Raises:
            TokenizationError: If tokenization fails
        """
        try:
            tokenized = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True if max_length else False,
                max_length=max_length,
            )
            
            # Move tensors to the correct device
            return {k: v.to(self.device) for k, v in tokenized.items()}
        except Exception as e:
            error_msg = f"Error tokenizing text: {str(e)}"
            self.logger.error(error_msg)
            raise TokenizationError(error_msg) from e
        
    def save(self, output_dir: str, save_tokenizer: bool = True) -> None:
        """
        Save the model and optionally the tokenizer to the specified directory.
        
        Args:
            output_dir: Directory to save the model to
            save_tokenizer: Whether to save the tokenizer as well
            
        Raises:
            Exception: If saving fails
        """
        try:
            # Create the output directory if it doesn't exist
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            self.logger.info(f"Saving model to {output_dir}")
            self.model.save_pretrained(output_dir)
            
            if save_tokenizer:
                self.logger.info(f"Saving tokenizer to {output_dir}")
                self.tokenizer.save_pretrained(output_dir)
                
            self.logger.info("Model and tokenizer saved successfully")
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            self.logger.error(error_msg)
            raise
    
    @classmethod
    def from_pretrained(cls, model_path: str, **kwargs) -> 'BaseTransformerModel':
        """
        Load a model from a pretrained checkpoint.
        
        Args:
            model_path: Path to the model checkpoint
            **kwargs: Additional arguments to pass to the constructor
            
        Returns:
            Initialized model
            
        Raises:
            ModelLoadError: If loading fails
        """
        try:
            if not os.path.exists(model_path):
                raise ModelLoadError(f"Model path does not exist: {model_path}")
                
            return cls(model_name_or_path=model_path, **kwargs)
        except Exception as e:
            if not isinstance(e, ModelLoadError):
                error_msg = f"Failed to load model from {model_path}: {str(e)}"
                raise ModelLoadError(error_msg) from e
            raise
            
    def prepare_for_training(self) -> None:
        """Prepare the model for training."""
        if hasattr(self, 'model'):
            self.model.train()
            
    def prepare_for_inference(self) -> None:
        """Prepare the model for inference."""
        if hasattr(self, 'model'):
            self.model.eval()
            
    def check_dependencies(self) -> Dict[str, Any]:
        """
        Check if all dependencies required by this model are installed.
        
        Returns:
            Dictionary with dependency status information.
        """
        status = {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_available": False,
            "cuda_available": False,
            "missing_dependencies": []
        }
        
        try:
            import torch
            status["torch_available"] = True
            status["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            status["missing_dependencies"].append("torch")
            
        if not TRANSFORMERS_AVAILABLE:
            status["missing_dependencies"].append("transformers")
            
        return status 
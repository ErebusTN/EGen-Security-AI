"""
Security Model Implementation

This module provides the base implementation for security models used in threat detection.
"""

import os
import json
import logging
from typing import Dict, List, Tuple, Union, Optional, Any
import traceback

try:
    import torch
    from torch import nn
    import torch.nn.functional as F
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        PreTrainedTokenizer,
        PreTrainedModel,
        AutoModelForCausalLM,
        pipeline
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    
import random
import numpy as np
from tqdm import tqdm

from src.ai.models.base_model import BaseModel
try:
    from src.config import settings
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
    # Create fallback settings if config module is not available
    class Settings:
        pass
    settings = Settings()

logger = logging.getLogger(__name__)

DEFAULT_CYBERSECURITY_MODEL = "segolilylabs/Lily-Cybersecurity-7B-v0.2"
DEFAULT_DEVICE = "cuda" if TRANSFORMERS_AVAILABLE and torch.cuda.is_available() else "cpu"

# Define custom exceptions
class SecurityModelError(Exception):
    """Base exception for security model errors."""
    pass

class SecurityModel(BaseModel):
    """
    Security model for threat detection using transformer-based models.
    
    This model leverages transformer architectures to classify potential security
    threats with confidence scores and explanation generation.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "segolilylabs/Lily-Cybersecurity-7B-v0.2",
        num_labels: int = 2,  # binary classification (threat or not)
        confidence_threshold: float = 0.7,
        device: str = DEFAULT_DEVICE,
        **kwargs
    ):
        """
        Initialize the security model.
        
        Args:
            model_name_or_path: The name or path of the pretrained model.
            num_labels: Number of classification labels.
            confidence_threshold: Threshold for confident predictions.
            device: Device to run the model on (cuda or cpu).
        """
        super().__init__()
        self.model_name = model_name_or_path
        self.num_labels = num_labels
        self.confidence_threshold = confidence_threshold
        self.device = device
        
        logger.info(f"Loading model {model_name_or_path} to {device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        
        # For LLM-based security models, we use causal language modeling
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            **kwargs
        )
        
        # Create text generation pipeline for threat detection
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device if device != "cpu" else -1
        )
        
        # Load threat categories
        self._load_threat_categories()
        
        logger.info(f"Successfully initialized SecurityModel with {model_name_or_path}")

    def _load_threat_categories(self):
        """Load threat categories from configuration or use defaults."""
        # Default categories - can be extended or loaded from config
        self.threat_categories = {
            0: "benign",
            1: "malware",
            2: "phishing",
            3: "dos_ddos",
            4: "privilege_escalation",
            5: "data_exfiltration",
            6: "social_engineering",
            7: "network_intrusion",
            8: "web_attack",
            9: "insider_threat",
        }
        
        # Map from category name to ID for reverse lookup
        self.category_to_id = {v: k for k, v in self.threat_categories.items()}
    
    def _craft_prompt(self, text: str) -> str:
        """
        Craft a prompt for the Lily Cybersecurity model.
        
        Args:
            text: The input text to analyze
            
        Returns:
            Formatted prompt string
        """
        # Using the recommended prompt format for Lily Cybersecurity models
        prompt = f"""<|im_start|>system
You are Lily, a cybersecurity assistant trained to analyze and detect security threats.
<|im_end|>
<|im_start|>user
Please analyze the following content for security threats and vulnerabilities:

{text}

Is this a security threat? If so, what type of threat is it and why?
<|im_end|>
<|im_start|>assistant
"""
        return prompt
    
    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse the model's response to extract threat classification and explanation.
        
        Args:
            response: The raw model response
            
        Returns:
            Dictionary with threat detection results including classification, 
            confidence and explanation
        """
        # Extract the assistant response part only
        result = {
            "is_threat": False,
            "threat_type": "benign",
            "confidence": 0.0,
            "explanation": ""
        }
        
        # Look for indicators of threats in the response
        response_lower = response.lower()
        
        # Check if the response identifies this as a threat
        if "yes, this is a security threat" in response_lower or "this is a security threat" in response_lower:
            result["is_threat"] = True
            
            # Try to identify threat type from response
            for category_name in self.category_to_id.keys():
                if category_name != "benign" and category_name in response_lower:
                    result["threat_type"] = category_name
                    break
            
            # Extract explanation - the part after identifying it as a threat
            try:
                explanation_parts = response.split("security threat")
                if len(explanation_parts) > 1:
                    result["explanation"] = explanation_parts[1].strip()
                else:
                    result["explanation"] = response
                
                # Set a high confidence since the model explicitly identified it as a threat
                result["confidence"] = 0.85
            except:
                result["explanation"] = response
                result["confidence"] = 0.75
        else:
            # If not identified as a threat
            result["explanation"] = response
            
            # Check confidence level based on certainty language
            if "definitely not" in response_lower or "certainly not" in response_lower:
                result["confidence"] = 0.9
            elif "likely not" in response_lower or "probably not" in response_lower:
                result["confidence"] = 0.7
            else:
                result["confidence"] = 0.6
        
        return result

    def predict(self, text: str) -> Dict[str, Any]:
        """
        Perform threat classification on the input text.
        
        Args:
            text: Text to classify
            
        Returns:
            Dictionary with classification results
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                "error": "Required dependencies not available: torch, transformers",
                "status": "failed",
                "is_threat": False
            }
            
        try:
            # Validate input
            if not text or not isinstance(text, str):
                return {
                    "error": "Input must be a non-empty string",
                    "status": "failed",
                    "is_threat": False
                }
                
            # Craft prompt for the model
            prompt = self._craft_prompt(text)
            
            # Generate response
            try:
                response = self.pipeline(
                    prompt,
                    max_new_tokens=512,
                    temperature=0.1,  # Low temperature for more deterministic responses
                    num_return_sequences=1,
                )[0]['generated_text']
                
                # Extract just the assistant's response
                if "<|im_start|>assistant" in response:
                    response = response.split("<|im_start|>assistant")[-1]
                    # Remove any trailing system or user messages
                    if "<|im_start|>" in response:
                        response = response.split("<|im_start|>")[0]
            except Exception as e:
                logger.error(f"Error during model inference: {str(e)}")
                logger.debug(traceback.format_exc())
                return {
                    "error": f"Model inference failed: {str(e)}",
                    "status": "failed",
                    "is_threat": False  # Default to non-threat in case of errors
                }
            
            # Parse the model's response
            prediction = self._parse_response(response)
            
            # Add metadata to the prediction
            prediction['model'] = self.model_name
            prediction['input_text_length'] = len(text)
            
            return prediction
        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            logger.debug(traceback.format_exc())
            return {
                "error": f"Prediction failed: {str(e)}",
                "status": "failed",
                "is_threat": False,  # Default to non-threat in case of errors
                "confidence": 0.0
            }

    def detect_threats(self, texts: List[str]) -> List[Dict[str, Any]]:
        """
        Run threat detection on a list of texts.
        
        Args:
            texts: List of text strings to analyze
            
        Returns:
            List of prediction dictionaries
        """
        if not TRANSFORMERS_AVAILABLE:
            return [{"error": "Required dependencies not available", "status": "failed"}] * len(texts)
            
        results = []
        
        try:
            for text in tqdm(texts, desc="Analyzing threats"):
                try:
                    prediction = self.predict(text)
                    results.append(prediction)
                except Exception as e:
                    logger.error(f"Error analyzing text: {str(e)}")
                    # Add error result but continue processing other texts
                    results.append({
                        "error": f"Analysis failed: {str(e)}",
                        "status": "failed",
                        "is_threat": False
                    })
            return results
        except Exception as e:
            logger.error(f"Error in batch threat detection: {str(e)}")
            # Return error results for all texts
            return [{"error": f"Batch processing failed: {str(e)}", "status": "failed"}] * len(texts)
    
    def generate_explanation(self, text: str, prediction: Dict[str, Any]) -> str:
        """
        Generate a human-readable explanation of the threat classification.
        
        Args:
            text: The original input text
            prediction: The prediction dictionary from predict()
            
        Returns:
            String explanation of the threat classification
        """
        if prediction.get("error"):
            return f"Error during analysis: {prediction['error']}"
            
        if prediction["is_threat"]:
            confidence_str = "high" if prediction["confidence"] > 0.8 else \
                            "medium" if prediction["confidence"] > 0.6 else "low"
                            
            explanation = (
                f"SECURITY THREAT DETECTED: {prediction['threat_type'].upper()}\n"
                f"Confidence: {confidence_str} ({prediction['confidence']:.2f})\n\n"
                f"Analysis: {prediction['explanation']}"
            )
        else:
            explanation = (
                f"No security threat detected (confidence: {prediction['confidence']:.2f})\n\n"
                f"Analysis: {prediction['explanation']}"
            )
            
        return explanation

    def save(self, path: str) -> Dict[str, Any]:
        """
        Save the model to the specified path.
        
        Args:
            path: Path to save the model
            
        Returns:
            Dictionary with status of the save operation
        """
        try:
            os.makedirs(path, exist_ok=True)
            
            # Save model and tokenizer
            self.model.save_pretrained(os.path.join(path, "model"))
            self.tokenizer.save_pretrained(os.path.join(path, "tokenizer"))
            
            # Save configuration
            config = {
                "model_name": self.model_name,
                "num_labels": self.num_labels,
                "confidence_threshold": self.confidence_threshold,
                "threat_categories": self.threat_categories
            }
            
            with open(os.path.join(path, "config.json"), "w") as f:
                json.dump(config, f, indent=2)
                
            logger.info(f"Model saved to {path}")
            return {"status": "success", "message": f"Model saved to {path}"}
        except Exception as e:
            error_msg = f"Error saving model: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": error_msg}
    
    def load(self, path: str) -> Dict[str, Any]:
        """
        Load the model from the specified path.
        
        Args:
            path: Path to load the model from
            
        Returns:
            Dictionary with status of the load operation
        """
        if not TRANSFORMERS_AVAILABLE:
            return {"status": "error", "message": "Required dependencies not available"}
            
        try:
            # Check if path exists
            if not os.path.exists(path):
                return {"status": "error", "message": f"Model path does not exist: {path}"}
                
            # Check if the path has the expected structure
            model_path = os.path.join(path, "model")
            tokenizer_path = os.path.join(path, "tokenizer")
            config_path = os.path.join(path, "config.json")
            
            if not os.path.exists(model_path):
                return {"status": "error", "message": f"Model directory not found at {model_path}"}
            if not os.path.exists(tokenizer_path):
                return {"status": "error", "message": f"Tokenizer directory not found at {tokenizer_path}"}
                
            # Load configuration if available
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    config = json.load(f)
                    
                self.model_name = config.get("model_name", self.model_name)
                self.num_labels = config.get("num_labels", self.num_labels)
                self.confidence_threshold = config.get("confidence_threshold", self.confidence_threshold)
                
                if "threat_categories" in config:
                    self.threat_categories = config["threat_categories"]
                    self.category_to_id = {v: k for k, v in self.threat_categories.items()}
            
            # Load model and tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
            )
            
            # Recreate the pipeline
            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=self.device if self.device != "cpu" else -1
            )
            
            logger.info(f"Model loaded from {path}")
            return {"status": "success", "message": f"Model loaded from {path}"}
        except Exception as e:
            error_msg = f"Error loading model: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {"status": "error", "message": error_msg}
            
    def check_dependencies(self) -> Dict[str, Any]:
        """Check if all required dependencies are available."""
        status = {
            "transformers_available": TRANSFORMERS_AVAILABLE,
            "torch_available": False,
            "cuda_available": False,
            "missing_dependencies": []
        }
        
        if not TRANSFORMERS_AVAILABLE:
            status["missing_dependencies"].append("transformers")
        
        try:
            import torch
            status["torch_available"] = True
            status["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            status["missing_dependencies"].append("torch")
            
        return status

class RobustSecurityModel(SecurityModel):
    """
    Enhanced security model with robustness features against adversarial attacks.
    
    This model extends the base SecurityModel with additional robustness techniques
    like ensemble predictions and adversarial detection.
    """
    
    def __init__(
        self,
        model_name_or_path: str = "segolilylabs/Lily-Cybersecurity-7B-v0.2",
        num_labels: int = 2,
        confidence_threshold: float = 0.7,
        device: str = DEFAULT_DEVICE,
        ensemble_size: int = 3,
        randomness_factor: float = 0.1,
        **kwargs
    ):
        """
        Initialize the robust security model.
        
        Args:
            model_name_or_path: The name or path of the pretrained model.
            num_labels: Number of classification labels.
            confidence_threshold: Threshold for confident predictions.
            device: Device to run the model on (cuda or cpu).
            ensemble_size: Number of predictions to make in the ensemble.
            randomness_factor: Temperature variation for ensemble diversity.
        """
        try:
            super().__init__(
                model_name_or_path=model_name_or_path,
                num_labels=num_labels,
                confidence_threshold=confidence_threshold,
                device=device,
                **kwargs
            )
            
            self.ensemble_size = max(1, ensemble_size)  # Ensure at least 1
            self.randomness_factor = randomness_factor
            logger.info(f"Initialized RobustSecurityModel with ensemble_size={ensemble_size}")
        except Exception as e:
            logger.error(f"Failed to initialize RobustSecurityModel: {str(e)}")
            logger.debug(traceback.format_exc())
            # Re-raise as we can't continue without initialization
            raise SecurityModelError(f"Failed to initialize robust security model: {str(e)}") from e

    def predict_with_ensemble(self, text: str) -> Dict[str, Any]:
        """
        Make ensemble predictions for improved robustness.
        
        Args:
            text: Text to classify
            
        Returns:
            Aggregated prediction with ensemble statistics
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                "error": "Required dependencies not available: torch, transformers",
                "status": "failed",
                "is_threat": False
            }
            
        # Validate input
        if not text or not isinstance(text, str):
            return {
                "error": "Input must be a non-empty string",
                "status": "failed",
                "is_threat": False
            }
            
        try:
            ensemble_results = []
            threat_counts = {}
            explanation_parts = []
            
            # Create the prompt once
            prompt = self._craft_prompt(text)
            
            # Make multiple predictions with varied parameters
            successful_predictions = 0
            for i in range(self.ensemble_size):
                try:
                    # Vary temperature slightly for each prediction
                    temperature = 0.1 + (random.random() * self.randomness_factor)
                    
                    # Generate response
                    response = self.pipeline(
                        prompt,
                        max_new_tokens=512,
                        temperature=temperature,
                        num_return_sequences=1,
                    )[0]['generated_text']
                    
                    # Extract the assistant's response
                    if "<|im_start|>assistant" in response:
                        response = response.split("<|im_start|>assistant")[-1]
                        if "<|im_start|>" in response:
                            response = response.split("<|im_start|>")[0]
                    
                    # Parse the response
                    result = self._parse_response(response)
                    ensemble_results.append(result)
                    successful_predictions += 1
                    
                    # Count threat types
                    threat_type = result.get('threat_type', 'benign')
                    if threat_type not in threat_counts:
                        threat_counts[threat_type] = 0
                    threat_counts[threat_type] += 1
                    
                    # Collect explanations
                    if result.get('explanation'):
                        explanation_parts.append(result['explanation'])
                except Exception as e:
                    logger.error(f"Error in ensemble prediction {i}: {str(e)}")
                    # Continue with the next prediction even if one fails
            
            # If all predictions failed, return an error
            if successful_predictions == 0:
                return {
                    "error": "All ensemble predictions failed",
                    "status": "failed",
                    "is_threat": False,
                    "confidence": 0.0
                }
            
            # Determine the majority threat type
            majority_type = max(threat_counts.items(), key=lambda x: x[1])[0] if threat_counts else 'benign'
            is_threat = majority_type != 'benign'
            
            # Calculate confidence based on agreement
            majority_count = threat_counts.get(majority_type, 0)
            confidence = majority_count / successful_predictions if successful_predictions > 0 else 0.0
            
            # Generate an aggregate explanation
            explanation = "No explanation could be generated."
            if explanation_parts:
                # Use the explanation from the majority prediction type
                majority_explanations = []
                for result in ensemble_results:
                    if result.get('threat_type') == majority_type and result.get('explanation'):
                        majority_explanations.append(result['explanation'])
                
                if majority_explanations:
                    # Use the longest explanation as it's likely most detailed
                    explanation = max(majority_explanations, key=len)
                else:
                    # Fallback to the longest explanation overall
                    explanation = max(explanation_parts, key=len)
            
            # Create the final prediction
            prediction = {
                'is_threat': is_threat,
                'threat_type': majority_type,
                'confidence': confidence,
                'explanation': explanation,
                'ensemble_size': self.ensemble_size,
                'successful_predictions': successful_predictions,
                'ensemble_agreement': majority_count / successful_predictions if successful_predictions > 0 else 0.0,
                'ensemble_results': threat_counts,
                'model': self.model_name
            }
            
            return prediction
        except Exception as e:
            error_msg = f"Error during ensemble prediction: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {
                "error": error_msg,
                "status": "failed",
                "is_threat": False,
                "confidence": 0.0
            }

    def predict(self, input_data: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Override predict to use ensemble prediction.
        
        Args:
            input_data: Text to classify or a dictionary with 'text' field
            
        Returns:
            Prediction results
        """
        try:
            # Extract text from input
            if isinstance(input_data, dict):
                if 'text' not in input_data:
                    return {
                        "error": "Input dictionary must contain a 'text' field",
                        "status": "failed",
                        "is_threat": False
                    }
                text = input_data['text']
            else:
                text = input_data
                
            return self.predict_with_ensemble(text)
        except Exception as e:
            error_msg = f"Error during prediction: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {
                "error": error_msg,
                "status": "failed",
                "is_threat": False,
                "confidence": 0.0
            }

    def adversarial_detection(self, text: str) -> Dict[str, Any]:
        """
        Perform adversarial detection on input text.
        
        This method attempts to identify adversarial examples designed to fool the model.
        
        Args:
            text: Text to analyze for adversarial patterns
            
        Returns:
            Dictionary with adversarial detection results
        """
        if not TRANSFORMERS_AVAILABLE:
            return {
                "error": "Required dependencies not available: torch, transformers",
                "status": "failed",
                "is_threat": False
            }
            
        # Validate input
        if not text or not isinstance(text, str):
            return {
                "error": "Input must be a non-empty string",
                "status": "failed",
                "is_threat": False
            }
            
        try:
            # First, get the regular prediction
            base_prediction = self.predict_with_ensemble(text)
            
            # Skip if the initial prediction failed
            if "error" in base_prediction and "status" in base_prediction and base_prediction["status"] == "failed":
                return base_prediction
            
            # Check for common adversarial patterns
            adversarial_patterns = [
                # Homoglyphs (characters that look like others)
                {'pattern': r'[А-Яа-я]', 'description': 'Cyrillic characters that may mimic Latin characters'},
                {'pattern': r'[\u2500-\u257F]', 'description': 'Box-drawing characters that may hide content'},
                # Zero-width characters
                {'pattern': r'[\u200B-\u200F\u2060\uFEFF]', 'description': 'Zero-width characters'},
                # Unusual spacing or formatting
                {'pattern': r'[\u2000-\u200A]', 'description': 'Unusual spacing characters'},
                # Control characters
                {'pattern': r'[\u0000-\u001F\u007F-\u009F]', 'description': 'Control characters'}
            ]
            
            suspicious_patterns = []
            try:
                import re
                for pattern in adversarial_patterns:
                    if re.search(pattern['pattern'], text):
                        suspicious_patterns.append(pattern['description'])
            except Exception as e:
                logger.error(f"Error checking adversarial patterns: {str(e)}")
                suspicious_patterns.append(f"Pattern check failed: {str(e)}")
            
            # Create variations of the text to check for robustness
            variations = []
            try:
                import re
                variations = [
                    text.lower(),  # Lowercase version
                    re.sub(r'\s+', ' ', text),  # Normalize spacing
                    re.sub(r'[^\w\s]', '', text)  # Remove punctuation
                ]
            except Exception as e:
                logger.error(f"Error creating text variations: {str(e)}")
            
            # Check if predictions are consistent across variations
            variation_results = []
            for variant in variations:
                try:
                    # Use simplified prediction to avoid endless recursion
                    variant_prediction = super().predict(variant)
                    variation_results.append({
                        'is_threat': variant_prediction.get('is_threat', False),
                        'threat_type': variant_prediction.get('threat_type', 'benign'),
                        'confidence': variant_prediction.get('confidence', 0.0)
                    })
                except Exception as e:
                    logger.error(f"Error predicting text variation: {str(e)}")
            
            # Check for inconsistency in predictions
            inconsistent = False
            confidence_variation = 0.0
            if variation_results:
                base_is_threat = base_prediction.get('is_threat', False)
                inconsistent = any(r.get('is_threat', False) != base_is_threat for r in variation_results)
                confidences = [r.get('confidence', 0) for r in variation_results if 'confidence' in r]
                
                if confidences and 'confidence' in base_prediction:
                    base_confidence = base_prediction['confidence']
                    confidence_variation = max(abs(conf - base_confidence) for conf in confidences)
            
            # Determine if this might be an adversarial example
            is_adversarial = len(suspicious_patterns) > 0 or (inconsistent and confidence_variation > 0.3)
            
            # Add adversarial analysis to the original prediction
            base_prediction.update({
                'adversarial_detection': {
                    'is_adversarial': is_adversarial,
                    'suspicious_patterns': suspicious_patterns,
                    'prediction_inconsistency': inconsistent,
                    'confidence_variation': confidence_variation,
                    'explanation': self._generate_adversarial_explanation(is_adversarial, suspicious_patterns, inconsistent)
                }
            })
            
            return base_prediction
        except Exception as e:
            error_msg = f"Error during adversarial detection: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            return {
                "error": error_msg,
                "status": "failed",
                "is_threat": False,
                "confidence": 0.0
            }
    
    def _generate_adversarial_explanation(self, is_adversarial, patterns, inconsistent):
        """Generate explanation for adversarial detection results."""
        try:
            if not is_adversarial:
                return "No adversarial patterns detected in the input."
                
            explanation = "Possible adversarial content detected:"
            
            if patterns:
                explanation += "\n- Suspicious patterns: " + ", ".join(patterns)
                
            if inconsistent:
                explanation += "\n- Inconsistent predictions across text variations"
                
            explanation += "\nThis may indicate an attempt to evade detection or manipulate the model."
            return explanation
        except Exception as e:
            logger.error(f"Error generating adversarial explanation: {str(e)}")
            return "Error generating adversarial explanation"

    def generate_explanation(self, text: str, prediction: Dict[str, Any]) -> str:
        """
        Generate a more detailed explanation for the prediction.
        
        This overrides the base class method to provide more robust explanations.
        
        Args:
            text: The original input text
            prediction: The prediction dictionary
            
        Returns:
            Enhanced explanation string
        """
        try:
            # Check if an adversarial explanation is available
            if (
                'adversarial_detection' in prediction and 
                prediction['adversarial_detection'].get('is_adversarial', False) and
                'explanation' in prediction['adversarial_detection']
            ):
                adv_explanation = prediction['adversarial_detection']['explanation']
                base_explanation = ""
                
                # Get base explanation safely
                try:
                    base_explanation = super().generate_explanation(text, prediction)
                except Exception as e:
                    logger.error(f"Error generating base explanation: {str(e)}")
                    base_explanation = prediction.get('explanation', "No explanation available")
                
                return f"{base_explanation}\n\nAdvisory: {adv_explanation}"
            else:
                # If not adversarial, use the base class explanation with ensemble information
                base_explanation = super().generate_explanation(text, prediction)
                
                if 'ensemble_agreement' in prediction:
                    agreement = prediction['ensemble_agreement'] * 100
                    ensemble_size = prediction.get('ensemble_size', 0)
                    successful = prediction.get('successful_predictions', ensemble_size)
                    ensemble_note = f"\n\nThis assessment is based on an ensemble of {successful} successful predictions out of {ensemble_size} attempts, with {agreement:.1f}% agreement."
                    return f"{base_explanation}{ensemble_note}"
                else:
                    return base_explanation
        except Exception as e:
            logger.error(f"Error generating robust explanation: {str(e)}")
            # Return a basic explanation in case of error
            return prediction.get('explanation', f"Could not generate explanation due to an error: {str(e)}") 
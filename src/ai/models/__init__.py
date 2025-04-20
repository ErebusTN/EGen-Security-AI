"""
AI models package for EGen Security AI.

This package contains AI model implementations for security applications.
"""

from src.ai.models.base_model import BaseModel
import logging
import sys

logger = logging.getLogger(__name__)

# Set up module-level variables for model availability
ADVANCED_MODELS_AVAILABLE = False
MISSING_DEPENDENCIES = []

# Try to import advanced models, but don't fail if dependencies aren't available
try:
    from src.ai.models.security_model import SecurityModel, RobustSecurityModel
    ADVANCED_MODELS_AVAILABLE = True
    __all__ = ['BaseModel', 'SecurityModel', 'RobustSecurityModel']
except ImportError as e:
    # Log the specific import error
    error_msg = f"Could not import security models: {str(e)}"
    logger.warning(error_msg)
    
    # Try to determine which specific dependencies are missing
    missing_deps = []
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing_deps.append("transformers")
        
    MISSING_DEPENDENCIES = missing_deps
    
    # Provide more helpful dummy classes with detailed error messages
    class DummySecurityModel(BaseModel):
        """Placeholder when the real SecurityModel can't be imported due to missing dependencies."""
        def __init__(self, *args, **kwargs):
            self.error_message = f"SecurityModel dependencies not installed: {', '.join(missing_deps or ['unknown'])}"
            self.missing_dependencies = missing_deps
            logger.warning(self.error_message)
            
        def predict(self, input_data):
            return {
                "error": self.error_message,
                "recommendation": "Install required dependencies: pip install " + " ".join(self.missing_dependencies)
            }
            
        def detect_threats(self, *args, **kwargs):
            return self.predict(None)
            
        def explain(self, *args, **kwargs):
            return self.predict(None)
            
    class DummyRobustSecurityModel(DummySecurityModel):
        """Placeholder when the real RobustSecurityModel can't be imported."""
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.error_message = f"RobustSecurityModel dependencies not installed: {', '.join(missing_deps or ['unknown'])}"
        
    SecurityModel = DummySecurityModel
    RobustSecurityModel = DummyRobustSecurityModel
    __all__ = ['BaseModel', 'SecurityModel', 'RobustSecurityModel']

# Make model availability information accessible
def get_model_availability():
    """Return information about model availability and missing dependencies."""
    return {
        "advanced_models_available": ADVANCED_MODELS_AVAILABLE,
        "missing_dependencies": MISSING_DEPENDENCIES
    } 
"""
AI package for EGen Security AI.

This package contains modules for AI models, training, and related utilities.
"""

import logging

logger = logging.getLogger(__name__)

# Track module availability
MODULES_AVAILABLE = {
    "models": True,  # Basic models should always be available
    "trainers": False
}

# Import base model - this should always work
from src.ai.models import BaseModel

# Try to import security model - may fail if dependencies are missing
try:
    from src.ai.models import SecurityModel
    from src.ai.models import get_model_availability
    model_status = get_model_availability()
    if not model_status["advanced_models_available"]:
        logger.warning(f"Advanced security models are not fully functional. Missing: {model_status['missing_dependencies']}")
except ImportError as e:
    logger.error(f"Failed to import SecurityModel: {str(e)}")
    # Create a minimal dummy class if import fails
    class DummySecurityModel:
        def __init__(self, *args, **kwargs):
            self.error = f"SecurityModel failed to import: {str(e)}"
        def predict(self, *args, **kwargs):
            return {"error": self.error}
    SecurityModel = DummySecurityModel

# Try to import trainers - these may not be available
try:
    from src.ai.trainers import BaseTrainer, SecurityTrainer
    MODULES_AVAILABLE["trainers"] = True
except ImportError as e:
    logger.warning(f"Could not import trainers: {str(e)}")
    # Create detailed dummy classes if imports fail
    class BaseTrainer:
        """Placeholder when the real BaseTrainer can't be imported."""
        def __init__(self, *args, **kwargs):
            self.error_message = f"Trainer modules not available: {str(e)}"
            logger.warning(self.error_message)
            
        def train(self, *args, **kwargs):
            return {
                "error": self.error_message,
                "status": "failed",
                "recommendation": "Check your installation and dependencies"
            }
            
        def evaluate(self, *args, **kwargs):
            return {
                "error": self.error_message,
                "status": "failed",
                "recommendation": "Check your installation and dependencies"
            }
            
    class SecurityTrainer(BaseTrainer):
        """Placeholder when the real SecurityTrainer can't be imported."""
        pass

def get_module_availability():
    """Return information about which AI modules are available."""
    status = MODULES_AVAILABLE.copy()
    
    # Add more detailed model information if available
    try:
        from src.ai.models import get_model_availability
        status["models_details"] = get_model_availability()
    except:
        status["models_details"] = {"error": "Could not retrieve detailed model information"}
        
    return status

__all__ = ['BaseModel', 'SecurityModel', 'BaseTrainer', 'SecurityTrainer', 'get_module_availability'] 
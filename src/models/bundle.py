import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

def save_model_bundle(model, features, config, cutoff, save_path="models/model_bundle.pkl"):
    bundle = {
        'model': model,
        'features': features,
        'config': config,
        'metadata': {
            'training_cutoff': str(cutoff),
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0'
        }
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(bundle, save_path)
    logger.info(f"Model bundle saved to {save_path}")

def load_model_bundle(load_path="models/model_bundle.pkl"):
    return joblib.load(load_path)

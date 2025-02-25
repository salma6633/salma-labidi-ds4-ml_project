import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def save_model(model, filepath):
    """
    Sauvegarde le modèle dans un fichier.
    """
    joblib.dump(model, filepath)
    logger.info(f"Modèle sauvegardé dans {filepath}.")

def load_model(filepath):
    """
    Charge un modèle depuis un fichier.
    """
    model = joblib.load(filepath)
    logger.info(f"Modèle chargé depuis {filepath}.")
    return model
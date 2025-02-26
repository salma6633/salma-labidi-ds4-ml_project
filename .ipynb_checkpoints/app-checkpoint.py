from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import (
    load_iris,
)  # Utilisé pour l'exemple (remplacer par vos propres données)
import logging

# Créer l'application FastAPI
app = FastAPI()

# Configuration du logger pour afficher les erreurs et informations
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chemin du modèle
MODEL_PATH = "customer_churn_gbm_model.pkl"

# Charger le modèle
try:
    model = joblib.load(MODEL_PATH)
    logger.info("Modèle chargé avec succès.")
except Exception as e:
    logger.error(f"Erreur lors du chargement du modèle: {e}")
    raise HTTPException(
        status_code=500, detail=f"Erreur lors du chargement du modèle: {e}"
    )


# Définir le schéma de données pour la prédiction
class InputData(BaseModel):
    Account_length: float
    International_plan: float
    Number_vmail_messages: float
    Total_day_calls: float
    Total_day_charge: float
    Total_eve_calls: float
    Total_eve_charge: float
    Total_night_calls: float
    Total_night_charge: float
    Total_intl_calls: float
    Total_intl_charge: float
    Customer_service_calls: float
    state: str  # Géré séparément dans la fonction


# Schéma pour les nouveaux hyperparamètres dans la fonction retrain
class Hyperparameters(BaseModel):
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_samples_split: int
    min_samples_leaf: int


# Fonction pour convertir l'état en valeur numérique
def encode_state(state: str) -> int:
    state_groups = {
        2: ["AR", "CA", "MD", "ME", "MI", "NJ", "MS", "MT", "NV", "SC", "TX", "WA"],
        0: [
            "AK",
            "AL",
            "AZ",
            "DC",
            "HI",
            "IA",
            "IL",
            "LA",
            "MO",
            "MD",
            "NE",
            "NM",
            "RI",
            "TN",
            "VA",
            "VT",
            "WI",
            "WV",
            "WY",
        ],
    }
    for key, states in state_groups.items():
        if state in states:
            return key
    return 1  # Par défaut, groupe 1


# Point de terminaison de prédiction
@app.post("/predict")
def predict(data: InputData):
    try:
        # Encoder l'état
        state_value = encode_state(data.state)

        # Calculer le score d'utilisation (exemple de pondération)
        weights = [0.4, 0.3, 0.2, 0.1]
        usage_score = (
            data.Total_day_charge * weights[0]
            + data.Total_eve_charge * weights[1]
            + data.Total_night_charge * weights[2]
            + data.Total_intl_charge * weights[3]
        )

        # Transformer les données en tableau NumPy
        features = np.array(
            [
                data.Account_length,
                data.International_plan,
                data.Number_vmail_messages,
                data.Total_day_calls,
                data.Total_day_charge,
                data.Total_eve_calls,
                data.Total_eve_charge,
                data.Total_night_calls,
                data.Total_night_charge,
                data.Total_intl_calls,
                data.Total_intl_charge,
                data.Customer_service_calls,
                state_value,
                usage_score,
            ]
        ).reshape(1, -1)

        # Prédiction
        prediction = model.predict(features)
        probability = model.predict_proba(features)[0][1]

        return {
            "prediction": int(prediction[0]),
            "churn_probability": round(probability, 4),
        }

    except Exception as e:
        logger.error(f"Erreur lors de la prédiction: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erreur lors de la prédiction: {e}"
        )


# Point de terminaison pour réentraîner le modèle
@app.post("/retrain")
def retrain(hyperparameters: Hyperparameters):
    try:
        # Charger les données (remplacer par vos propres données)
        data = load_iris()  # Exemple avec Iris dataset
        X = data.data
        y = data.target

        # Diviser les données en jeux d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Réentraîner le modèle avec les nouveaux hyperparamètres
        logger.info(f"Réentraîner avec les hyperparamètres: {hyperparameters.dict()}")

        # Initialisation du modèle avec les nouveaux hyperparamètres
        model = GradientBoostingClassifier(
            n_estimators=hyperparameters.n_estimators,
            learning_rate=hyperparameters.learning_rate,
            max_depth=hyperparameters.max_depth,
            min_samples_split=hyperparameters.min_samples_split,
            min_samples_leaf=hyperparameters.min_samples_leaf,
        )

        # Entraîner le modèle
        model.fit(X_train, y_train)

        # Sauvegarder le modèle réentrainé
        joblib.dump(model, MODEL_PATH)

        logger.info("Modèle réentraîné avec succès")

        return {
            "message": "Modèle réentrainé avec succès",
            "new_model_path": MODEL_PATH,
        }

    except Exception as e:
        logger.error(f"Erreur lors du réentrainement du modèle: {e}")
        raise HTTPException(
            status_code=500, detail=f"Erreur lors du réentrainement du modèle: {e}"
        )

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_model(X_train, y_train):
    """
    Entraîne un modèle Gradient Boosting Machine (GBM) avec RandomizedSearchCV.
    """
    logger.info("Début de l'entraînement du modèle...")
    gbm = GradientBoostingClassifier(random_state=42)

    # Grille d'hyperparamètres
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 10],
        "min_samples_leaf": [3, 5, 7],
        "max_features": ["sqrt", "log2", None],
    }

    # Recherche aléatoire
    random_search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_grid,
        n_iter=10,
        scoring="f1",
        cv=3,
        verbose=2,
        n_jobs=-1,
    )

    random_search.fit(X_train, y_train)
    best_model = random_search.best_estimator_
    logger.info(f"Meilleurs hyperparamètres trouvés : {random_search.best_params_}")

    return best_model

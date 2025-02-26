import argparse
import logging
from modelpipe import run_pipeline
from data_preparation import prepare_data
from model_training import train_model
from model_evaluation import evaluate_model
from model_utils import save_model, load_model
from mlflow_utils import log_model_to_mlflow, move_model_to_stage_automatically
from elasticsearch_utils import log_metrics_to_es
import mlflow
from mlflow.tracking import MlflowClient

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    # Configurer l'URI de suivi pour utiliser le serveur MLflow local
    mlflow.set_tracking_uri("http://localhost:5000")

    # Définir le nom de l'expérience MLflow
    mlflow.set_experiment("ExperimentFinal")

    # Configuration des arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la prédiction de churn."
    )
    parser.add_argument(
        "--data", type=str, required=True, help="Chemin du fichier de données."
    )
    parser.add_argument(
        "--run-all", action="store_true", help="Exécuter toutes les étapes du pipeline."
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Étape à exécuter : prepare, train, evaluate, save, load, log_model, promote",
    )
    args = parser.parse_args()

    if args.run_all:
        # Exécuter toutes les étapes du pipeline
        logger.info("Démarrage du pipeline complet...")
        metrics = run_pipeline(args.data)

        # Promouvoir le modèle vers un stage en fonction de l'accuracy
        if metrics and "accuracy" in metrics:
            accuracy = metrics["accuracy"]
            move_model_to_stage_automatically("CustomerChurnModel", accuracy)
        else:
            logger.error("Impossible de promouvoir le modèle : métriques manquantes.")

        logger.info("Pipeline terminé avec succès.")
    elif args.step:
        # Exécuter une étape spécifique
        logger.info(f"Démarrage de l'étape : {args.step}")
        if args.step == "prepare":
            # Étape 1 : Préparation des données
            logger.info("Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(args.data)
            logger.info("Données préparées avec succès.")

        elif args.step == "train":
            # Étape 2 : Entraînement du modèle
            logger.info("Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(args.data)
            logger.info("Entraînement du modèle...")
            model = train_model(X_train, y_train)
            logger.info("Modèle entraîné avec succès.")

        elif args.step == "evaluate":
            # Étape 3 : Évaluation du modèle
            logger.info("Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(args.data)
            logger.info("Chargement du modèle...")
            model = load_model("customer_churn_model.pkl")
            logger.info("Évaluation du modèle...")
            metrics = evaluate_model(model, X_test, y_test)
            logger.info(f"Métriques d'évaluation : {metrics}")

        elif args.step == "save":
            # Étape 4 : Sauvegarde du modèle
            logger.info("Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(args.data)
            logger.info("Entraînement du modèle...")
            model = train_model(X_train, y_train)
            logger.info("Sauvegarde du modèle...")
            save_model(model, "customer_churn_model.pkl")
            logger.info("Modèle sauvegardé avec succès.")

        elif args.step == "load":
            # Étape 5 : Chargement du modèle
            logger.info("Chargement du modèle...")
            model = load_model("customer_churn_model.pkl")
            logger.info("Modèle chargé avec succès.")

        elif args.step == "log_model":
            # Étape 6 : Enregistrement du modèle dans MLflow
            logger.info("Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(args.data)
            logger.info("Chargement du modèle...")
            model = load_model("customer_churn_model.pkl")
            logger.info("Enregistrement du modèle dans MLflow...")
            log_model_to_mlflow(model, X_train, X_test)
            logger.info("Modèle enregistré dans MLflow.")

        elif args.step == "promote":
            # Étape 7 : Promotion du modèle vers un stage
            logger.info("Préparation des données...")
            X_train, X_test, y_train, y_test = prepare_data(args.data)
            logger.info("Chargement du modèle...")
            model = load_model("customer_churn_model.pkl")
            logger.info("Évaluation du modèle...")
            metrics = evaluate_model(model, X_test, y_test)
            if metrics and "accuracy" in metrics:
                accuracy = metrics["accuracy"]
                move_model_to_stage_automatically("CustomerChurnModel", accuracy)
            else:
                logger.error(
                    "Impossible de promouvoir le modèle : métriques manquantes."
                )

        else:
            logger.error(f"Étape non reconnue : {args.step}")
    else:
        logger.error("Veuillez spécifier une option : --run-all ou --step <étape>.")


if __name__ == "__main__":
    main()

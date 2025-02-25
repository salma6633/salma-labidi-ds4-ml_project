import pickle
import argparse
import model_pipeline as mp
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import random
import os
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
from elasticsearch import Elasticsearch
import logging
from datetime import datetime

# Connexion à Elasticsearch
es = Elasticsearch([{"scheme": "http", "host": "localhost", "port": 9200}])
if es.ping():
    print("\033[92mConnexion à Elasticsearch réussie!\033[0m")
else:
    print("\033[91mLa connexion à Elasticsearch a échoué!\033[0m")

# Vérifier si l'index existe et le créer si nécessaire
index_name = "mlflow-metrics"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)
    print(f"\033[94mL'index '{index_name}' a été créé.\033[0m")

# Configurer le logger pour capturer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Fonction pour envoyer les logs à Elasticsearch
def log_metrics_to_es(metrics):
    try:
        if metrics:
            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            es.index(index=index_name, body=metrics)
            print("\033[92mMétriques envoyées à Elasticsearch.\033[0m")
        else:
            print("\033[91mAucune métrique à envoyer.\033[0m")
    except Exception as e:
        print(
            f"\033[91mErreur lors de l'envoi des métriques vers Elasticsearch : {e}\033[0m"
        )


# Fonction pour générer un nom aléatoire
def generate_random_name():
    adjectives = ["fast", "bright", "lucky", "bold", "clever"]
    animals = ["tiger", "panda", "eagle", "shark", "falcon"]
    return f"{random.choice(adjectives)}-{random.choice(animals)}-{random.randint(100, 999)}"


# Fonction pour tracer la courbe ROC
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label="Courbe ROC")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("Faux Positifs")
    plt.ylabel("Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()


# Fonction pour tracer la matrice de confusion
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Matrice de Confusion")
    plt.colorbar()
    plt.xlabel("Prédictions")
    plt.ylabel("Véritables")
    plt.savefig("confusion_matrix.png")
    plt.close()


# Fonction pour enregistrer le modèle dans MLflow
def log_model_to_mlflow(model, X_train, X_test, model_name="CustomerChurnModel"):
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",  # Chemin relatif pour stocker le modèle
        registered_model_name=model_name,  # Nom du modèle enregistré
        signature=signature,  # Signature du modèle
        input_example=X_test[:5],  # Exemple d'entrée pour le modèle
    )
    print(f"\033[92mModèle sauvegardé dans MLflow sous le nom '{model_name}'.\033[0m")


# Fonction pour évaluer le modèle
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "auc": roc_auc_score(y_test, y_pred_proba),
    }

    mlflow.log_metrics(metrics)
    log_metrics_to_es(metrics)

    plot_roc_curve(y_test, y_pred_proba)
    mlflow.log_artifact("roc_curve.png")
    plot_confusion_matrix(y_test, y_pred)
    mlflow.log_artifact("confusion_matrix.png")

    report = classification_report(y_test, y_pred)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    print("\033[94mRapport de classification :\033[0m")
    print(report)

    return metrics


# Fonction pour promouvoir le modèle vers un stage
def promote_model(model_name, stage, accuracy):
    client = MlflowClient()

    # Trouver la dernière version du modèle
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if len(model_versions) == 0:
        raise ValueError(
            f"\033[91mAucun modèle trouvé avec le nom {model_name}.\033[0m"
        )

    model_version = model_versions[0].version

    # Promouvoir le modèle vers le stage spécifié
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
    )
    print(
        f"\033[92mModèle {model_name} version {model_version} déplacé vers {stage}.\033[0m"
    )

    # Ajouter des tags pour le stage et l'accuracy
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="stage",
        value=stage,
    )
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="accuracy",
        value=str(accuracy),
    )
    print(f"\033[94mTags ajoutés : stage={stage}, accuracy={accuracy}\033[0m")


# Fonction pour sauvegarder le modèle
def save_model(model, filepath="customer_churn_model.pkl"):
    """
    Sauvegarde le modèle entraîné dans un fichier.

    Args:
        model: Le modèle entraîné.
        filepath (str): Chemin vers le fichier où le modèle sera sauvegardé.
    """
    import joblib

    joblib.dump(model, filepath)
    logger.info(f"Modèle sauvegardé avec succès dans {filepath}.")


# Fonction pour charger le modèle
def load_model(filepath="customer_churn_model.pkl"):
    """
    Charge un modèle sauvegardé à partir d'un fichier.

    Args:
        filepath (str): Chemin vers le fichier du modèle.

    Returns:
        model: Le modèle chargé.
    """
    import joblib

    model = joblib.load(filepath)
    logger.info(f"Modèle chargé avec succès depuis {filepath}.")
    return model


# Fonction pour déplacer automatiquement le modèle vers un stage
def move_model_to_stage_automatically(model_name, accuracy):
    if accuracy > 0.95:
        promote_model(model_name, "Production", accuracy)
    elif accuracy > 0.90:
        promote_model(model_name, "Staging", accuracy)
    else:
        promote_model(model_name, "Archived", accuracy)


# Fonction principale
def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la prédiction de churn avec MLflow."
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Étape à exécuter: prepare, train, evaluate, save, load, log_model, promote",
    )
    parser.add_argument("--data", type=str, help="Chemin du fichier de données")
    parser.add_argument(
        "--run-all", action="store_true", help="Exécuter toutes les étapes du pipeline"
    )
    args = parser.parse_args()

    mlflow.set_experiment("ExperimentFinal")

    if args.run_all:
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--run-all'.\033[0m"
            )

        print("\033[94mExécution complète du pipeline avec MLflow...\033[0m")
        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        with mlflow.start_run(run_name=generate_random_name()):
            print("\033[94mEntraînement du modèle...\033[0m")
            model = mp.train_model(X_train, y_train)
            mlflow.log_params(model.get_params())
            print("\033[92mEntraînement terminé.\033[0m")

            # Sauvegarder le modèle
            save_model(model)

            # Évaluer le modèle
            metrics = evaluate_model(model, X_test, y_test)

            # Enregistrer le modèle dans MLflow
            log_model_to_mlflow(model, X_train, X_test)

            # Promouvoir le modèle vers un stage
            move_model_to_stage_automatically("CustomerChurnModel", metrics["accuracy"])

    elif args.step == "prepare":
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--step prepare'.\033[0m"
            )

        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        # Sauvegarder les données préparées
        with open("customer_churn_model.pkl", "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print(
            "\033[92mDonnées préparées sauvegardées dans 'customer_churn_model.pkl'.\033[0m"
        )

    elif args.step == "train":
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--step train'.\033[0m"
            )

        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        print("\033[94mEntraînement du modèle...\033[0m")
        model = mp.train_model(X_train, y_train)
        print("\033[92mEntraînement terminé.\033[0m")

        # Sauvegarder le modèle
        save_model(model)

    elif args.step == "evaluate":
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--step evaluate'.\033[0m"
            )

        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        # Charger le modèle
        model = load_model()
        print("\033[92mModèle chargé avec succès.\033[0m")

        # Démarrer un run MLflow
        with mlflow.start_run(run_name="evaluate_run"):
            # Évaluer le modèle
            metrics = evaluate_model(model, X_test, y_test)

    elif args.step == "save":
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--step save'.\033[0m"
            )

        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        print("\033[94mEntraînement du modèle...\033[0m")
        model = mp.train_model(X_train, y_train)
        print("\033[92mEntraînement terminé.\033[0m")

        # Sauvegarder le modèle
        save_model(model)

    elif args.step == "load":
        # Charger le modèle
        model = load_model()
        print("\033[92mModèle chargé avec succès.\033[0m")

    elif args.step == "log_model":
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--step log_model'.\033[0m"
            )

        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        # Charger le modèle
        model = load_model()
        print("\033[92mModèle chargé avec succès.\033[0m")

        # Enregistrer le modèle dans MLflow
        with mlflow.start_run(run_name="log_model_run"):
            log_model_to_mlflow(model, X_train, X_test)

    elif args.step == "promote":
        if not args.data:
            raise ValueError(
                "\033[91mLe chemin du fichier de données est requis pour '--step promote'.\033[0m"
            )

        print("\033[94mPréparation des données...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        print("\033[92mPréparation terminée.\033[0m")

        # Charger le modèle
        model = load_model()
        print("\033[92mModèle chargé avec succès.\033[0m")

        # Évaluer le modèle
        metrics = evaluate_model(model, X_test, y_test)

        # Promouvoir le modèle
        move_model_to_stage_automatically("CustomerChurnModel", metrics["accuracy"])

    else:
        print(
            "\033[91mÉtape non reconnue. Utilisez --help pour voir les options disponibles.\033[0m"
        )


if __name__ == "__main__":
    main()

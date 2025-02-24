import pickle
import argparse
import model_pipeline as mp
import mlflow
import mlflow.sklearn
import os
import warnings
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

# Configuration MLflow
MLFLOW_DIR = os.path.join(os.getcwd(), "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

# Suppression des warnings indésirables
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow.types.utils")

# Connexion à Elasticsearch
es = Elasticsearch(
    [{"scheme": "http", "host": "localhost", "port": 9200}]
)  # Ajoute 'scheme' ici
if es.ping():
    print("✅ Connecté à Elasticsearch")
else:
    print("❌ Impossible de se connecter à Elasticsearch")

# Vérifier si l'index existe et le créer si nécessaire
index_name = "mlflow-metrics"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)
    print(f"L'index '{index_name}' a été créé.")

# Fonction pour envoyer les logs à Elasticsearch
def log_metrics_to_es(metrics):
    try:
        # Indexer les métriques dans Elasticsearch
        if metrics:
            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            es.index(index=index_name, body=metrics)
            print("✅ Métriques envoyées à Elasticsearch.")
        else:
            print("⚠️Aucune métrique à envoyer.")
    except Exception as e:
        print(f"❌Erreur lors de l'envoi des métriques vers Elasticsearch : {e}")

# Configurer le logger pour capturer les logs
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_random_name():
    adjectives = ["fast", "bright", "lucky", "bold", "clever"]
    animals = ["tiger", "panda", "eagle", "shark", "falcon"]
    return f"{random.choice(adjectives)}-{random.choice(animals)}-{random.randint(100, 999)}"

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

def move_model_to_stage_automatically(model_name, accuracy):
    """Promotion automatique du modèle vers un stage basé sur son accuracy."""
    print(f"Promotion du modèle {model_name} en fonction de l'accuracy : {accuracy}")

    if accuracy > 0.95:
        new_stage = "Production"
    elif accuracy > 0.90:
        new_stage = "Staging"
    else:
        new_stage = "Archived"

    # Obtenir la version la plus récente du modèle enregistré
    client = MlflowClient()
    model_versions = client.search_model_versions(f"name='{model_name}'")

    if len(model_versions) == 0:
        raise ValueError(f"Aucun modèle trouvé avec le nom {model_name}.")

    model_version = model_versions[0].version
    print(f"Version du modèle : {model_version}")

    # Transitionner le modèle vers le stage calculé
    client.transition_model_version_stage(
        name=model_name, version=model_version, stage=new_stage
    )
    print(f"Modèle {model_name} version {model_version} déplacé vers {new_stage}.")

    # Ajouter un tag pour indiquer le stage
    client.set_model_version_tag(
        name=model_name, version=model_version, key="stage", value=new_stage
    )
    print(f"Tag 'stage' ajouté avec la valeur '{new_stage}'.")

    # Ajouter un tag pour l'accuracy
    client.set_model_version_tag(
        name=model_name, version=model_version, key="accuracy", value=str(accuracy)
    )
    print(f"Tag 'accuracy' ajouté avec la valeur '{accuracy}'.")

    # Attendre quelques secondes pour la synchronisation
    time.sleep(5)  # Attente de 5 secondes pour permettre à MLflow d'appliquer les changements

    # Vérifier si le tag a bien été ajouté
    model_version_metadata = client.get_model_version(model_name, model_version)
    tags = model_version_metadata.tags
    if "stage" in tags:
        print(f"Tag 'stage' trouvé avec la valeur : {tags['stage']}")
    else:
        print("Le tag 'stage' n'a pas été trouvé.")

mlflow.set_experiment("ExperimentFinal")

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline ML pour la prédiction de churn avec MLflow."
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Étape à exécuter: prepare, train, evaluate, save, load, staging, production",
    )
    parser.add_argument("--data", type=str, help="Chemin du fichier de données")
    parser.add_argument(
        "--run-all", action="store_true", help="Exécuter toutes les étapes du pipeline"
    )
    args = parser.parse_args()

    if args.run_all:
        if not args.data:
            raise ValueError(
                "Le chemin du fichier de données est requis pour '--run-all'."
            )

        print("Exécution complète du pipeline avec MLflow...")
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        with open("customer_churn_model.pkl", "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print("Préparation terminée.")

        with mlflow.start_run(run_name=generate_random_name()):
            print("Entraînement du modèle...")
            model = mp.train_model(X_train, y_train)
            mlflow.log_params(model.get_params())
            print("Entraînement terminé.")

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

            # Vérification et envoi des métriques vers Elasticsearch
            print("Métriques à envoyer:", metrics)
            log_metrics_to_es(metrics)  # Envoi des métriques vers Elasticsearch

            plot_roc_curve(y_test, y_pred_proba)
            mlflow.log_artifact("roc_curve.png")
            plot_confusion_matrix(y_test, y_pred)
            mlflow.log_artifact("confusion_matrix.png")

            # Enregistrer le rapport de classification
            report = classification_report(y_test, y_pred)
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

            # Afficher le rapport de classification dans le terminal
            print("Rapport de classification :")
            print(report)

            # Enregistrer le modèle dans le registre MLflow
            signature = infer_signature(X_train, model.predict(X_train))
           
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name="ChurnModel",
                metadata={"experiment_path": MLFLOW_DIR}
            )

            print("Modèle sauvegardé.")

            # Promotion automatique vers le stage en fonction de l'accuracy
            move_model_to_stage_automatically("CustomerChurnModel", metrics["accuracy"])

    elif args.step == "prepare":
        if not args.data:
            raise ValueError(
                "Le chemin du fichier de données est requis pour 'prepare'."
            )
        print("Préparation des données...")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        with open("customer_churn_model.pkl", "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        print("Préparation terminée.")

    elif args.step == "train":
        print("Chargement des données...")
        with open("customer_churn_model.pkl", "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        with mlflow.start_run(run_name=generate_random_name()):
            print("Entraînement du modèle...")
            model = mp.train_model(X_train, y_train)

            mlflow.log_params(model.get_params())

            # Enregistrer le modèle dans le registre MLflow
            signature = infer_signature(X_train, model.predict(X_train))
            mlflow.sklearn.log_model(
                model,
                "customer_churn_model",
                input_example=X_test[:5],
                signature=signature,
                registered_model_name="CustomerChurnModel",
            )

            print("Modèle sauvegardé.")

    elif args.step == "evaluate":
        print("Chargement des données...")
        with open("customer_churn_model.pkl", "rb") as f:
            X_train, X_test, y_train, y_test = pickle.load(f)

        print("Chargement du modèle...")
        model = mp.load_model("customer_churn_gbm_model.pkl")

        with mlflow.start_run(run_name=generate_random_name()):
            print("Évaluation du modèle...")

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
            log_metrics_to_es(
                metrics
            )  # Appel à la fonction pour envoyer vers Elasticsearch

            plot_roc_curve(y_test, y_pred_proba)
            mlflow.log_artifact("roc_curve.png")
            plot_confusion_matrix(y_test, y_pred)
            mlflow.log_artifact("confusion_matrix.png")

            # Enregistrer le rapport de classification
            report = classification_report(y_test, y_pred)
            with open("classification_report.txt", "w") as f:
                f.write(report)
            mlflow.log_artifact("classification_report.txt")

            # Afficher le rapport de classification dans le terminal
            print("Rapport de classification :")
            print(report)

            print("Évaluation terminée.")

    elif args.step == "staging":
        model_name = "CustomerChurnModel"
        print("Déplacement du modèle vers Staging...")
        move_model_to_stage(model_name, "staging")
        print(f"Modèle {model_name} déplacé vers Staging avec le tag.")

    elif args.step == "production":
        model_name = "CustomerChurnModel"
        print("Déplacement du modèle vers Production...")
        move_model_to_stage(model_name, "production")
        print(f"Modèle {model_name} déplacé vers Production avec le tag.")

if __name__ == "__main__":
    main()
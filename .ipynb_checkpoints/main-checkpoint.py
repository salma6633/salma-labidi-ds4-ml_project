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
import pandas as pd
import numpy as np
from elasticsearch import Elasticsearch
import logging
from datetime import datetime

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connect to Elasticsearch
es = Elasticsearch([{"scheme": "http", "host": "localhost", "port": 9200}])
if es.ping():
    logger.info("\033[92mConnected to Elasticsearch!\033[0m")
else:
    logger.error("\033[91mFailed to connect to Elasticsearch!\033[0m")

# Check if the index exists, create it if not
index_name = "mlflow-metrics"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)
    logger.info(f"\033[94mIndex '{index_name}' created.\033[0m")

# Function to log metrics to Elasticsearch
def log_metrics_to_es(metrics):
    try:
        if metrics:
            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            es.index(index=index_name, body=metrics)
            logger.info("\033[92mMetrics sent to Elasticsearch.\033[0m")
        else:
            logger.warning("\033[91mNo metrics to send.\033[0m")
    except Exception as e:
        logger.error(f"\033[91mError sending metrics to Elasticsearch: {e}\033[0m")

# Function to generate a random name
def generate_random_name():
    adjectives = ["fast", "bright", "lucky", "bold", "clever"]
    animals = ["tiger", "panda", "eagle", "shark", "falcon"]
    return f"{random.choice(adjectives)}-{random.choice(animals)}-{random.randint(100, 999)}"

# Function to plot ROC curve
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    plt.close()

# Function to log model to MLflow
def log_model_to_mlflow(model, X_train, X_test, model_name="CustomerChurnModel"):
    try:
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",  # Relative path to store the model
            registered_model_name=model_name,  # Registered model name
            signature=signature,  # Model signature
            input_example=X_test[:5],  # Example input for the model
        )
        logger.info(f"\033[92mModel saved in MLflow as '{model_name}'.\033[0m")
    except Exception as e:
        logger.error(f"\033[91mError logging model to MLflow: {e}\033[0m")

# Function to evaluate the model
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

    # Generate and save ROC curve
    plot_roc_curve(y_test, y_pred_proba)
    mlflow.log_artifact("roc_curve.png")

    # Generate and save confusion matrix
    plot_confusion_matrix(y_test, y_pred)
    mlflow.log_artifact("confusion_matrix.png")

    # Save classification report
    report_path = "classification_report.txt"
    report = classification_report(y_test, y_pred)
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    logger.info("\033[94mClassification Report:\033[0m")
    logger.info(report)

    return metrics

# Function to promote model to a stage
def promote_model(model_name, stage, accuracy):
    client = MlflowClient()

    # Find the latest model version
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if len(model_versions) == 0:
        raise ValueError(f"\033[91mNo model found with name {model_name}.\033[0m")

    model_version = model_versions[0].version

    # Promote the model to the specified stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
    )
    logger.info(f"\033[92mModel {model_name} version {model_version} moved to {stage}.\033[0m")

    # Add tags for stage and accuracy
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
    logger.info(f"\033[94mTags added: stage={stage}, accuracy={accuracy}\033[0m")

# Function to save the model
def save_model(model, filepath="customer_churn_model.pkl"):
    try:
        import joblib
        joblib.dump(model, filepath)
        logger.info(f"\033[92mModel saved successfully to {filepath}.\033[0m")
    except Exception as e:
        logger.error(f"\033[91mError saving model: {e}\033[0m")

# Function to load the model
def load_model(filepath="customer_churn_model.pkl"):
    try:
        import joblib
        model = joblib.load(filepath)
        logger.info(f"\033[92mModel loaded successfully from {filepath}.\033[0m")
        return model
    except Exception as e:
        logger.error(f"\033[91mError loading model: {e}\033[0m")
        raise

# Function to move model to stage automatically
def move_model_to_stage_automatically(model_name, accuracy):
    if accuracy > 0.95:
        promote_model(model_name, "Production", accuracy)
    elif accuracy > 0.90:
        promote_model(model_name, "Staging", accuracy)
    else:
        promote_model(model_name, "Archived", accuracy)

# Main function
def main():
    parser = argparse.ArgumentParser(
        description="ML pipeline for churn prediction with MLflow."
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Step to execute: prepare, train, evaluate, save, load, log_model, promote",
    )
    parser.add_argument("--data", type=str, help="Path to the data file")
    parser.add_argument(
        "--run-all", action="store_true", help="Run all pipeline steps"
    )
    args = parser.parse_args()

    mlflow.set_experiment("ExperimentFinal")

    if args.run_all:
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--run-all'.\033[0m"
            )

        logger.info("\033[94mRunning full pipeline with MLflow...\033[0m")
        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        with mlflow.start_run(run_name=generate_random_name()):
            logger.info("\033[94mTraining model...\033[0m")
            model = mp.train_model(X_train, y_train)
            mlflow.log_params(model.get_params())
            logger.info("\033[92mTraining complete.\033[0m")

            # Save the model
            save_model(model)

            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test)

            # Log the model to MLflow
            log_model_to_mlflow(model, X_train, X_test)

            # Promote the model to a stage
            move_model_to_stage_automatically("CustomerChurnModel", metrics["accuracy"])

    elif args.step == "prepare":
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--step prepare'.\033[0m"
            )

        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        # Save prepared data
        with open("customer_churn_model.pkl", "wb") as f:
            pickle.dump((X_train, X_test, y_train, y_test), f)
        logger.info("\033[92mPrepared data saved to 'customer_churn_model.pkl'.\033[0m")

    elif args.step == "train":
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--step train'.\033[0m"
            )

        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        logger.info("\033[94mTraining model...\033[0m")
        model = mp.train_model(X_train, y_train)
        logger.info("\033[92mTraining complete.\033[0m")

        # Save the model
        save_model(model)

    elif args.step == "evaluate":
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--step evaluate'.\033[0m"
            )

        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        # Load the model
        model = load_model()
        logger.info("\033[92mModel loaded successfully.\033[0m")

        # Start an MLflow run
        with mlflow.start_run(run_name="evaluate_run"):
            # Evaluate the model
            metrics = evaluate_model(model, X_test, y_test)

    elif args.step == "save":
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--step save'.\033[0m"
            )

        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        logger.info("\033[94mTraining model...\033[0m")
        model = mp.train_model(X_train, y_train)
        logger.info("\033[92mTraining complete.\033[0m")

        # Save the model
        save_model(model)

    elif args.step == "load":
        # Load the model
        model = load_model()
        logger.info("\033[92mModel loaded successfully.\033[0m")

    elif args.step == "log_model":
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--step log_model'.\033[0m"
            )

        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        # Load the model
        model = load_model()
        logger.info("\033[92mModel loaded successfully.\033[0m")

        # Log the model to MLflow
        with mlflow.start_run(run_name="log_model_run"):
            log_model_to_mlflow(model, X_train, X_test)

    elif args.step == "promote":
        if not args.data:
            raise ValueError(
                "\033[91mData file path is required for '--step promote'.\033[0m"
            )

        logger.info("\033[94mPreparing data...\033[0m")
        X_train, X_test, y_train, y_test = mp.prepare_data(args.data)
        logger.info("\033[92mData preparation complete.\033[0m")

        # Load the model
        model = load_model()
        logger.info("\033[92mModel loaded successfully.\033[0m")

        # Evaluate the model
        metrics = evaluate_model(model, X_test, y_test)

        # Promote the model
        move_model_to_stage_automatically("CustomerChurnModel", metrics["accuracy"])

    else:
        logger.error("\033[91mUnrecognized step. Use --help to see available options.\033[0m")

if __name__ == "__main__":
    main()
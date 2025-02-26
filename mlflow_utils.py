import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature


def log_model_to_mlflow(model, X_train, X_test, model_name="CustomerChurnModel"):
    """
    Enregistre le modèle dans MLflow.

    Args:
        model: Modèle entraîné.
        X_train (pd.DataFrame): Données d'entraînement.
        X_test (pd.DataFrame): Données de test.
        model_name (str): Nom du modèle.
    """
    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(
        model,
        artifact_path="model",
        registered_model_name=model_name,
        signature=signature,
        input_example=X_test[:5],
    )
    print(f"Modèle sauvegardé dans MLflow sous le nom '{model_name}'.")

def move_model_to_stage_automatically(model_name, accuracy):
    """
    Déplace automatiquement le modèle vers un stage en fonction de son accuracy.

    Args:
        model_name (str): Nom du modèle.
        accuracy (float): Accuracy du modèle.
    """
    client = MlflowClient()

    # Trouver la dernière version du modèle
    model_versions = client.search_model_versions(f"name='{model_name}'")
    if len(model_versions) == 0:
        raise ValueError(f"Aucun modèle trouvé avec le nom {model_name}.")

    model_version = model_versions[0].version

    # Définir le stage en fonction de l'accuracy
    if accuracy > 0.95:
        stage = "Production"
    elif accuracy > 0.90:
        stage = "Staging"
    else:
        stage = "Archived"

    # Promouvoir le modèle vers le stage calculé
    client.transition_model_version_stage(
        name=model_name,
        version=model_version,
        stage=stage,
    )
    print(f"Modèle {model_name} version {model_version} déplacé vers {stage}.")

    # Ajouter un tag pour indiquer le stage
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="stage",
        value=stage,
    )
    print(f"Tag 'stage' ajouté avec la valeur : {stage}")

    # Ajouter un tag pour l'accuracy
    client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="accuracy",
        value=str(accuracy),
    )
    print(f"Tag 'accuracy' ajouté avec la valeur : {accuracy}")
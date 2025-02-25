from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle.
    """
    logger.info("Évaluation du modèle...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calcul des métriques
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    report = classification_report(y_test, y_pred)

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    plt.close()

    # Courbe ROC
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color="blue", label="ROC Curve")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("Faux Positifs")
    plt.ylabel("Vrais Positifs")
    plt.title("Courbe ROC")
    plt.legend(loc="lower right")
    plt.savefig("roc_curve.png")
    plt.close()

    # Retourner les métriques
    return {
        "accuracy": accuracy,
        "roc_auc_score": roc_auc,
        "classification_report": report,
    }
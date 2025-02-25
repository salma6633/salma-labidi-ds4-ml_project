# Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from scipy.optimize import minimize
from sklearn.cluster import KMeans

# Encodage des variables catégorielles
from sklearn.preprocessing import LabelEncoder

# Division des données
from sklearn.model_selection import train_test_split

# Normalisation des données
from sklearn.preprocessing import StandardScaler

# Gestion du déséquilibre des classes
from imblearn.over_sampling import SMOTE

# Modélisation
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

# Évaluation
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

# Sauvegarde et chargement des modèles
import joblib

# Logging
import logging

# Configuration du logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_data(data_path):
    """
    Charge les données, traite les outliers, encode les variables catégorielles,
    et divise les données en ensembles d'entraînement et de test.

    Args:
        data_path (str): Chemin vers le fichier CSV contenant les données.

    Returns:
        X_train_scaled_smote, X_test_scaled_smote, y_train_smote, y_test_smote: Les données préparées.
    """
    logger.info("Chargement des données...")
    df = pd.read_csv(data_path)

    # Traitement des outliers
    logger.info("Traitement des outliers...")
    columns_outliers = ["Total eve calls", "Total day calls", "Total intl calls"]
    for column in columns_outliers:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    # Sauvegarde du boxplot après traitement des outliers
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df.drop(columns=["Churn"]), whis=3)
    plt.title("Boxplot after handling outliers")
    plt.xticks(rotation=45)
    plt.savefig("boxplot_outliers.png")
    plt.close()

    # Encodage des variables catégorielles
    logger.info("Encodage des variables catégorielles...")
    label_encoder = LabelEncoder()
    df["International plan"] = label_encoder.fit_transform(df["International plan"])
    df["Voice mail plan"] = label_encoder.fit_transform(df["Voice mail plan"])
    df["Churn"] = label_encoder.fit_transform(df["Churn"])

    # Clustering des états par taux de churn
    logger.info("Clustering des états par taux de churn...")
    state_churn_rate = df.groupby("State")["Churn"].mean().reset_index()
    state_churn_rate.rename(columns={"Churn": "Churn_Rate"}, inplace=True)
    X = state_churn_rate[["Churn_Rate"]].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    state_churn_rate["Cluster"] = kmeans.fit_predict(X)
    cluster_mapping = (
        state_churn_rate.groupby("Cluster")["Churn_Rate"]
        .mean()
        .sort_values()
        .index.to_list()
    )
    cluster_labels = {
        cluster_mapping[0]: "Low",
        cluster_mapping[1]: "Medium",
        cluster_mapping[2]: "High",
    }
    state_churn_rate["State_Category"] = state_churn_rate["Cluster"].map(cluster_labels)

    # Sauvegarde du graphique des clusters
    plt.figure(figsize=(14, 4))
    for category, group in state_churn_rate.groupby("State_Category"):
        plt.scatter(group["State"], group["Churn_Rate"], label=category)
    plt.xlabel("State")
    plt.ylabel("Churn Rate")
    plt.title("Churn Rate Clusters")
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig("churn_rate_clusters.png")
    plt.close()

    # Encodage personnalisé pour State_Category
    custom_encoding = {"Low": 0, "Medium": 1, "High": 2}
    df = df.merge(state_churn_rate[["State", "State_Category"]], on="State", how="left")
    df["State_Category"] = df["State_Category"].map(custom_encoding)

    # Statistiques par catégorie d'état
    state_stats = (
        df.groupby("State_Category")
        .agg(Num_Customers=("Churn", "size"), Churn_Rate=("Churn", "mean"))
        .reset_index()
    )
    logger.info(f"Statistiques par catégorie d'état :\n{state_stats}")

    # Suppression de la colonne State
    df.drop(columns=["State"], inplace=True)

    # Sauvegarde de la matrice de corrélation
    correlation_matrix = df.select_dtypes(include=["number"]).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix of Numerical Features")
    plt.savefig("correlation_matrix.png")
    plt.close()

    # Suppression de colonnes inutiles
    df.drop(
        columns=[
            "Area code",
            "Voice mail plan",
            "Total day minutes",
            "Total eve minutes",
            "Total night minutes",
            "Total intl minutes",
        ],
        inplace=True,
    )

    # Calcul du score d'utilisation
    corr_day = df["Total day charge"].corr(df["Churn"])
    corr_eve = df["Total eve charge"].corr(df["Churn"])
    corr_night = df["Total night charge"].corr(df["Churn"])
    corr_intl = df["Total intl charge"].corr(df["Churn"])

    total_corr = abs(corr_day) + abs(corr_eve) + abs(corr_night) + abs(corr_intl)
    weights = [
        abs(corr_day) / total_corr,
        abs(corr_eve) / total_corr,
        abs(corr_night) / total_corr,
        abs(corr_intl) / total_corr,
    ]

    logger.info(f"Pondérations dynamiques basées sur la corrélation : {weights}")

    df["Usage Score"] = (
        df["Total day charge"] * weights[0]
        + df["Total eve charge"] * weights[1]
        + df["Total night charge"] * weights[2]
        + df["Total intl charge"] * weights[3]
    )

    # Définir les colonnes numériques
    numerical_columns_n = df.select_dtypes(include=["int64", "float64"]).columns
    numerical_columns_n = numerical_columns_n.drop(
        ["State_Category", "International plan", "Churn"]
    )

    logger.info(f"Colonnes numériques : {numerical_columns_n}")

    # Séparation des données en ensembles d'entraînement et de test
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_columns_n] = scaler.fit_transform(
        X_train[numerical_columns_n]
    )

    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_columns_n] = scaler.transform(X_test[numerical_columns_n])

    # Application de SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    X_test_scaled_smote, y_test_smote = smote.fit_resample(X_test_scaled, y_test)

    logger.info("Préparation des données terminée.")
    return X_train_scaled_smote, X_test_scaled_smote, y_train_smote, y_test_smote


def train_model(X_train, y_train):
    """
    Entraîne un modèle Gradient Boosting Machine (GBM) en utilisant RandomizedSearchCV.

    Args:
        X_train: Les données d'entraînement (features).
        y_train: Les labels d'entraînement (target).

    Returns:
        model: Le modèle entraîné.
    """
    logger.info("Début de l'entraînement du modèle...")
    gbm = GradientBoostingClassifier(random_state=42)

    # Grille d'hyperparamètres optimisée
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 5, 10],
        "min_samples_leaf": [3, 5, 7],
        "max_features": ["sqrt", "log2", None],
    }

    # RandomizedSearchCV
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


def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur les données de test.

    Args:
        model: Le modèle entraîné.
        X_test: Les données de test (features).
        y_test: Les labels de test (target).
    """
    logger.info("Évaluation du modèle...")
    y_pred = model.predict(X_test)

    # Rapport de classification
    logger.info("\nClassification Report:")
    logger.info(classification_report(y_test, y_pred))

    # Matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig("confusion_matrix.png")
    plt.close()


def save_model(model, filepath):
    """
    Sauvegarde le modèle entraîné dans un fichier.

    Args:
        model: Le modèle entraîné.
        filepath (str): Chemin vers le fichier où le modèle sera sauvegardé.
    """
    joblib.dump(model, filepath)
    logger.info(f"Modèle sauvegardé avec succès dans {filepath}.")


def load_model(filepath):
    """
    Charge un modèle sauvegardé à partir d'un fichier.

    Args:
        filepath (str): Chemin vers le fichier du modèle.

    Returns:
        model: Le modèle chargé.
    """
    model = joblib.load(filepath)
    logger.info(f"Modèle chargé avec succès depuis {filepath}.")
    return model

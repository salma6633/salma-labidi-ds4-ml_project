import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_data(data_path):
    """
    Charge les données, traite les outliers, encode les variables catégorielles,
    et divise les données en ensembles d'entraînement et de test.
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

    # Encodage personnalisé pour State_Category
    custom_encoding = {"Low": 0, "Medium": 1, "High": 2}
    df = df.merge(state_churn_rate[["State", "State_Category"]], on="State", how="left")
    df["State_Category"] = df["State_Category"].map(custom_encoding)

    # Suppression de la colonne State
    df.drop(columns=["State"], inplace=True)  # Correction ici : indentation correcte

    # Normalisation des données
    numerical_columns = df.select_dtypes(include=["int64", "float64"]).columns
    numerical_columns = numerical_columns.drop(["State_Category", "International plan", "Churn"])

    # Séparation des données
    X = df.drop(columns=["Churn"])
    y = df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_columns] = scaler.fit_transform(X_train[numerical_columns])
    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_columns] = scaler.transform(X_test[numerical_columns])

    # Application de SMOTE
    smote = SMOTE(random_state=42)
    X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    X_test_scaled_smote, y_test_smote = smote.fit_resample(X_test_scaled, y_test)

    logger.info("Préparation des données terminée.")
    return X_train_scaled_smote, X_test_scaled_smote, y_train_smote, y_test_smote
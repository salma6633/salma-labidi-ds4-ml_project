#Imports :
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import plot_tree
from scipy.optimize import minimize
#Imports for encoding categorical features
from sklearn.preprocessing import LabelEncoder
#Imports for splitting
from sklearn.model_selection import train_test_split
#Imports for Standarization
from sklearn.preprocessing import StandardScaler
#Imports for Class Imbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
#Imports for Modeling
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cluster import KMeans
from scipy.stats import mode
import xgboost as xgb
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
#Imports of hyperparameters
from sklearn.model_selection import GridSearchCV
#Imports for evaluation
import shap
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix ,ConfusionMatrixDisplay ,roc_curve, auc ,precision_score , recall_score,f1_score, roc_auc_score, precision_recall_curve
#Imports for deployment
import joblib
from joblib import dump, load
from flask import Flask, request, jsonify
from PIL import Image
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV

def prepare_data(data_path):
    """
    Charge les données, traite les outliers, encode les variables catégorielles,
    et divise les données en ensembles d'entraînement et de test.

    Args:
        data_path (str): Chemin vers le fichier CSV contenant les données.

    Returns:
        X_train_scaled_smote, X_test_scaled_smote, y_train_smote, y_test_smote: Les données préparées.
    """
    # Charger le fichier data_churn.csv
    df = pd.read_csv(data_path)

    # Traitement des outliers
    columns_outliers = ['Total eve calls', 'Total day calls', 'Total intl calls']
    for column in columns_outliers:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 3 * IQR
        upper_bound = Q3 + 3 * IQR
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)

    # Sauvegarde du boxplot après traitement des outliers
    plt.figure(figsize=(15, 6))
    sns.boxplot(data=df.drop(columns=['Churn']), whis=3)
    plt.title("Boxplot after handling outliers")
    plt.xticks(rotation=45)
    plt.savefig('boxplot_outliers.png')  # Sauvegarde le graphique
    plt.close()

    # Encodage des variables catégorielles
    label_encoder = LabelEncoder()
    df['International plan'] = label_encoder.fit_transform(df['International plan'])
    df['Voice mail plan'] = label_encoder.fit_transform(df['Voice mail plan'])
    df['Churn'] = label_encoder.fit_transform(df['Churn'])

    # Clustering des états par taux de churn
    state_churn_rate = df.groupby('State')['Churn'].mean().reset_index()
    state_churn_rate.rename(columns={'Churn': 'Churn_Rate'}, inplace=True)
    X = state_churn_rate[['Churn_Rate']].values
    kmeans = KMeans(n_clusters=3, random_state=42)
    state_churn_rate['Cluster'] = kmeans.fit_predict(X)
    cluster_mapping = (
        state_churn_rate.groupby('Cluster')['Churn_Rate']
        .mean()
        .sort_values()
        .index.to_list()
    )
    cluster_labels = {cluster_mapping[0]: 'Low', cluster_mapping[1]: 'Medium', cluster_mapping[2]: 'High'}
    state_churn_rate['State_Category'] = state_churn_rate['Cluster'].map(cluster_labels)

    # Sauvegarde du graphique des clusters
    plt.figure(figsize=(14, 4))
    for category, group in state_churn_rate.groupby('State_Category'):
        plt.scatter(group['State'], group['Churn_Rate'], label=category)
    plt.xlabel('State')
    plt.ylabel('Churn Rate')
    plt.title('Churn Rate Clusters')
    plt.legend()
    plt.xticks(rotation=90)
    plt.savefig('churn_rate_clusters.png')  # Sauvegarde le graphique
    plt.close()

    # Encodage personnalisé pour State_Category
    custom_encoding = {'Low': 0, 'Medium': 1, 'High': 2}
    df = df.merge(state_churn_rate[['State', 'State_Category']], on='State', how='left')
    df['State_Category'] = df['State_Category'].map(custom_encoding)

    # Statistiques par catégorie d'état
    state_stats = df.groupby('State_Category').agg(
        Num_Customers=('Churn', 'size'),
        Churn_Rate=('Churn', 'mean')
    ).reset_index()
    print(state_stats)

    # Suppression de la colonne State
    df.drop(columns=['State'], inplace=True)

    # Sauvegarde de la matrice de corrélation
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Correlation Matrix of Numerical Features')
    plt.savefig('correlation_matrix.png')  # Sauvegarde le graphique
    plt.close()

    # Suppression de colonnes inutiles
    df.drop(columns=['Area code', 'Voice mail plan', 'Total day minutes', 'Total eve minutes', 'Total night minutes', 'Total intl minutes'], inplace=True)

    # Calcul du score d'utilisation
    corr_day = df['Total day charge'].corr(df['Churn'])
    corr_eve = df['Total eve charge'].corr(df['Churn'])
    corr_night = df['Total night charge'].corr(df['Churn'])
    corr_intl = df['Total intl charge'].corr(df['Churn'])

    total_corr = abs(corr_day) + abs(corr_eve) + abs(corr_night) + abs(corr_intl)
    weights = [abs(corr_day) / total_corr, abs(corr_eve) / total_corr, abs(corr_night) / total_corr, abs(corr_intl) / total_corr]

    print(f"Dynamic Weights Based on Correlation: {weights}")

    df['Usage Score'] = (
        df['Total day charge'] * weights[0] +
        df['Total eve charge'] * weights[1] +
        df['Total night charge'] * weights[2] +
        df['Total intl charge'] * weights[3]
    )

    # Définir les colonnes numériques
    numerical_columns_n = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_columns_n = numerical_columns_n.drop(['State_Category', 'International plan', 'Churn'])

    print("Colonnes numériques :", numerical_columns_n)

    # Séparation des données en ensembles d'entraînement et de test
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Normalisation des données
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_train_scaled[numerical_columns_n] = scaler.fit_transform(X_train[numerical_columns_n])

    X_test_scaled = X_test.copy()
    X_test_scaled[numerical_columns_n] = scaler.transform(X_test[numerical_columns_n])

    # Application de SMOTE pour équilibrer les classes
    smote = SMOTE(random_state=42)
    X_train_scaled_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)
    X_test_scaled_smote, y_test_smote = smote.fit_resample(X_test_scaled, y_test)

    return X_train_scaled_smote, X_test_scaled_smote, y_train_smote, y_test_smote

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV

def train_model(X_train, y_train):
    """
    Entraîne un modèle Gradient Boosting Machine (GBM) en utilisant RandomizedSearchCV.

    Args:
        X_train: Les données d'entraînement (features).
        y_train: Les labels d'entraînement (target).

    Returns:
        model: Le modèle entraîné.
    """
    # Initialisation du modèle GBM
    gbm = GradientBoostingClassifier(random_state=42)

    # Grille d'hyperparamètres optimisée
    param_grid = {
        'n_estimators': [50, 100, 150],  # Réduire le nombre d'arbres
        'learning_rate': [0.01, 0.05, 0.1],  # Apprentissage plus rapide
        'max_depth': [3, 5, 10],  # Profondeur limitée pour éviter le surajustement
        'min_samples_leaf': [3, 5, 7],  # Moins de feuilles pour moins de complexité
        'max_features': ['sqrt', 'log2', None],  # Moins de variables utilisées par arbre
    }

    # RandomizedSearchCV au lieu de GridSearchCV (plus rapide)
    random_search = RandomizedSearchCV(
        estimator=gbm,
        param_distributions=param_grid,
        n_iter=10,  # Réduction du nombre d'itérations
        scoring='f1',
        cv=3,  # Moins de folds en validation croisée
        verbose=2,
        n_jobs=-1  # Utilisation de tous les cœurs du processeur
    )

    # Entraînement
    print("Début de l'entraînement du modèle...")
    random_search.fit(X_train, y_train)

    # Meilleur modèle trouvé
    best_model = random_search.best_estimator_
    print(f"Meilleurs hyperparamètres trouvés : {random_search.best_params_}")

    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Évalue les performances du modèle sur les données de test.

    Args:
        model: Le modèle entraîné.
        X_test: Les données de test (features).
        y_test: Les labels de test (target).
    """
    # Faire des prédictions sur les données de test
    y_pred = model.predict(X_test)

    # Afficher le rapport de classification
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Afficher la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.savefig('confusion_matrix.png')  # Sauvegarde la matrice de confusion
    plt.close()

def save_model(model, filepath):
    """
    Sauvegarde le modèle entraîné dans un fichier.

    Args:
        model: Le modèle entraîné.
        filepath (str): Chemin vers le fichier où le modèle sera sauvegardé.
    """
    joblib.dump(model, filepath)
    print(f"Modèle sauvegardé avec succès dans {filepath}.") 

def load_model(filepath):
    """
    Charge un modèle sauvegardé à partir d'un fichier.

    Args:
        filepath (str): Chemin vers le fichier du modèle.

    Returns:
        model: Le modèle chargé.
    """
    model = joblib.load(filepath)
    print(f"Modèle chargé avec succès depuis {filepath}.")
    return model

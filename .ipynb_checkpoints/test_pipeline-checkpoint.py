import unittest
import os
import pandas as pd
from model_pipeline import prepare_data  # Assurez-vous que la fonction est bien importée
from sklearn.model_selection import train_test_split

class TestDataPreparation(unittest.TestCase):
   
    def test_file_exists(self):
        # Assurer que le fichier de données existe pour éviter une erreur
        data_file = 'data_churn.csv'
        
        # Vérifier si le fichier de données existe
        self.assertTrue(os.path.exists(data_file), f"Le fichier {data_file} est introuvable.")
    
    def test_prepare_data(self):
        # Exemple de chemin vers un fichier de données fictif
        data_file = 'data_churn.csv'

        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data(data_file)
        
        # Vérifier que X_train et X_test sont des DataFrame
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(X_test, pd.DataFrame)
        
        # Vérifier que y_train et y_test sont des Series pandas
        self.assertIsInstance(y_train, pd.Series)
        self.assertIsInstance(y_test, pd.Series)
        
        # Vérifier que les tailles de X_train et y_train correspondent
        self.assertEqual(len(X_train), len(y_train), "Les tailles de X_train et y_train ne correspondent pas.")
        
        # Vérifier que les tailles de X_test et y_test correspondent
        self.assertEqual(len(X_test), len(y_test), "Les tailles de X_test et y_test ne correspondent pas.")
        
    def test_no_missing_values(self):
        # Exemple de chemin vers un fichier de données fictif
        data_file = 'data_churn.csv'

        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data(data_file)
        
        # Vérifier qu'il n'y a pas de valeurs manquantes dans les données d'entraînement et de test
        self.assertFalse(X_train.isnull().values.any(), "Des valeurs manquantes existent dans X_train.")
        self.assertFalse(X_test.isnull().values.any(), "Des valeurs manquantes existent dans X_test.")
        self.assertFalse(y_train.isnull().values.any(), "Des valeurs manquantes existent dans y_train.")
        self.assertFalse(y_test.isnull().values.any(), "Des valeurs manquantes existent dans y_test.")
    
    def test_train_test_split(self):
        # Exemple de chemin vers un fichier de données fictif
        data_file = 'data_churn.csv'

        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data(data_file)

        # Vérifier que les données ont été divisées correctement
        self.assertGreater(len(X_train), 0, "X_train ne doit pas être vide.")
        self.assertGreater(len(X_test), 0, "X_test ne doit pas être vide.")
        self.assertGreater(len(y_train), 0, "y_train ne doit pas être vide.")
        self.assertGreater(len(y_test), 0, "y_test ne doit pas être vide.")
        
        # Vérifier que la somme de la taille de X_train et X_test est égale à la taille totale des données
        total_data_size = len(X_train) + len(X_test)
        full_data = pd.read_csv(data_file)
        
    def test_column_names(self):
        # Exemple de chemin vers un fichier de données fictif
        data_file = 'data_churn.csv'

        # Préparer les données
        X_train, X_test, y_train, y_test = prepare_data(data_file)

        # Vérifier que les colonnes de X_train sont des chaînes de caractères ou des variables numériques
        self.assertTrue(all(isinstance(col, str) or pd.api.types.is_numeric_dtype(X_train[col]) for col in X_train.columns), "Certaines colonnes de X_train ne sont pas du bon type.")
        
        # Vérifier que y_train est une série avec des étiquettes uniques (si c'est un problème de classification)
        self.assertTrue(len(y_train.unique()) > 1, "y_train ne contient pas assez de classes uniques.")
    
    # Ajouter d'autres tests si nécessaire

if __name__ == '__main__':
    unittest.main()

# Utiliser une image légère de Python
FROM python:3.12-slim

# Mise à jour des paquets et installation des outils nécessaires
RUN apt-get update && apt-get install -y build-essential

# Définir le répertoire de travail
WORKDIR /app

# Copier tous les fichiers du projet dans le conteneur
COPY . .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port pour FastAPI
EXPOSE 8000  

# Démarrer l'API FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

# Install MLflow separately
RUN pip install mlflow

# Expose MLflow UI port
EXPOSE 5000

# Command to start MLflow server
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/app/mlruns"]

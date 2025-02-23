# Déclaration des variables
SHELL := /bin/bash
PYTHON = python
VENV = venv
REQUIREMENTS = requirements.txt
MAIN_SCRIPT = main.py
DATA_PATH = data_churn.csv
DOCKER_IMAGE = salma6633/salma_labidi_4ds4_mlops
DOCKER_TAG = latest
PORT = 8000

# 1. Configuration de l'environnement
setup:
	@echo "Installation des dépendances dans l'environnement existant..."
	@if [ -d "$(VENV)" ]; then \
		echo "Environnement détecté, installation des dépendances..."; \
		. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQUIREMENTS); \
	else \
		echo "Création de l'environnement virtuel..."; \
		$(PYTHON) -m venv $(VENV) && source $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQUIREMENTS); \
	fi

# 2. Analyse statique du code avec mypy
lint:
	@echo "Vérification statique du code avec mypy..."
	@mypy --ignore-missing-imports --exclude $(VENV) .

format:
	@echo "Formatage du code avec black..."
	@black .

# 3. Préparation des données
data:
	@echo "Préparation des données..."
	@$(PYTHON) $(MAIN_SCRIPT) --step prepare --data $(DATA_PATH)

# 4. Entraînement du modèle
train: data
	@echo "Entraînement du modèle..."
	@$(PYTHON) $(MAIN_SCRIPT) --step train --data $(DATA_PATH)

# 5. Tests unitaires
test:
	@echo "Exécution des tests unitaires avec pytest..."
	@pytest --maxfail=1 --disable-warnings -q
deploy:
	python main.py --run-all --data $(DATA_PATH)  # ✅ Correct (avec double tiret)

# 7. Nettoyage
clean:
	@echo "Nettoyage des fichiers temporaires..."
	@rm -rf __pycache__ *.pkl $(ENV_NAME)

# 8. Démarrage du serveur Jupyter Notebook
.PHONY: notebook
notebook:
	@echo "Démarrage de Jupyter Notebook..."
	@source $(ENV_NAME)/bin/activate && jupyter notebook

# 9. Exécution complète
all: setup lint format data train test deploy

# Démarrage de l'API FastAPI
run-api:
	@echo "Démarrage de l'API FastAPI..."
	@uvicorn app:app --reload --host 0.0.0.0 --port 8000

# Démarrage de l'interface MLflow
mlflow-ui:
	@echo "Démarrage de l'interface MLflow..."
	@mlflow ui --host 0.0.0.0 --port 5000

# 10. Construire l'image Docker
build-docker:
	@echo "Construction de l'image Docker..."
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

# 11. Pousser l'image sur Docker Hub
push-docker:
	@echo "Push de l'image sur Docker Hub..."
	@docker push $(DOCKER_IMAGE):$(DOCKER_TAG)

# 12. Lancer le conteneur Docker
run-docker:
	@echo "Démarrage du conteneur Docker..."
	@docker run -d -p $(PORT):$(PORT) $(DOCKER_IMAGE):$(DOCKER_TAG)

# 13. Supprimer les conteneurs et images inutilisés
clean-docker:
	@echo "Nettoyage des images et conteneurs Docker..."
	@docker system prune -f

# Déclaration des variables
SHELL := /bin/bash
PYTHON = python3
VENV = venv
REQUIREMENTS = requirements.txt
MAIN_SCRIPT = main.py
DATA_PATH = data_churn.csv
DOCKER_IMAGE = salma6633/salma_labidi_4ds4_mlops
DOCKER_TAG = latest
PORT = 8000
MLFLOW_PORT = 5000
PROMETHEUS_PORT = 9090
GRAFANA_PORT = 3000
ELASTICSEARCH_PORT = 9200
KIBANA_PORT = 5601

# Couleurs ANSI pour l'affichage
GREEN = \033[92m
BLUE = \033[94m
RED = \033[91m
RESET = \033[0m

# 1. Configuration de l'environnement
setup:
	@echo -e "$(BLUE)Installation des dépendances dans l'environnement existant...$(RESET)"
	@if [ -d "$(VENV)" ]; then \
		echo -e "$(BLUE)Environnement détecté, installation des dépendances...$(RESET)"; \
		. $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQUIREMENTS); \
	else \
		echo -e "$(BLUE)Création de l'environnement virtuel...$(RESET)"; \
		$(PYTHON) -m venv $(VENV) && . $(VENV)/bin/activate && pip install --upgrade pip && pip install -r $(REQUIREMENTS); \
	fi
	@echo -e "$(GREEN)Environnement configuré avec succès.$(RESET)"

# 2. Analyse statique du code avec mypy
lint:
	@echo -e "$(BLUE)Vérification statique du code avec mypy...$(RESET)"
	@mypy --ignore-missing-imports --exclude venv .
	@echo -e "$(GREEN)Vérification terminée.$(RESET)"

# 3. Formatage du code avec black
format:
	@echo -e "$(BLUE)Formatage du code avec black...$(RESET)"
	@black .
	@echo -e "$(GREEN)Formatage terminé.$(RESET)"

# 4. Préparation des données
data:
	@echo -e "$(BLUE)Préparation des données...$(RESET)"
	@$(PYTHON) $(MAIN_SCRIPT) --step prepare --data $(DATA_PATH)
	@echo -e "$(GREEN)Préparation des données terminée.$(RESET)"

# 5. Entraînement du modèle
train: data
	@echo -e "$(BLUE)Entraînement du modèle...$(RESET)"
	@$(PYTHON) $(MAIN_SCRIPT) --step train --data $(DATA_PATH)
	@echo -e "$(GREEN)Entraînement terminé.$(RESET)"

# 6. Tests unitaires
test:
	@echo -e "$(BLUE)Exécution des tests unitaires avec pytest...$(RESET)"
	@pytest --maxfail=1 --disable-warnings -q
	@echo -e "$(GREEN)Tests terminés.$(RESET)"

# 7. Déploiement complet du pipeline
deploy:
	@echo -e "$(BLUE)Déploiement complet du pipeline...$(RESET)"
	@$(PYTHON) $(MAIN_SCRIPT) --run-all --data $(DATA_PATH)
	@echo -e "$(GREEN)Déploiement terminé.$(RESET)"

# 8. Nettoyage
clean:
	@echo -e "$(BLUE)Nettoyage des fichiers temporaires...$(RESET)"
	@rm -rf __pycache__ *.pkl $(VENV)
	@echo -e "$(GREEN)Nettoyage terminé.$(RESET)"

# 9. Démarrage du serveur Jupyter Notebook
notebook:
	@echo -e "$(BLUE)Démarrage de Jupyter Notebook...$(RESET)"
	@. $(VENV)/bin/activate && jupyter notebook
	@echo -e "$(GREEN)Jupyter Notebook arrêté.$(RESET)"

# 10. Exécution complète (setup + lint + format + data + train + test + deploy)
all: setup lint format data train test deploy
	@echo -e "$(GREEN)Toutes les étapes ont été exécutées avec succès.$(RESET)"

# 11. Démarrage de l'API FastAPI
run-api:
	@echo -e "$(BLUE)Démarrage de l'API FastAPI...$(RESET)"
	@. $(VENV)/bin/activate && uvicorn app:app --reload --host 0.0.0.0 --port $(PORT)
	@echo -e "$(GREEN)API FastAPI arrêtée.$(RESET)"

# 12. Démarrage de l'interface MLflow
mlflow-ui:
	@echo -e "$(BLUE)Démarrage de l'interface MLflow...$(RESET)"
	@. $(VENV)/bin/activate && mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT)
	@echo -e "$(GREEN)Interface MLflow arrêtée.$(RESET)"

# 13. Construire l'image Docker
build-docker:
	@echo -e "$(BLUE)Construction de l'image Docker...$(RESET)"
	@docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .
	@echo -e "$(GREEN)Image Docker construite avec succès.$(RESET)"

# 14. Pousser l'image sur Docker Hub
push-docker:
	@echo -e "$(BLUE)Push de l'image sur Docker Hub...$(RESET)"
	@docker push $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo -e "$(GREEN)Image Docker poussée avec succès.$(RESET)"

# 15. Lancer le conteneur Docker
run-docker:
	@echo -e "$(BLUE)Démarrage du conteneur Docker...$(RESET)"
	@docker run -d -p $(PORT):$(PORT) $(DOCKER_IMAGE):$(DOCKER_TAG)
	@echo -e "$(GREEN)Conteneur Docker démarré.$(RESET)"

# 16. Supprimer les conteneurs et images inutilisés
clean-docker:
	@echo -e "$(BLUE)Nettoyage des images et conteneurs Docker...$(RESET)"
	@docker system prune -f
	@echo -e "$(GREEN)Nettoyage Docker terminé.$(RESET)"

# 17. Monitoring avec Docker Compose
monitoring:
	@echo -e "$(BLUE)Démarrage de la stack de monitoring...$(RESET)"
	@docker-compose up -d
	@mlflow ui --host 0.0.0.0 --port $(MLFLOW_PORT) &
	@echo -e "$(GREEN)Monitoring en cours.$(RESET)"

stop-monitoring:
	@echo -e "$(BLUE)Arrêt de la stack de monitoring...$(RESET)"
	@docker-compose down
	@echo -e "$(GREEN)Monitoring arrêté.$(RESET)"

# 18. Afficher l'aide
help:
	@echo -e "$(BLUE)Commandes disponibles :$(RESET)"
	@echo "  setup         : Configurer l'environnement virtuel et installer les dépendances."
	@echo "  lint          : Vérifier le code avec mypy."
	@echo "  format        : Formater le code avec black."
	@echo "  data          : Préparer les données."
	@echo "  train         : Entraîner le modèle."
	@echo "  test          : Exécuter les tests unitaires."
	@echo "  deploy        : Exécuter le pipeline complet."
	@echo "  clean         : Nettoyer les fichiers temporaires."
	@echo "  notebook      : Démarrer Jupyter Notebook."
	@echo "  run-api       : Démarrer l'API FastAPI."
	@echo "  mlflow-ui     : Démarrer l'interface MLflow."
	@echo "  build-docker  : Construire l'image Docker."
	@echo "  push-docker   : Pousser l'image Docker sur Docker Hub."
	@echo "  run-docker    : Démarrer le conteneur Docker."
	@echo "  clean-docker  : Nettoyer les images et conteneurs Docker inutilisés."

.PHONY: setup lint format data train test deploy clean notebook all run-api mlflow-ui build-docker push-docker run-docker clean-docker help

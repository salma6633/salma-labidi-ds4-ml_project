def run_pipeline(data_path):
    """
    Exécute le pipeline complet et retourne les métriques d'évaluation.
    """
    logger.info(Fore.CYAN + "=" * 80 + Style.RESET_ALL)
    logger.info(Fore.CYAN + "🚀 Démarrage du pipeline complet..." + Style.RESET_ALL)
    logger.info(Fore.CYAN + "=" * 80 + Style.RESET_ALL)

    # Étape 1 : Préparation des données
    logger.info(Fore.BLUE + "\n=== Étape 1 : Préparation des données ===" + Style.RESET_ALL)
    logger.info("Préparation des données...")
    X_train, X_test, y_train, y_test = prepare_data(data_path)
    logger.info(Fore.GREEN + "✅ Données préparées avec succès." + Style.RESET_ALL)

    # Étape 2 : Entraînement du modèle
    logger.info(Fore.BLUE + "\n=== Étape 2 : Entraînement du modèle ===" + Style.RESET_ALL)
    logger.info("Entraînement du modèle...")
    model = train_model(X_train, y_train)
    logger.info(Fore.GREEN + "✅ Modèle entraîné avec succès." + Style.RESET_ALL)

    # Étape 3 : Évaluation du modèle
    logger.info(Fore.BLUE + "\n=== Étape 3 : Évaluation du modèle ===" + Style.RESET_ALL)
    logger.info("Évaluation du modèle...")
    metrics = evaluate_model(model, X_test, y_test)
    logger.info(Fore.YELLOW + "📊 Métriques d'évaluation :" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + f"+-------------------+-------------------+" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + f"| Métrique          | Valeur            |" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + f"+-------------------+-------------------+" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + f"| Accuracy          | {metrics['accuracy']:.4f}            |" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + f"| ROC AUC Score     | {metrics['roc_auc_score']:.4f}            |" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + f"+-------------------+-------------------+" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + "📝 Rapport de classification :" + Style.RESET_ALL)
    logger.info(Fore.YELLOW + metrics['classification_report'] + Style.RESET_ALL)
    logger.info(Fore.GREEN + "✅ Évaluation terminée." + Style.RESET_ALL)

    # Étape 4 : Sauvegarde du modèle
    logger.info(Fore.BLUE + "\n=== Étape 4 : Sauvegarde du modèle ===" + Style.RESET_ALL)
    logger.info("Sauvegarde du modèle...")
    save_model(model, "customer_churn_model.pkl")
    logger.info(Fore.GREEN + "✅ Modèle sauvegardé avec succès." + Style.RESET_ALL)

    # Étape 5 : Enregistrement du modèle dans MLflow
    logger.info(Fore.BLUE + "\n=== Étape 5 : Enregistrement du modèle dans MLflow ===" + Style.RESET_ALL)
    logger.info("Enregistrement du modèle dans MLflow...")
    log_model_to_mlflow(model, X_train, X_test)
    logger.info(Fore.GREEN + "✅ Modèle enregistré dans MLflow." + Style.RESET_ALL)

    # Étape 6 : Envoi des métriques à Elasticsearch
    logger.info(Fore.BLUE + "\n=== Étape 6 : Envoi des métriques à Elasticsearch ===" + Style.RESET_ALL)
    logger.info("Envoi des métriques à Elasticsearch...")
    log_metrics_to_es(metrics)
    logger.info(Fore.GREEN + "✅ Métriques envoyées à Elasticsearch." + Style.RESET_ALL)

    logger.info(Fore.CYAN + "=" * 80 + Style.RESET_ALL)
    logger.info(Fore.CYAN + "🎉 Pipeline terminé avec succès !" + Style.RESET_ALL)
    logger.info(Fore.CYAN + "=" * 80 + Style.RESET_ALL)

    return metrics
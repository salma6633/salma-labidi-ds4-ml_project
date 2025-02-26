from elasticsearch import Elasticsearch
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Connexion à Elasticsearch
es = Elasticsearch([{"scheme": "http", "host": "localhost", "port": 9200}])
if es.ping():
    logger.info("\033[92mConnexion à Elasticsearch réussie!\033[0m")
else:
    logger.error("\033[91mLa connexion à Elasticsearch a échoué!\033[0m")

# Vérifier si l'index existe et le créer si nécessaire
index_name = "mlflow-metrics"
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name)
    logger.info(f"\033[94mL'index '{index_name}' a été créé.\033[0m")

def log_metrics_to_es(metrics):
    """
    Envoie les métriques à Elasticsearch.
    """
    try:
        if metrics:
            metrics["timestamp"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            es.index(index=index_name, body=metrics)
            logger.info("\033[92mMétriques envoyées à Elasticsearch.\033[0m")
        else:
            logger.warning("\033[91mAucune métrique à envoyer.\033[0m")
    except Exception as e:
        logger.error(f"\033[91mErreur lors de l'envoi des métriques vers Elasticsearch : {e}\033[0m")

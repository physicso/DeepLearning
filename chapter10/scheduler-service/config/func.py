from app.job.job_processing import dogcat_app
from config import var as v

# Define mappings between Kafka topics and processing logic.
APP_MAPPING = {
    v.TOPIC_DOGCAT_REQUEST: dogcat_app,
}

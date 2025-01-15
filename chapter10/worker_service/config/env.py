# coding: UTF-8
import os

# Kafka configuration.
KAFKA_HOST = os.environ.get('KAFKA_HOST', '127.0.0.1')
KAFKA_PORT = int(os.environ.get('KAFKA_PORT', 9092))
KAFKA_SERVICE = ['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)]

# TensorFlow Serving configuration.
TF_SERVING_HOST = os.environ.get('TF_SERVING_HOST', '127.0.0.1')
TF_SERVING_PORT = int(os.environ.get('TF_SERVING_PORT', 9000))

# Celery configuration.
USE_CACHE = (os.environ.get('USE_CACHE', 'False') == 'True')
# The Worker sends an acknowledgement to the Broker after the task has been executed, telling the queue that the task has been processed, rather than sending it after it has been received and before it is executed.
CELERY_ACKS_LATE = True
# Customize your own log handler.
CELERYD_HIJACK_ROOT_LOGGER = False
CELERY_DEFAULT_QUEUE = 'default'
# Set priority.
BROKER_TRANSPORT_OPTIONS['priority_steps'] = list(range(9))

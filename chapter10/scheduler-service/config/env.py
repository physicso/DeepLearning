# coding: UTF-8
import os

# Kafka configuration.
KAFKA_HOST = os.environ.get('KAFKA_HOST', '127.0.0.1')
KAFKA_PORT = int(os.environ.get('KAFKA_PORT', 9092))
KAFKA_SERVICE = ['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)]

# Celery configuration.
# The Worker sends an acknowledgement to the Broker after the task has been executed, telling the queue that the task has been processed, rather than sending it after it has been received and before it is executed.
CELERY_ACKS_LATE = True
# Customize your own log handler.
CELERYD_HIJACK_ROOT_LOGGER = False
CELERY_DEFAULT_QUEUE = 'default'
# Set priority.
BROKER_TRANSPORT_OPTIONS['priority_steps'] = list(range(9))

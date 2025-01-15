# coding: UTF-8
import os

# Kafka configuration.
KAFKA_HOST = os.environ.get('KAFKA_HOST', '127.0.0.1')
KAFKA_PORT = int(os.environ.get('KAFKA_PORT', 9092))
KAFKA_SERVICE = ['{}:{}'.format(KAFKA_HOST, KAFKA_PORT)]

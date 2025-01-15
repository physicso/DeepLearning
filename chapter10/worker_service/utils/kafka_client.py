from __future__ import absolute_import
import json
from kafka import KafkaProducer
from kafka import KafkaConsumer
from kafka.errors import KafkaError
from kiel import clients
from config import env


class Producer():
    def __init__(self, topic):
        self.topic = topic
        self.producer = KafkaProducer(
            bootstrap_servers=env.KAFKA_SERVICE,
            compression_type='gzip',
            retries=5,
            buffer_memory=67108864,  # 64M
            max_request_size=33554432  # 32M
        )

    def produce(self, msg, topic=None):
        if topic is None or topic == '':
            topic = self.topic
        try:
            parmas_msg = json.dumps(msg)
            self.producer.send(topic, parmas_msg.encode('utf-8'))
            self.producer.flush()
        except KafkaError as e:
            print(e)


class Consumer():
    def __init__(self, topic, groupid):
        self.topic = topic
        self.groupid = groupid
        self.consumer = KafkaConsumer(
            self.topic,
            group_id=self.groupid,
            bootstrap_servers=env.KAFKA_SERVICE
        )

    def consume(self):
        try:
            for msg in self.consumer:
                yield msg
        except KeyboardInterrupt as e:
            print(e)


def get_producer(topic):
    return Producer(topic)


def get_consumer():
    # Return Consumer(topic, group_id).
    return KafkaConsumer(
        bootstrap_servers=env.KAFKA_SERVICE
    )


def get_consumer_async():
    return clients.SingleConsumer(
        brokers=env.KAFKA_SERVICE,
        deserializer=None,
        max_wait_time=1000,  # in milliseconds
        min_bytes=1,
        max_bytes=(4 * 1024 * 1024),
    )

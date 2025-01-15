# -*- coding: utf-8 -*-
from __future__ import absolute_import
import json
import multiprocessing
from kafka import KafkaConsumer
from config import env, var, func


# The scheduler is essentially a message listener, implemented through multiple threads.
class Consumer(multiprocessing.Process):

    def __init__(self, topic):
        multiprocessing.Process.__init__(self)
        self.stop_event = multiprocessing.Event()
        self.consumer = KafkaConsumer(
            bootstrap_servers=env.KAFKA_SERVICE,
            consumer_timeout_ms=10000,
            fetch_max_bytes=52428800)
        self.consumer.subscribe(topic)
        # Find the processing logic for different message topics based on the mapping.
        if topic in func.APP_MAPPING.keys():
            # Invokes the processing logic for the corresponding topic.
            self.exec_fun = func.APP_MAPPING[topic]
        else:
            pass

    def stop(self):
        self.stop_event.set()

    def run(self):
        while not self.stop_event.is_set():
            for message in self.consumer:
                msg = json.loads(message.value)
                # Process the message
                self.exec_fun(msg)
        else:
            self.consumer.close()


def start_service():
    print('DeepGo started successfully. Listening...')
    # Different tasks can be defined here.
    tasks = [
        Consumer(topic=var.TOPIC_DOGCAT_REQUEST),
    ]

    for t in tasks:
        t.start()

    for t in tasks:
        t.join()

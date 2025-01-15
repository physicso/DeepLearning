#!/usr/bin/env python
import os
import time
import multiprocessing
from kafka import KafkaConsumer
from config import env


# Used to receive log information and save it to the log folder.
class Consumer(multiprocessing.Process):

    def __init__(self, topics):
        multiprocessing.Process.__init__(self)
        self.stop_event = multiprocessing.Event()
        self.topics = topics
        self.log_path = 'logs/'

    def stop(self):
        self.stop_event.set()

    def run(self):
        consumer = KafkaConsumer(
            bootstrap_servers=env.KAFKA_SERVICE,
            consumer_timeout_ms=10000,
            fetch_max_bytes=52428800)
        consumer.subscribe(self.topics)
        if not os.path.exists(self.log_path):
            os.mkdir(self.log_path)

        date = time.strftime("%Y-%m-%d")
        f_all = open(self.log_path + date + '.log', 'a+')
        f_job = open(self.log_path + 'job' + '.log', 'a+')

        while not self.stop_event.is_set():
            for message in consumer:
                n_date = time.strftime("%Y-%m-%d")
                if date != n_date:
                    date = n_date
                    f_all.close()
                    f_all = open(self.log_path + date + '.log', 'a+')
                f_job.write(message.value.decode() + '\n')
                f_job.flush()
                f_all.write(message.value.decode() + '\n')
                f_all.flush()
                f_job.flush()
                if self.stop_event.is_set():
                    f_all.close()
                    f_job.close()
                    break

        consumer.close()


def initial():
    log_dirs = [
        'logs',
    ]

    for path in log_dirs:
        if not os.path.exists(path):
            os.makedirs(path)


def main():
    time.sleep(20)
    initial()
    tasks = [
        Consumer([env.TOPIC_LOG_WORKER]),
    ]
    for t in tasks:
        t.start()
    for t in tasks:
        t.join()


if __name__ == "__main__":
    main()

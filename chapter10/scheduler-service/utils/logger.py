from kafka.client import SimpleClient
from kafka.producer import SimpleProducer, KeyedProducer
import logging
from celery.utils.log import get_task_logger
from config import env, var as v


class KafkaLoggingHandler(logging.Handler):

    def __init__(self, hosts_list, topic, **kwargs):
        logging.Handler.__init__(self)

        self.kafka_client = SimpleClient(hosts_list)
        self.key = kwargs.get("key", None)
        self.kafka_topic_name = topic

        if not self.key:
            self.producer = SimpleProducer(self.kafka_client, **kwargs)
        else:
            self.producer = KeyedProducer(self.kafka_client, **kwargs)

    def emit(self, record):
        # Note that Kafka's own logs are removed here.
        if record.name == 'kafka':
            return
        try:
            # Use the default format
            msg = self.format(record)
            if isinstance(msg, str):
                msg = msg.encode("utf-8")

            # Send a message
            if not self.key:
                self.producer.send_messages(self.kafka_topic_name, msg)
            else:
                self.producer.send_messages(
                    self.kafka_topic_name, self.key, msg)
        except Exception as e:
            print('Failed to send log: {} to kafka, error info: {}'.format(record, e))

    def close(self):
        if self.producer is not None:
            self.producer.stop()
        logging.Handler.close(self)


def get_kafka_logger(name, log='logging'):
    if log == 'celery':
        logger = get_task_logger(name)
    else:
        logger = logging.getLogger(name)

    logger.setLevel(logging.INFO)
    kafka_handler = KafkaLoggingHandler(env.KAFKA_SERVICE, v.TOPIC_LOG_SCHEDULER)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    kafka_handler.setFormatter(formatter)
    logger.addHandler(kafka_handler)
    return logger

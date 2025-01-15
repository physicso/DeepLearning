# -*- coding: utf-8 -*-
from app.celery_app import celery_app
from celery import Task
from utils.kafka_client import get_producer
from worker.dogcat import DogCat
from config import env, var


# Basic task class.
class BaseTask(Task):

    @staticmethod
    def produce_msg(msg, topic=None):
        BaseTask.producer = get_producer(topic)
        BaseTask.producer.produce(msg, topic=topic)


# The following is a fixed notation used to define Celery task processing logic.
@celery_app.task(base=BaseTask, max_retries=3)
def dogcat(msg):
    dogcat_app = DogCat(msg)
    dogcat_result = dogcat_app.get_tasks()
    BaseTask.produce_msg(dogcat_result, topic=var.TOPIC_DOGCAT_RESPONSE)

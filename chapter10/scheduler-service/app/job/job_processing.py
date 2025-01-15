# -*- coding: utf-8 -*-
from __future__ import absolute_import
import time
import json
from app.celery_app import celery_app
from utils.logger import get_kafka_logger

logger = get_kafka_logger('deepgo.scheduler.' + __name__)


# Define the processing logic for the Cat vs. Dog task.
def dogcat_app(msg, priority=6):
    _logger(msg, 'DogCat')
    job_id = str(msg.get('job_id', ''))
    salt = str(int(time.time()))
    task_name = '{}_{}'.format(job_id, salt)
    # Here's a permanent script that you use to pick up Celery work node.
    dogcat = celery_app.signature(
        'worker.tasks.dogcat',
        kwargs={'msg': json.dumps(msg)},
        app=celery_app,
        priority=9 if priority == 6 else 3,
        task_id='deepgo_{}_task_dogcat'.format(task_name)
    )
    dogcat.apply_async()
    time.sleep(0.3)


# Log recording.
def _logger(msg, name):
    try:
        logger.info('[Receive job][{}] submit message: {}'.format(str(msg), name))
    except Exception as e:
        print('Logging exception: {}'.format(e))
        logger.info('Logging exception: {}'.format(e))

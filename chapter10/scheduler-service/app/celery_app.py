from __future__ import absolute_import
from celery import Celery

celery_app = Celery('worker.tasks')
celery_app.config_from_object('config.env')

# Start Celery.
if __name__ == '__main__':
    celery_app.start()

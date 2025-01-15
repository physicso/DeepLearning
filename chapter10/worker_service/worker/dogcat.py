import json
import numpy as np
from PIL import Image
from utils.template import Template
from utils.logger import get_kafka_logger
from utils.log_info import LogInfo
from config import env
from model.tfserving import TFServing

logger = get_kafka_logger('deepgo.worker.' + __name__, log='celery')
log_worker = LogInfo('DogCat', logger)


# The core predictive logic of the Cat vs. Dog task.
class DogCat(object):
    json_validity = True

    def __init__(self, msg):
        super(DogCat, self).__init__()
        try:
            self.dogcat_json_dict = json.loads(msg)
        except Exception as e:
            logger.exception(e)
            self.json_validity = False
        else:
            self.job_id = self.dogcat_json_dict['job_id']
            self.image = self.dogcat_json_dict['image']

    # Here is the processing logic.
    def get_tasks(self):
        try:
            log_worker.start_job(self.job_id,
                                 json.dumps(self.dogcat_json_dict))
        except Exception as e:
            logger.exception(e)
            raise
        if not self.json_validity:
            log_worker.finish_job(self.job_id, False,
                                  json.dumps(self._empty_json('Json Format Error.')))
            return self._empty_json('Json Format Error.')
        prediction = None
        print(json.dumps(self.dogcat_json_dict))
        try:
            # Call TensorFlow Serving to make predictions.
            tf_client = TFServing(env.TF_SERVING_HOST, env.TF_SERVING_PORT)
            img = Image.open(self.image).resize((224, 224))
            prediction = tf_client.predict_new(np.array(img), 'dogcat')
        except Exception as e:
            logger.exception(e)
        template = Template(self.job_id, True)
        result_json = template.dogcat_template(prediction)
        log_worker.finish_job(self.job_id, True, result_json)
        return result_json

    def _empty_json(self, e):
        empty_template = Template(self.job_id, False)
        return empty_template.dogcat_template(None, e)

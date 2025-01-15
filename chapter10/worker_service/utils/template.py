import json


# Generic return message template.
class Template(object):
    def __init__(self, job_id, success=True):
        self.job_id = job_id
        self.success = success

    def dogcat_template(self, prediction, e=''):
        json_dict = {
            'job_id': self.job_id,
            'success': self.success,
            'prediction': prediction,
            'error_message': str(e)
        }
        return json.dumps(json_dict)

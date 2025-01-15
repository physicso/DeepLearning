# Record the task execution process.
class LogInfo(object):
    def __init__(self, name, logger):
        super(LogInfo, self).__init__()
        self.name = name
        self.logger = logger

    def start_job(self, job_id, msg):
        self.logger.info('[# Start Job][{}] Job_ID: {}, Received Msg: {}'.format(self.name, job_id, str(msg)))

    def progress(self, job_id, msg):
        self.logger.info('[Job Progress][{}] Job_ID: {}, '.format(self.name, job_id, str(msg)))

    def finish_job(self, job_id, success, msg):
        self.logger.info(
            '[# Finish Task][{}] Job_ID: {}, Success: {}, '
            'Return Msg: {}'.format(self.name, job_id, success, msg)
        )

import grpc
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import tensorflow as tf
from config import var


# Interact with TensorFlow Serving.
class TFServing(object):

    def __init__(self, host, port):
        super(TFServing, self).__init__()

        channel = grpc.insecure_channel('%s:%d' % (host, port), options=(('grpc.enable_http_proxy', 0),))
        self._stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    def predict(self, image_input, model_name):
        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.inputs[var.INPUT_KEY].CopyFrom(
            tf.contrib.util.make_tensor_proto(image_input, shape=[1, ] + list(image_input.shape), dtype=tf.float32))
        try:
            result = self._stub.Predict(request, 5.0)
            prediction = result.outputs[var.PREDICT_KEY].float_val
        except Exception as e:
            raise e
        else:
            return prediction

import os
import numpy as np
import json
import logging

from pipeline_model import TensorFlowServingModel
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['predict']


_labels= {'model_runtime': 'tfserving',
          'model_type': 'tensorflow',
          'model_name': 'inception',
          'model_tag': 'cpu',
          'model_chip': 'cpu',
         }


def _initialize_upon_import() -> TensorFlowServingModel:
    ''' Initialize / Restore Model Object.
    '''
    return TensorFlowServingModel(host='localhost',
                                  port=9000,
                                  model_name='inception',
                                  model_signature_name='predict_images',
                                  timeout_seconds=10.0)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()

# https://www.tensorflow.org/serving/serving_inception
@log(labels=_labels, logger=_logger)
def predict(request: bytes) -> bytes:
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="predict"):
#        predictions = _model.predict(transformed_request)

        channel = implementations.insecure_channel(host, int(port))
        stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
        # Send request
        with open(image_file_path, 'rb') as f:
            # See prediction_service.proto for gRPC request/response details.
            data = f.read()
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'inception'
            request.model_spec.signature_name = 'predict_images'
            #request_dict = {'images': data}
            request.inputs['images'].CopyFrom(tf.make_tensor_proto(data, shape=[1]))
            result = stub.Predict(request, 10.0)  # 10 secs timeout
       print(result)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(predictions)

    return transformed_response


def _transform_request(request: bytes) -> dict:
    # Convert from bytes to tf.tensor, np.array, etc.
    pass


def _transform_response(response: dict) -> json:
    # Convert from tf.tensor, np.array, etc. to bytes
    pass

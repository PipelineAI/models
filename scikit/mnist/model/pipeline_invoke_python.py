import os
import numpy as np
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
from sklearn.externals import joblib
import json
import logging

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
           'name': 'mnist',
           'tag': 'v1',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'scikit',
          }


def _initialize_upon_import():
    model_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

    return joblib.load(model_pkl_path).best_estimator_


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""
    transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.predict(transformed_request)

    return _transform_response(response)


def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = np.array(request_json['image'], dtype=np.uint8)
    request_np = request_np.reshape(1, 784)
    return request_np


def _transform_response(response):
    response_np = response.data.tolist()
    response_json = json.dumps({"classes": response_np, "probabilities": [[]]})
    return response_json


if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

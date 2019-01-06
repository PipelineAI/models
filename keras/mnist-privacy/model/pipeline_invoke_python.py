import os
import numpy as np
import json
import logging

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

import tensorflow as tf
from tensorflow.contrib import predictor

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
           'resource_subtype': 'keras',
          }


def _initialize_upon_import():
    """Initialize / Restore Model Object."""
    saved_model_path = './pipeline_tfserving/0'
    return predictor.from_saved_model(saved_model_path)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = (np.array(request_json['image'], dtype=np.float32) / 255.0).reshape(1, 28, 28)
    return {"image": request_np}


def _transform_response(response):
    print(response)
    return json.dumps({"classes": response['classes'].tolist(),
                       "probabilities": response['probabilities'].tolist(),
                      })


if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

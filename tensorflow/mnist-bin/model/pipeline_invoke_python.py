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
           'model_name': 'mnist',
           'model_tag': 'raw',
           'model_type': 'tensorflow',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    saved_model_path = './pipeline_tfserving/0'
    return predictor.from_saved_model(saved_model_path)


_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request):
    request_image_tensor = tf.image.decode_png(request, channels=1, dtype=tf.uint8, name=None)

    request_image_tensor_resized = tf.image.resize_images(request_image_tensor, size=[28,28])

    sess = tf.Session()
    with sess.as_default():
        request_np = request_image_tensor_resized.eval()

    return {"image": request_np.reshape(1, 28, 28)}


def _transform_response(response):
    return json.dumps({"classes": response['classes'].tolist(), 
                       "probabilities": response['probabilities'].tolist(),
                      })


# Note:  This is a faux test
if __name__ == '__main__':
    with open('../input/predict/test_request.png', 'rb') as fb:
        request_bytes = fb.read()
        response_json = invoke(request_bytes)
        print(response_json)

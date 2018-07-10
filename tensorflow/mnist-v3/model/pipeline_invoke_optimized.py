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
           'model_tag': 'v3',
           'model_type': 'tensorflow',
           'model_runtime': 'tfserving',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    ''' Initialize / Restore Model Object.
    '''
    # Load TFLite model and allocate tensors.
    interpreter = tf.contrib.lite.Interpreter(model_path='./pipeline_tfserving/0/optimized/optimized_model.tflite') 
    interpreter.allocate_tensors()

# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()

@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    # Test model on random input data.
    #input_shape = input_details[0]['shape']
    #input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)
    #print('Input: %s' % input_data)
    #interpreter.set_tensor(input_details[0]['index'], input_data)

    with monitor(labels=_labels, name="invoke"):
        input_details = _model.get_input_details()

        response = _model.invoke(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0).reshape(1, 28, 28)
    image_tensor = tf.make_tensor_proto(request_np, dtype=tf.float32)
    return {"image": image_tensor}


def _transform_response(response):
    return json.dumps({"classes": response['classes'].tolist(), 
                       "probabilities": response['probabilities'].tolist(),
                      })

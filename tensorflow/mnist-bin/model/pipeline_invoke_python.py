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
           'model_tag': 'bin',
           'model_type': 'tensorflow',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    ''' Initialize / Restore Model Object.
    '''
    saved_model_path = './pipeline_tfserving/0'
    return predictor.from_saved_model(saved_model_path)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()

# request == byte array
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

# request == byte array
def _transform_request(request):
    # TODO:  Convert this request byte array into a numpy matrix of 28x28 that represents the 
    #        grayscale representation of the png  
    # TODO:  Google how to convert a raw bytes array (png) into a numpy array

    # TODO:  This is the old code, but gives you an idea of what we were doing before to convert json => np
    # request_np = ((255 - np.array(request_json, dtype=np.uint8)) / 255.0).reshape(1, 28, 28)

    # Note:  Don't change this!!
    #        This is what needs to be passed back to the invoke function to make the prediction in TensorFlow.
    return {"image": request_np}

# TODO:  Don't worry about this.  We don't need to change it.
# response = dict of 'classes': (0..9) and 'probabilities' (for each digit in 'classes', there is a corresponding probabilities)
def _transform_response(response):
    # Note:  Don't change this!!
    #        This is what needs to be passed back to the invoke function to return to the caller. 
    return json.dumps({"classes": response['classes'].tolist(), 
                       "probabilities": response['probabilities'].tolist(),
                      })

# TODO:  THIS IS A MINI TEST!!
if __name__ == '__main__':
    with open('../input/predict/test_request.png', 'rb') as fb:
        request_bytes = fb.read()
        print(request_bytes)

        response_bytes = invoke(request_bytes)
        print(response_bytes)

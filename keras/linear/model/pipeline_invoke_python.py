import os
os.environ['KERAS_BACKEND'] = 'theano'
os.environ['THEANO_FLAGS'] = 'floatX=float32,device=cpu'

import pandas as pd
import numpy as np
import json
import logging

from keras_theano_model import KerasTheanoModel
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'name': 'linear',
           'tag': 'v1',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'keras',
          }

def _initialize_upon_import(model_state_path):
    """ Initialize / Restore Model Object.
    """
    return KerasTheanoModel(model_state_path)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state/keras_theano_linear_model_state.h5'))


@monitor(labels=_labels, name="transform_request")
def _json_to_numpy(request):
    request_str = request.decode('utf-8')
    request_str = request_str.strip().replace('\n', ',')
    # surround the json with '[' ']' to prepare for conversion
    request_str = '[%s]' % request_str
    request_json = json.loads(request_str)
    request_transformed = ([json_line['ppt'] for json_line in request_json])
    return np.array(request_transformed)


@monitor(labels=_labels, name="transform_response")
def _numpy_to_json(response):
    return json.dumps(response.tolist())


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""
    transformed_request = _json_to_numpy(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.predict(transformed_request)

    return _numpy_to_json(predictions)

import os
import numpy as np
import json
import cloudpickle as pickle
import logging

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

# The public objects from this module, see:
#    https://docs.python.org/3/tutorial/modules.html#importing-from-a-package

__all__ = ['invoke']


_labels = {
           'model_name': 'zscore',
           'model_tag': 'v1',
           'model_type': 'python',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    model_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

    # Load pickled model from model directory
    with open(model_pkl_path, 'rb') as fh:
        restored_model = pickle.load(fh)

    return restored_model


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''
    transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.predict(transformed_request)

    return _transform_response(response)


def _invoke(inputs):
    cat_affinity_score = sum([ d['weight'] * d['user_score'] for d in inputs if 'cat' in d['tags'] ])
    dog_affinity_score = sum([ d['weight'] * d['user_score'] for d in inputs if 'dog' in d['tags'] ])

    # create normalized z score for compare/classify
    cat_zscore = (cat_affinity_score - _model['cat_mean'])/_model['cat_stdv']
    dog_zscore = (dog_affinity_score - _model['dog_mean'])/_model['dog_stdv']

    # classify
    if abs(cat_zscore) > abs(dog_zscore):
        if cat_zscore >= 0:
            category = 'cat_lover'
        else:
            category = 'cat_hater'
    else:
        if dog_zscore >= 0:
            category = 'dog_lover'
        else:
            category = 'dog_hater'

    response = {
        'category': category,
        'cat_affinity_score': cat_affinity_score,
        'dog_affinity_score': dog_affinity_score,
        'cat_zscore': cat_zscore,
        'cat_zscore': dog_zscore
    }

    return response


@monitor(labels=_labels, name="transform_request")
def _transform_request(request):
    request_str = request.decode('utf-8')
    request_str = request_str.strip().replace('\n', ',')
    request_dict = json.loads(request_str)
    return request_dict


@monitor(labels=_labels, name="transform_response")
def _transform_response(response):
    response_json = json.dumps(response)
    return response_json

import os
import numpy as np
import json
import collections
import logging
from pipeline_model import TensorFlowServingModel
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
from io import StringIO 
import tensorflow as tf

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'model_name': 'linear',
           'model_tag': 'v1',
           'model_type': 'tensorflow',
           'model_runtime': 'tfserving',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    ''' Initialize / Restore Model Object.
    '''

    return TensorFlowServingModel(host='localhost',
                                  port=9000,
                                  model_name=os.environ['PIPELINE_MODEL_NAME'],
                                  model_signature_name=None,
                                  timeout_seconds=10.0)


_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''
    transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.predict(transformed_request)

    return _transform_response(response)


@monitor(labels=_labels, name="transform_request")
def _transform_request(request):
    request_str = StringIO(request.decode('utf-8'))
    request_np = np.loadtxt(request_str, delimiter=',')
    price_tensor = tf.make_tensor_proto(request_np[:,0], dtype=tf.float32)
    inventory_tensor = tf.make_tensor_proto(request_np[:,1], dtype=tf.float32)
    return {"price": price_tensor, "inventory": inventory_tensor}   
    

@monitor(labels=_labels, name="transform_response")
def _transform_response(response):
    return json.dumps({"outputs": response['relevance'].tolist()})

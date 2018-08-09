import os
import numpy as np
import json
import logging

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

import mxnet as mx

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']

_labels = {
           'model_name': 'mnist',
           'model_tag': 'cpu',
           'model_type': 'mxnet',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    ''' Initialize / Restore Model Object.
    '''
    ctx = mx.cpu()

    symbol, arg_params, aux_params = mx.model.load_checkpoint('mnist', 1)

    mod = mx.mod.Module(symbol=symbol,
                        context=ctx,
                        label_names=None)

    # batch-num_filter-y-x
    mod.bind(for_training=False,
             data_shapes=[('data', (1, 1, 28, 28))],
             label_shapes=mod._label_shapes)

    mod.set_params(arg_params,
                   aux_params,
                   allow_missing=True)

    return mod 


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.forward(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0).reshape(1, 28, 28)
    return {"image": request_np}


def _transform_response(response):
    return json.dumps({"classes": response['classes'].tolist(), 
                       "probabilities": response['probabilities'].tolist(),
                      })


if __name__ == '__main__':
    with open('../input/predict/test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

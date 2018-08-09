import os
import numpy as np
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import ujson
import logging
import mxnet as mx

_logger = logging.getLogger('pipelineai')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
           'model_name': 'mnist',
           'model_tag': 'v1',
           'model_type': 'mxnet',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    ctx = mx.cpu()

    sym, arg_params, aux_params = mx.model.load_checkpoint('mnist', 0)

    mod = mx.mod.Module(symbol=sym,
                        context=ctx,
                        label_names=None)

    mod.bind(for_training=False,
             data_shapes[('data', (1,28,28))],
             label_shapes=mod._label_shapes)

    mod.set_params(arg_params,
                   aux_params,
                   allow_missing=True)

    return restored_model


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""
    transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        reponse = _model.predict(transformed_request)

    return _transform_response(response)


# TODO:  Implement this:
#   https://mxnet.incubator.apache.org/tutorials/python/predict_image.html


@monitor(labels=_labels, name="transform_request")
def _transform_request(request):
    request_str = request.decode('utf-8')
    request_str = request_str.strip().replace('\n', ',')
    # surround the json with '[' ']' to prepare for conversion
    request_str = '[%s]' % request_str
    request_json = ujson.loads(request_str)
    request_transformed = ([json_line['feature0'] for json_line in request_json])

#    print(request_transformed.reshape(-1, 1))

    return np.array(request_transformed).reshape(-1, 1)


@monitor(labels=_labels, name="transform_response")
def _transform_response(response):
    return ujson.dumps(response.tolist())

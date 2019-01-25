import os
import numpy as np
import json
import logging

from pipeline_model import TensorFlowServingModel
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import tensorflow as tf

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'name': 'transfer',
           'tag': 'v2',
           'runtime': 'tfserving',
           'chip': 'cpu',
           'resource_name': '83f05e58transfer',
#           'resource_tag': '',
           'resource_type': 'model',
           'resource_subtype': 'keras',
          }

def _initialize_upon_import():
    """ Initialize / Restore Model Object."""
    return TensorFlowServingModel(host='localhost',
                                  port=9000,
                                  model_name='tfserving',
                                  model_signature_name=None,
                                  timeout_seconds=10.0)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


# Schema
#   input: bytes
#   return: json
@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model.predict(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response


# input: bytes
# return: dict{}
def _transform_request(request):
    """
    Convert from bytes/json/etc to dict of tf.tensor/np.array/etc
    :param request:
    :return:
    """
    # TODO: Uncomment out one of the examples below - or provide your own implementation
    #
    # Note:  The dict keys used below (ie. 'image') depend on the TF SignatureDef of your exported SavedModel
    #
    # Example 1: Convert json version of an image starting from raw bytes => dict{} to feed TF Serving
    #
    # request_str = request.decode('utf-8')
    # request_json = json.loads(request_str)
    # request_np = (np.array(request_json['image'], dtype=np.float32) / 255.0).reshape(1, 28, 28)
    # image_tensor = tf.make_tensor_proto(request_np, dtype=tf.float32)
    # transformed_request_dict = {"image": image_tensor}
    # return transformed_request_dict   # Becomes `PredictRequest.inputs['image'] = image_tensor`

    # Example 2: Convert raw bytes version of an image => dict{} to feed TF Serving
    #
    # image_tensor = tf.make_tensor_proto([request], shape=[1])
    # transformed_request_dict['image'] = image_tensor
    # return transformed_request_dict   # Becomes `PredictRequest.inputs['image'] = image_tensor`


# input: dict{}
# return: anything you want! (json, bytes, etc)
def _transform_response(response):
    """
    Convert from dict{tf.tensor/np.array/etc} to bytes/json/etc
    :param response:
    :return:
    """
    # TODO:  Uncomment out the example below - or provide your own implementation
    #
    # Note:  The dict keys used below (ie. 'classes', 'probabilities') depend on the TF SignatureDef of your exported SavedModel
    #
    # Example 1:
    #
    # classes_list = response['classes'].tolist()  # Comes from TF Serving's `PredictResponse.outputs['classes']
    # probabilities_list = response['probabilities'].tolist() # Comes from TF Serving's `PredictResponse.outputs['probabilities']
    #
    # response_dict = {"classes": classes_list,
    #                  "probabilities": probabilities,
    #                 }
    #
    # transformed_response = json.dumps(response_dict)
    # return transformed_response
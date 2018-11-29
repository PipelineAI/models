import os
import json
import logging
import requests
import numpy as np

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'name': 'mnist',
           'tag': 'ensemble',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'python',
          }

# There is no model to import.
# This function is merely calling other functions/models
#   and aggregating the results.
# def _initialize_upon_import():
#    return


# This is called unconditionally at *module import time*...
# _model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        # TODO: Handle any failure responses such as Fallback/Circuit-Breaker, etc

        timeout_seconds=1200

        # TODO: Can we use internal dns name (predict-mnist)
        # TODO: Pass along the request-tracing headers
        url_model_a = 'https://community.cloud.pipeline.ai/predict/83f05e58/mnista/invoke'
        response_a = requests.post(
            url=url_model_a,
            data=transformed_request,
            timeout=timeout_seconds
        )

        url_model_b = 'https://community.cloud.pipeline.ai/predict/83f05e58/mnistb/invoke'
        response_b = requests.post(
            url=url_model_b,
            data=transformed_request,
            timeout=timeout_seconds
        )

        url_model_c = 'https://community.cloud.pipeline.ai/predict/83f05e58/mnistc/invoke'
        response_c = requests.post(
            url=url_model_c,
            data=transformed_request,
            timeout=timeout_seconds
        )

    # TODO: Aggregate the responses into a single response
    #       * Classification:  Return the majority class from all predicted classes
    #       * Regression:  Average the result
    # TODO: Include all models that participated in the response (including confidences, timings, etc)

    response = [response_a.json(), response_b.json(), response_c.json()]

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_responsei(response_a, response_b, response_c)

    return transformed_response


def _transform_request(request):
#    request_str = request.decode('utf-8')
#    request_json = json.loads(request_str)
#    request_np = (np.array(request_json['image'], dtype=np.float32) / 255.0).reshape(1, 28, 28)
#    return {"image": json.load(request_np)}
    return request


def _transform_response(response_a, response_b, response_c):
#    return json.dumps({"classes": response['classes'].tolist(),
#                       "probabilities": response['probabilities'].tolist(),
#                      })
    # TODO:  Apply quorum aggregator function vs. hard-coding to class_c 
    class_a_list = response_a['classes']
    class_b_list = response_b['classes']
    class_c_list = response_c['classes']
    
    # TODO:  Aggregate probabilities? 
    probabilities_a_list_list = response_a['probabilities']
    probabilities_b_list_list = response_b['probabilities']
    probabilities_c_list_list = response_c['probabilities']

    # TODO:  
    return json.dumps({"classes": class_c_list,
                       "probabilities": probabilities_c_list_list})


# Note:  This is a mini test
if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

import os
import json
import logging

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'model_name': 'mnist',
           'model_tag': 'ensemble',
           'model_type': 'python',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }

# There is no model to import.  
# This function is merely calling other functions/models
#   and aggregating the results.
def _initialize_upon_import():
    return 


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
# TODO:  Using python requests, implement a call to 1 other model (for now)
#          using a model that has been deployed to dev or prod
#        For now, just use the full external URL displayed in the http snippet
#          in the dev or prod UI.
#
#        response = ...

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response

# Note:  Don't change this...
def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0).reshape(1, 28, 28)
    return {"image": request_np}

# Note:  Don't change this...
def _transform_response(response):
    return json.dumps({"classes": response['classes'].tolist(),
                       "probabilities": response['probabilities'].tolist(),
                      })

# Note:  This is a mini test
if __name__ == '__main__':
    with open('../input/predict/test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

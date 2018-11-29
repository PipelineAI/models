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
    'name': 'echo',
    'tag': 'v1',
    'runtime': 'python',
    'chip': 'cpu',
    'resource_type': 'model',
    'resource_subtype': 'autoroute',
}


def _echo(request):
   return request


@log(labels=_labels, logger=_logger)
def invoke(request: bytes) -> str:
    """
    Transform bytes posted to the api into a python dictionary containing the
    existing resource routes by tag and weight.
    Predict least expensive routes and adjust higher weights to lower cost routes.
    Transform the model prediction output from python dictionary to a JSON formatted str
    containing the new resource routes by tag and weight

    :param bytes request:   bytes containing the payload to supply to the predict method

    :return:                Response obj serialized to a JSON formatted str
                                containing the new resource routes by tag and weight
    """
    with monitor(labels=_labels, name='transform_request'):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name='invoke'):
        response = _echo(transformed_request)

    with monitor(labels=_labels, name='transform_response'):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request: bytes) -> dict:
    """
    Transform bytes posted to the api into a python dictionary containing
    the resource routes by tag and weight

    :param bytes request:   containing the payload to supply to the predict method

    :return:                dict containing the resource routes by tag and weight
    """
    return dict(json.loads(request.decode('utf-8'))['resource_split_tag_and_weight_dict'])


def _transform_response(response: dict) -> str:
    """
    Transform response from a python dictionary to a JSON formatted string

    :param dict response:     dict containing the new resource routes by tag and weight

    :return:                  Response obj serialized to a JSON formatted str
    """

    return json.dumps({
        'resource_split_tag_and_weight_dict': response
    })


if __name__ == '__main__':
    with open('pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

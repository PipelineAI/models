import json
import logging
import random

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
    'name': 'random',
    'tag': 'v1',
    'runtime': 'python',
    'chip': 'cpu',
    'resource_type': 'model',
    'resource_subtype': 'autoroute'
}


def _randomize(routes: dict) -> dict:
    """
    TODO: Implement bandit logic to optimize route weights

    :param dict routes:     existing routes by tag and weight

    :return:                dict: bandit optimized routes by tag and weight
    """

    n = len(routes)
    autoroutes = dict()
    cumulative_total = 0

    # TODO: replace deployment target cost simulator with your custom bandit logic
    # deployment target cost simulator
    for (k, v) in routes.items():
        if n == 1:
            i = 100 - cumulative_total
        else:
            i = random.randint(1, 21)*5
            if cumulative_total + i > 100:
               i = 0

        cumulative_total += i
        autoroutes[k] = i
        n -= 1

    if cumulative_total < 100:
        autoroutes[next(iter(routes.keys()))] = 100 - cumulative_total

    return autoroutes


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
        response = _randomize(transformed_request)

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

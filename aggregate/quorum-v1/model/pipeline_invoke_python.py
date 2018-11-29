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
    'name': 'quorum',
    'tag': 'v1',
    'runtime': 'python',
    'chip': 'cpu',
    'resource_type': 'model',
    'resource_subtype': 'aggregate',
}


def _quorum(tag_and_prediction_dict: dict):
   quorum = -1

   # TODO: Iterate through items and find value of the majority
   #       throw exception if we can't find majority 
   return quorum


@log(labels=_labels, logger=_logger)
def invoke(request: bytes) -> str:
    with monitor(labels=_labels, name='transform_request'):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name='invoke'):
        response = _quorum(transformed_request)

    with monitor(labels=_labels, name='transform_response'):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request: bytes) -> dict:
    return dict(json.loads(request.decode('utf-8')))


def _transform_response(response) -> str:
    return json.dumps({
        'quorum': response
    })


if __name__ == '__main__':
    with open('pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

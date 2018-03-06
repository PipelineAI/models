import os
import json
import logging
import requests

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

from slackclient import SlackClient

#_slack_token = os.environ["SLACK_API_TOKEN"]
#_slack_token = ''
_sc = SlackClient(_slack_token)

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['predict']

_model_tag = os.environ['PIPELINE_MODEL_TAG']

_labels= {'model_runtime': 'python',
          'model_type': 'python', 
          'model_name': 'gitstar',
          'model_tag': _model_tag,
          'model_chip': 'cpu'
         }

_stream_url = 'http://stream-gitstar-%s:8082' % _model_tag 
_stream_url = _stream_url.rstrip('/')

_stream_topic = 'gitstar-%s-input' % _model_tag

_endpoint_url = '%s/topics/%s' % (_stream_url, _stream_topic)
_endpoint_url = _endpoint_url.rstrip('/')

_accept_and_content_type_headers = {"Accept": "application/vnd.kafka.v2+json",
                                    "Content-Type": "application/vnd.kafka.json.v2+json"}

@log(labels=_labels, logger=_logger)
def predict(request: bytes) -> bytes:
    with monitor(labels=_labels, name="predict"):

        request_str = request.decode('utf-8')

        body = '{"records": [{"value":%s}]}' % request_str

        response = requests.post(url=_endpoint_url,
                                 headers=_accept_and_content_type_headers,
                                 data=body.encode('utf-8'),
                                 timeout=30)

        _sc.api_call(
          "chat.postMessage",
          channel="G9L5CFPHD",
          text=response.text
        )

        return {'response': response.text}

#predict(b'{"blah": "blah"}')

import os
import json
import logging
import requests
import urllib

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

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

_stream_endpoint_url = '%s/topics/%s' % (_stream_url, _stream_topic)
_stream_endpoint_url = _stream_endpoint_url.rstrip('/')

_stream_accept_and_content_type_headers = {"Accept": "application/vnd.kafka.v2+json",
                                           "Content-Type": "application/vnd.kafka.json.v2+json"}

#_slack_url = 'http://hooks.slack.com:443/services/T6QHWMRD4/B9KNAA0BS/dsglc5SFARz3hISU4pDlAms3'
#_slack_url = "curl -X POST --data 'token=xoxa-228608739446-303548610531-303548610803-376b8dcda37e59fc571c660eb0fb9c1d&channel=%40cfregly&text=%s' https://slack.com//api/chat.postMessage"

@log(labels=_labels, logger=_logger)
def predict(request: bytes) -> bytes:
    with monitor(labels=_labels, name="predict"):

        request_str = request.decode('utf-8')
        print(request_str)

        avatar_url = json.loads(request_str)['sender']['avatar_url']
        print(avatar_url)
        
#        stream_body = '{"records": [{"value":%s}]}' % request_str
#        response = requests.post(url=_stream_endpoint_url,
#                                 headers=_stream_accept_and_content_type_headers,
#                                 data=stream_body.encode('utf-8'),
#                                 timeout=30)

        #import urllib
        #avatar_url = urllib.parse.quote(avatar_url)

        cmd = 'curl -X POST --data "token=xoxa-228608739446-303548610531-303548610803-376b8dcda37e59fc571c660eb0fb9c1d&channel=demo-community&text=%s" http://slack.com:443/api/chat.postMessage' % avatar_url

#        cmd = 'curl -X POST --data-urlencode "payload={\\"unfurl_links\\": true, \\"channel\\": \\"#demo-community\\", \\"username\\": \\"pipelineai_bot\\", \\"text\\": \\"%s\\"}" http://hooks.slack.com:443/services/T6QHWMRD4/B9KNAA0BS/dsglc5SFARz3hISU4pDlAms3' % avatar_url 
        print(cmd)

        import subprocess
        subprocess.call(cmd, shell=True)

#        if slack_response.status_code != 200:
#            raise ValueError(
#                'Request to slack returned an error %s, the response is:\n%s'
#                % (slack_response.status_code, slack_response.text)
#        )

        return {'response': 'OK'}

# TODO:  create _transform_request(), _transform_response()

#predict(b'{"blah": "https://avatars1.githubusercontent.com/u/1438064?s=460&v=4"}')
#predict(b'{"image_url": "https://avatars1.githubusercontent.com/u/1438064?s=460%26v=4"}')

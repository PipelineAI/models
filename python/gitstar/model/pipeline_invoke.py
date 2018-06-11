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

__all__ = ['invoke']

_labels = {
           'model_name': 'gitstar',
           'model_tag': 'v1',
           'model_type': 'python',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }

#_stream_url = 'http://stream-gitstar-v1:8082' 
#_stream_url = _stream_url.rstrip('/')

#_stream_topic = 'gitstar-v1-input' 

#_stream_endpoint_url = '%s/topics/%s' % (_stream_url, _stream_topic)
#_stream_endpoint_url = _stream_endpoint_url.rstrip('/')

#_stream_accept_and_content_type_headers = {"Accept": "application/vnd.kafka.v2+json",
#                                           "Content-Type": "application/vnd.kafka.json.v2+json"}

#_slack_url = 'http://hooks.slack.com:443/services/T6QHWMRD4/B9KNAA0BS/dsglc5SFARz3hISU4pDlAms3'

@log(labels=_labels, logger=_logger)
def invoke(request):
    with monitor(labels=_labels, name="invoke"):

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

        # Note:  https:// doesn't work through istio - it appears we need to use http://...:443/, however this doesn't work well with the slack api (Shows a CloudFront issue when we try to do this...)

#        headers = {'Content-Type': 'text/plain'}
#        response = requests.post(headers=headers,
#                                 url='https://slack.com/api/chat.postMessage?token=xoxp-228608739446-227185306481-378989962386-a2ee86b00339b0d15440910408a08b73&channel=demo-community&text=%s' % avatar_url)

#        response = response.text

        cmd = 'curl -X POST --data-urlencode "payload={\\"unfurl_links\\": true, \\"channel\\": \\"#demo-community\\", \\"username\\": \\"pipelineai_bot\\", \\"text\\": \\"%s\\"}" https://hooks.slack.com/services/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' % avatar_url 
        print(cmd)
        import subprocess
        response = subprocess.check_output(cmd, shell=True).decode('utf-8')

#        payload = {
#                   "unfurl_links": "true", 
#                   "channel": "#demo-community", 
#                   "username": "pipelineai_bot",
#                   "text": "%s" % avatar_url
#                  }

#        response = requests.post(
#                                 url='https://hooks.slack.com/services/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',
#                                 data=payload
#                                )
#        response = response.text

#        import subprocess
#        response = subprocess.check_output(cmd, shell=True).decode('utf-8')

#        if slack_response.status_code != 200:
#            raise ValueError(
#                'Request to slack returned an error %s, the response is:\n%s'
#                % (slack_response.status_code, slack_response.text)
#        )

        return {'response': response}

# TODO:  create _transform_request(), _transform_response()

#predict(b'{"blah": "https://avatars1.githubusercontent.com/u/1438064?s=460&v=4"}')
#predict(b'{"image_url": "https://avatars1.githubusercontent.com/u/1438064?s=460%26v=4"}')

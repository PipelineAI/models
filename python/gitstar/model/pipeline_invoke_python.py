import os
import json
import logging
import requests
import urllib
import subprocess
#import tweepy
import inception

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']

_labels = {
           'name': 'gitstar',
           'tag': 'v1',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'python',
          }

@log(labels=_labels, logger=_logger)
def invoke(request):
    with monitor(labels=_labels, name="invoke"):
        request_str = request.decode('utf-8')
        avatar_url = json.loads(request_str)['sender']['avatar_url']
        github_login = json.loads(request_str)['sender']['login']
        classification_response = inception.invoke(avatar_url)

        classification_response_json= json.loads(classification_response)

        classification_response_formatted = '\\n '.join("%s%%\t%s" % (str((100 * item['score']))[0:4], item['name']) for item in classification_response_json)

        cmd = 'curl -X POST --data-urlencode "payload={\\"unfurl_links\\": true, \\"channel\\": \\"#community\\", \\"username\\": \\"pipelineai_bot\\", \\"text\\": \\"%s has starred the PipelineAI GitHub Repo!\n%s\nTheir avatar picture is classified as follows:\n%s\nTo classify your avatar picture, star the PipelineAI GitHub Repo @ https://github.com/PipelineAI/pipeline\\"}" https://hooks.slack.com/services/T/B/o' % (github_login, avatar_url, (classification_response_formatted or ''))
        response = subprocess.check_output(cmd, shell=True).decode('utf-8')

# https://github.com/alexellis/faas-twitter-fanclub/blob/master/tweet-stargazer/handler.py
#
#        auth = tweepy.OAuthHandler(os.environ["consumer_key"], os.environ["consumer_secret"])
#        auth.set_access_token(os.environ["access_token"], os.environ["access_token_secret"])
#        github_login = json.loads(request_str)['sender']['login']
#        api = tweepy.API(auth)
#        api.update_with_media('%s' % filename, '%s' % github_login)

        filename = avatar_url.split('/')
        if filename:
            idx = len(filename) - 1
            filename = filename[idx]
            if os.path.exists('inception/%s' % filename):
                os.remove('inception/%s' % filename)

        return {'response': response}


if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fh:
        contents_binary = fh.read()
        response = invoke(contents_binary)
        print(response)

import os
import numpy as np
import json
import logging
import random
import urllib.request
from urllib.parse import unquote

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

import tensorflow as tf
from tensorflow.contrib import predictor

from tensorflow.keras.applications.mobilenet import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import glob

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'name': 'transfer',
           'tag': 'v1',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'keras',
          }


def _initialize_upon_import():
    """Initialize / Restore Model Object."""
    saved_model_path = './pipeline_tfserving/0'
    loaded_model = tf.contrib.saved_model.load_keras_model(saved_model_path)
    return loaded_model

_classes_list = glob.glob("./images/train/*")

# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()

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


def download_image(url):
    name = random.randrange(1,100000)
    fullname = '/tmp/' + str(name) + ".img"
    urllib.request.urlretrieve(url, fullname)     
    return fullname


def _transform_request(request):
    request = request.decode('utf-8')
    request = unquote(request) 

    # Direct http example
    if request.startswith('http'):
        request = download_image(request)
    else:
        # Slack Label Example
        request_array = request.split('&')
        print(request_array)

        result = [value for value in request_array if value.startswith('text=')]
        if len(result) > 0:
            request = download_image(result[0][5:])
            print(request)
               
    predict_img = image.load_img(request, target_size=(224, 224))
    predict_img_array = image.img_to_array(predict_img)
    predict_img_array = np.expand_dims(predict_img_array, axis=0)
    predict_preprocess_img = preprocess_input(predict_img_array)

    return predict_preprocess_img


def _transform_response(response):
    return json.dumps({"classes": _classes_list,
                       "probabilities": response.tolist()[0],
                      })


if __name__ == '__main__':
#    request = './images/predict/cat.jpg'
#    request = b'http%3A%2F%2Fsite.meishij.net%2Fr%2F58%2F25%2F3568808%2Fa3568808_142682562777944.jpg'

    request = b'token=ncx3XaUkJTXbGmVUoeO5HdKI&team_id=T6QHWMRD4&team_domain=pipelineai&channel_id=G9L5CFPHD&channel_name=privategroup&user_id=U6P5F90E5&user_name=cfregly&command=/predict&text=https://avatars1.githubusercontent.com/u/1438064?s=460&v=4&response_url=https://hooks.slack.com/commands/T6QHWMRD4/516521813893/rnZVzpibSOZ4TqIkq7MaSjPQ&trigger_id=515584150832.228608739446.0381131e21d800f95e33fbeb038e03d3'

    response = invoke(request)
    print(response)
#    with open('./pipeline_test_request.json', 'rb') as fb:
#        request_bytes = fb.read()
#        response_bytes = invoke(request_bytes)
#        print(response_bytes)

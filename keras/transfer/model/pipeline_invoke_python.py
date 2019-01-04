import os
import numpy as np
import json
import logging

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


import random
import urllib.request

def download_image(url):
    name = random.randrange(1,100000)
    fullname = str(name)+".pimg"
    urllib.request.urlretrieve(url, fullname)     
    return fullname

def _transform_request(request):
    if 'http' in request:
        request = download_image(request)
                
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
    request = 'http://site.meishij.net/r/58/25/3568808/a3568808_142682562777944.jpg'
    response = invoke(request)
    print(response)
#    with open('./pipeline_test_request.json', 'rb') as fb:
#        request_bytes = fb.read()
#        response_bytes = invoke(request_bytes)
#        print(response_bytes)

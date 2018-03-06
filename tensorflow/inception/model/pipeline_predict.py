import os
import numpy as np
import json
import logging

from pipeline_model import TensorFlowServingModel
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import tensorflow as tf

import requests
from PIL import Image
from io import BytesIO

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['predict']


_labels= {'model_runtime': 'tfserving',
          'model_type': 'tensorflow',
          'model_name': 'inception',
          'model_tag': 'cpu',
          'model_chip': 'cpu',
         }


def _initialize_upon_import() -> TensorFlowServingModel:
    ''' Initialize / Restore Model Object.
    '''
    return TensorFlowServingModel(host='localhost',
                                  port=9000,
                                  model_name='inception',
                                  model_signature_name='predict_images',
                                  timeout_seconds=10.0)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


# https://www.tensorflow.org/serving/serving_inception
@log(labels=_labels, logger=_logger)
def predict(request: bytes) -> bytes:
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="predict"):
       predictions = _model.predict(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(predictions)

    return transformed_response


def _transform_request(request: bytes) -> dict:
    # Convert from bytes to tf.tensor, np.array, etc.
    # This needs to be a JPEG
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    
    image_url = request_json['image_url']

    image_response = requests.get(image_url, stream=True)
    if image_response.status_code == 200:
#        image_bytes = image_response.read()  #BytesIO(image_response.content)
#        image = Image.open(image_bytes)

        #image_file_path = './inception/cropped_panda.jpg' 
        #with open(image_file_path, 'rb') as f:
        #    image = f.read()
#        image.show()

#        with open(path, 'wb') as f:
#            for chunk in r.iter_content(1024):
#                f.write(chunk)

        image = Image.open(BytesIO(image_response.content))
        image_tensor = tf.make_tensor_proto(image)
                                            #shape=[1])

        return {"images": image_tensor}


def _transform_response(response: dict) -> json:
    # Convert from tf.tensor, np.array, etc. to bytes
    pass

predict(b'{"image_url": "https://avatars1.githubusercontent.com/u/1438064?s=460&v=4"}')

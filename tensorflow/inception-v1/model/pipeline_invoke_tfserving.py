import os
import json
import logging

from pipeline_model import TensorFlowServingModel
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import tensorflow as tf

import requests
from PIL import Image
from io import StringIO, BytesIO

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'model_name': 'inception',
           'model_tag': 'v1',
           'model_type': 'tensorflow',
           'model_runtime': 'tfserving',
           'model_chip': 'cpu',
         }


def _initialize_upon_import():
    """ Initialize / Restore Model Object.
    """
    return TensorFlowServingModel(host='localhost',
                                  port=9000,
                                  model_name='tfserving',
                                  model_signature_name='predict_images',
                                  timeout_seconds=10.0)


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


# https://www.tensorflow.org/serving/serving_inception
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


def _transform_request(request):
    # Convert from bytes to tf.tensor, np.array, etc.
    # This needs to be a JPEG
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)

#    image_url = request_json['image_url']
#    image = Image.open(requests.get(image_url, stream=True).raw)

    image_response = requests.get(request_json['image_url'])
#    with BytesIO(image_response.content) as f:
#        with Image.open(f) as img:
#            print(img)
#            image = img

#    image_file_path = '%s/images/fregly_avatar.jpg' % os.environ['PIPELINE_INPUT_PATH']

    from datetime import datetime
    version = int(datetime.now().strftime("%s"))

    image_file_path = 'blah-%s.jpg' % version

    with open(image_file_path, 'wb') as f:
        f.write(image_response.content)

    with open(image_file_path, 'rb') as f:
        image = f.read()

# TODO:  https://towardsdatascience.com/tensorflow-serving-client-make-it-slimmer-and-faster-b3e5f71208fb
#        https://github.com/Vetal1977/tf_serving_example/tree/master/tensorflow/core/framework
    #image_tensor = tf.make_tensor_proto(image)
    #                                    shape=[1])
    # NEW STUFF - pipeline_model==1.10+
    # Replacement for tf.make_tensor_proto(image, shape=[1])
    # Create TensorProto object for a request
    #
    #from tensorflow.core.framework import tensor_pb2
    #from tensorflow.core.framework import tensor_shape_pb2
    #from tensorflow.core.framework import types_pb2
    #
    #dims = [tensor_shape_pb2.TensorShapeProto.Dim(size=1)]
    #tensor_shape_proto = tensor_shape_pb2.TensorShapeProto(dim=dims)
    #image_tensor_proto = tensor_pb2.TensorProto(dtype=types_pb2.DT_STRING,
    #                                            tensor_shape=tensor_shape_proto,
    #                                            string_val=[image])
    #
    image_tensor_proto = tf.make_tensor_proto(image,
                                              shape=[1])

    return {"images": image_tensor_proto}


def _transform_response(response):
    # Convert from tf.tensor, np.array, etc. to bytes

    # TODO:  Optimize this to avoid tf.make_ndarray similar to _transform_request() above
    class_list = tf.make_ndarray(response['classes']).tolist()[0]
    class_list_str = [clazz.decode('utf-8') for clazz in class_list]
    score_list = tf.make_ndarray(response['scores']).tolist()[0]

    return {"classes": class_list_str,
            "scores": score_list}


#predict(b'{"image_url": "https://avatars1.githubusercontent.com/u/1438064?s=460&v=4"}')

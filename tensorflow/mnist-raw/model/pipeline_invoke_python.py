# ------------- Builtin imports --------------------------------------------------------------------
import json
import logging

# ------------- 3rd-party imports ------------------------------------------------------------------
import tensorflow as tf
from tensorflow.contrib import predictor

# --- PipelineAI imports ---------------------------------------------------------------------------
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log


_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.DEBUG)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
    'model_name': 'mnist',
    'model_tag': 'raw',
    'model_type': 'tensorflow',
    'model_runtime': 'python',
    'model_chip': 'cpu',
}


def _initialize_upon_import():

    try:

        saved_model_path = './pipeline_tfserving/0'
        return predictor.from_saved_model(saved_model_path)

    except Exception:
        _logger.error('pipeline_invoke_python._initialize_upon_import.Exception:', exc_info=True)

    return None


_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """
    Transform bytes posted to the api into a Tensor.
    Classify the image
    Transform the model prediction output from a 1D array to a list of classes and probabilities

    :param bytes request:   byte array containing the content required by the predict method

    :return:                Response obj serialized to a JSON formatted str
                                containing a list of classes and a list of probabilities
    """
    try:

        with monitor(labels=_labels, name="transform_request"):
            transformed_request = _transform_request(request)

        with monitor(labels=_labels, name="invoke"):
            response = _model(transformed_request)

        with monitor(labels=_labels, name="transform_response"):
            transformed_response = _transform_response(response)

        return transformed_response

    except Exception:
        _logger.error('pipeline_invoke_python.invoke.Exception:', exc_info=True)


def _transform_request(request):
    _logger.info('_transform_request: request: %s' % request)

    # channels indicates the desired number of color channels for the decoded image.
    #
    # Defaults to 0
    #
    # Accepted values are:
    #
    #   0: Use the number of channels in the PNG-encoded image.
    #   1: output a grayscale image.
    #   3: output an RGB image.
    #   4: output an RGBA image.

    request_image_tensor = tf.image.decode_png(request, channels=1)
    request_image_tensor_resized = tf.image.resize_images(request_image_tensor, size=[28, 28])

    sess = tf.Session()
    with sess.as_default():
        request_np = request_image_tensor_resized.eval()
        request_np = (request_np / 255.0).reshape(1, 28, 28)

    return {"image": request_np}


def _transform_response(response):

    return json.dumps({
        "classes": response['classes'].tolist(),
        "probabilities": response['probabilities'].tolist(),
    })


if __name__ == '__main__':
    with open('./pipeline_test_request.png', 'rb') as fb:
        request_bytes = fb.read()
        response_json = invoke(request_bytes)
        print(response_json)

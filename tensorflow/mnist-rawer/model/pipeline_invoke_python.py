# ------------- Builtin imports --------------------------------------------------------------------
import json
import logging
import os

# ------------- 3rd-party imports ------------------------------------------------------------------
import tensorflow as tf
from tensorflow.contrib import predictor

# --- PipelineAI imports ---------------------------------------------------------------------------
# from pipeline_monitor import prometheus_monitor as monitor
# from pipeline_logger import log


_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.DEBUG)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
    'model_name': 'mnist',
    'model_tag': 'rawer',
    'model_type': 'tensorflow',
    'model_runtime': 'python',
    'model_chip': 'cpu',
}


def _initialize_upon_import():

    try:

        saved_model_path = './pipeline_tfserving/0'
        # saved_model_path = os.path.expandvars(saved_model_path)
        # saved_model_path = os.path.expanduser(saved_model_path)
        # saved_model_path = os.path.normpath(saved_model_path)
        # saved_model_path = os.path.abspath(saved_model_path)

        _logger.info('saved_model_path: {}'.format(saved_model_path))
        _logger.info('os.path.exists(saved_model_path): {}'.format(os.path.exists(saved_model_path)))

        return predictor.from_saved_model(saved_model_path)

    except Exception as exc:
        print('pipeline_invoke_python._initialize_upon_import.Exception: {}'.format(exc))
        _logger.error('pipeline_invoke_python._initialize_upon_import.Exception:', exc_info=True)

    return None


_model = _initialize_upon_import()


# @log(labels=_labels, logger=_logger)
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
        _logger.info('invoke: raw request: {}'.format(request))
        print('invoke: raw request: {}'.format(request))
        # _logger.info('invoke: raw request: %s' % request)
        # with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)
        _logger.info('invoke: transformed request: {}'.format(transformed_request))

        # with monitor(labels=_labels, name="invoke"):
        response = _model(transformed_request)
        _logger.info('invoke: response: {}'.format(response))

        # with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)
        _logger.info('invoke: transformed response: {}'.format(transformed_response))

        return transformed_response

    except Exception as exc:
        print('pipeline_invoke_python.invoke.Exception: {}'.format(exc))
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
    _logger.info('_transform_request: request_image_tensor: %s' % request_image_tensor)

    request_image_tensor_resized = tf.image.resize_images(request_image_tensor, size=[28, 28])
    _logger.info('_transform_request: request_image_tensor_resized: {}'.format(request_image_tensor_resized))

    sess = tf.Session()
    with sess.as_default():
        request_np = request_image_tensor_resized.eval()
        _logger.info('_transform_request: request_np: %s' % request_np)

        reshaped_request_np = request_np.reshape(1, 28, 28)
        _logger.info('_transform_request: reshaped_request_np: %s' % reshaped_request_np)

    return {"image": reshaped_request_np}


def _transform_response(response):
    _logger.info('_transform_response: raw response: %s' % response)

    return json.dumps({
        "classes": response['classes'].tolist(),
        "probabilities": response['probabilities'].tolist(),
    })


# if __name__ == '__main__':
#     with open('9.png', 'rb') as fb:
#         request_bytes = fb.read()
#         response_json = invoke(request_bytes)
#         print(response_json)

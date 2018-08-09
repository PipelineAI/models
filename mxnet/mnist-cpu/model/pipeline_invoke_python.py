import os
import numpy as np
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log
import logging
import mxnet as mx

_logger = logging.getLogger('pipelineai')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
           'model_name': 'mnist',
           'model_tag': 'v1',
           'model_type': 'mxnet',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


#LOOK HERE:
# * https://github.com/awslabs/keras-apache-mxnet/wiki/Save-MXNet-model-from-Keras-MXNet#import-the-model-in-mxnet-for-inference
# * https://github.com/awslabs/keras-apache-mxnet/wiki/Keras-MXNet-2.1.6-Release-Notes


def _initialize_upon_import():
    # Step1: Load the model in MXNet

    # Use the same prefix and epoch parameters we used in save_mxnet_model API.
    sym, arg_params, aux_params = mx.model.load_checkpoint(prefix='mnist_cnn', epoch=0)

    # We use the data_names and data_shapes returned by save_mxnet_model API.
    mod = mx.mod.Module(symbol=sym,
                    data_names=['/conv2d_1_input1'],
                    context=mx.cpu(),
                    label_names=None)
    mod.bind(for_training=False,
             data_shapes=[('/conv2d_1_input1', (1,1,28,28))],
             label_shapes=mod._label_shapes)
    mod.set_params(arg_params, aux_params, allow_missing=True)

    # Now, the model is loaded in MXNet and ready for Inference!

    return mod

# Check what is the predicted value and actual value
# We have predicted 10000 samples in test_data. Use different indexes to see different sample results.
idx = 1020
print("Predicted - ", np.argmax(result[idx].asnumpy()))
print("Actual - ", labels[idx])


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    """Where the magic happens..."""
    transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        data_iter = mx.io.NDArrayIter(transformed_request, None, 1)
        response = mod.predict(data_iter)
    return _transform_response(response)


def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0).reshape(1, 28, 28)

    # TODO:  This must match whatever the model is expecting
    return {"image": request_np}


def _transform_response(response):
#    return json.dumps({"classes": response['classes'].tolist(),
#                       "probabilities": response['probabilities'].tolist(),
#                      })
    print(response)
    return response

if __name__ == '__main__':
    with open('../input/predict/test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

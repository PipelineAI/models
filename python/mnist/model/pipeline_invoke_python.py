#!/usr/bin/env python3
# --- Builtin imports ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import glob
import json
import logging
import os

# --- 3rd-party imports ----------------------------------------------------------------------------
import numpy as np
from sklearn.externals import joblib
import xgboost as xgb

# --- PipelineAI imports ---------------------------------------------------------------------------
from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']

_labels = {
           'name': 'mnist',
           'tag': 'v1',
           'runtime': 'python',
           'chip': 'cpu',
           'resource_type': 'model',
           'resource_subtype': 'python',
          }

def _initialize_upon_import(file_name: str=None, train_datetime: str=None) -> xgb.core.Booster:
    """
    Load artifact into memory from a pickled binary archive.

    The most recently created artifact is returned
    when file_name or train_datetime is not supplied.

    :param str file_name:       The name of the pickled estimator binary artifact, not including path
    :param str train_datetime:  The date and time the training session that created the estimator
                                in the format: YmdHMS

    :return:                    xgboost.core.Booster: un-pickled artifact
    """

    if not file_name:

        compressor = 'bz2'

        if train_datetime:
            file_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     '{}_model.pkl.{}'.format(train_datetime, compressor))
        else:
            file_mask = '*model.pkl.{}'.format(compressor)
            pathname = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_mask)
            file_list = glob.glob(pathname)
            latest_file = max(file_list, key=os.path.getctime)
            file_name = os.path.basename(latest_file)

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), file_name)
    _logger.info('model path: {}'.format(path))

    with open(path, 'rb') as f:
        artifact = joblib.load(f)

    return artifact


_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request: bytes) -> str:
    """
    Transform bytes posted to the api into an XGBoost DMatrix which is an
    internal data structure that is used by XGBoost which is optimized for
    both memory efficiency and training speed.
    Classify the image
    Transform the model prediction output from a 1D array to a list of classes and probabilities

    :param bytes request:   bytes containing the payload to supply to the predict method

    :return:                Response obj serialized to a JSON formatted str
                            containing a list of classes and a list of probabilities
    """
    with monitor(labels=_labels, name='transform_request'):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name='invoke'):
        response = _model.predict(transformed_request)

    with monitor(labels=_labels, name='transform_response'):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request: bytes) -> xgb.DMatrix:
    """
    Transform bytes posted to the api into an XGBoost DMatrix which is an
    internal data structure that is used by XGBoost which is optimized for
    both memory efficiency and training speed

    :param bytes request:   containing the payload to supply to the predict method

    :return:                xgb.DMatrix containing a 1 X 784 matrix of pixels
    """
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    nda = (np.array(request_json['image'], dtype=np.uint8) / 255.0).reshape(1, 784)
    return xgb.DMatrix(nda)


def _transform_response(response: np.ndarray) -> str:
    """
    Transform response from a 1D array to a list of classes and probabilities

    :param np.ndarray response:     np.ndarray containing the model prediction output

    :return:                        Response obj serialized to a JSON formatted str
    """
    # get prediction, this is in 1D array, need reshape to (ndata, nclass)
    probabilities = response.reshape(response.shape[0], response.shape[1])

    return json.dumps({
        'classes': np.argmax(probabilities, axis=1).tolist(),
        'probabilities': probabilities.tolist(),
    })


if __name__ == '__main__':
    with open('./pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

#!/usr/bin/env python3
# --- Builtin imports ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import asyncio
import glob
import json
import logging
from logging import Logger
import os
from typing import Callable

# --- 3rd-party imports ----------------------------------------------------------------------------
import numpy as np
from sklearn.externals import joblib
import xgboost as xgb

# --- PipelineAI imports ---------------------------------------------------------------------------
from pipeline_monitor import prometheus_monitor as monitor

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
           'model_name': 'mnist',
           'model_tag': 'async',
           'model_type': 'xgboost',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


class log(object):
    """
    Async logging decorator
    """

    def __init__(
        self,
        labels: dict,
        logger: Logger,
        custom_inputs_fn: Callable=None,
        custom_outputs_fn: Callable=None
    ):

        self._labels = labels
        self._logger = logger
        self._custom_inputs_fn = custom_inputs_fn
        self._custom_outputs_fn = custom_outputs_fn

    def __call__(self, fn):

        async def wrapped_function(*args: bytes):
            log_dict = {
                'log_labels': self._labels,
                'log_inputs': str(args),
            }

            if self._custom_inputs_fn:
                custom_inputs = self._custom_inputs_fn(*args),
                log_dict['log_custom_inputs'] = custom_inputs

            outputs = await fn(*args)

            log_dict['log_outputs'] = outputs

            if self._custom_outputs_fn:
                custom_outputs = self._custom_outputs_fn(outputs)
                log_dict['log_custom_outputs'] = custom_outputs

            self._logger.info(json.dumps(log_dict))

            return outputs

        return wrapped_function


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


async def main():
    with open('./pipeline_test_request.json', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = await invoke(request_bytes)
        return response_bytes


@log(labels=_labels, logger=_logger)
async def invoke(request: bytes) -> str:
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
        transformed_request = await _transform_request(request)

    with monitor(labels=_labels, name='invoke'):
        response = _model.predict(transformed_request)

    with monitor(labels=_labels, name='transform_response'):
        transformed_response = await _transform_response(response)

    return transformed_response


async def _transform_request(request: bytes) -> xgb.DMatrix:
    """
    Transform bytes posted to the api into an XGBoost DMatrix which is an
    internal data structure that is used by XGBoost which is optimized for
    both memory efficiency and training speed

    :param bytes request:   containing the payload to supply to the predict method

    :return:                xgb.DMatrix containing a 1 X 784 matrix of pixels
    """
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    nda = np.array(request_json['image'], dtype=np.uint8).reshape(1, 784)
    dnda = xgb.DMatrix(nda)
    return dnda


async def _transform_response(response: np.ndarray) -> str:
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
    """ multiprocessing wants the fork to happen in a __main__ protected block """
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

#!/usr/bin/env python3
# ------------- Builtin imports --------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import asyncio
from datetime import datetime
import json
import logging
import os

# ------------- 3rd-party imports ------------------------------------------------------------------
from sklearn.externals import joblib

# ------------- 3rd-party imports ------------------------------------------------------------------
from pipeline_autorouter import Model

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


async def test_model(request: bytes):
    """Main function.

    This function orchestrates the process of loading data,
    initializing the model, training the model, and evaluating the
    result.

    :param bytes request:   request payload containing existing routes by tag and weight

    """

    cmd_args, data, model = await gather_async_results(request)

    # model = await train(model, data, cmd_args)
    await evaluate(model, data, cmd_args)
    # save the models for later
    await _pickle_artifact(model, cmd_args)


async def gather_async_results(request: bytes) -> (argparse.Namespace, dict, Model):
    """

    :param bytes request:   request payload containing existing routes by tag and weight

    :return:
    """

    cmd_args = None
    data = None
    model = None

    tasks = [
        asyncio.ensure_future(parse_args()),
        asyncio.ensure_future(init_data(request)),
        asyncio.ensure_future(init_model())
    ]

    # passing `return_exceptions=True` will return the Exception object if produced
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # iterate the results
    for result in results:
        # manually check the types
        if isinstance(result, Exception):
            _logger.error('pipeline_train.main.Exception', exc_info=True)
        elif isinstance(result, argparse.Namespace):
            cmd_args = result
        elif isinstance(result, dict):
            data = result
        elif isinstance(result, Model):
            model = result
        else:
            _logger.warning('pipeline_train.main.Unexpected_Result_Type: {}'.format(type(result)))

    return cmd_args, data, model


async def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    :return:    argparse.Namespace: An object to take the attributes.
                    The default is a new empty Namespace object.
    """
    # create id to uniquely identify this training session
    training_run_datetime = datetime.today().strftime('%Y%m%d%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_datetime', type=str, default=training_run_datetime)
    parser.add_argument('--model_dir', type=str, default='.')
    cmd_args = parser.parse_args()
    return cmd_args


async def init_data(request: bytes) -> dict:
    """Initialize data for training and evaluation.

    :param bytes request:   request payload containing existing routes by tag and weight

    :return:                dict containing cost by deployment target
    """
    data = dict(json.loads(request.decode('utf-8'))['resource_split_tag_and_weight_dict'])
    return data


async def init_model() -> Model:
    """Initialize the model """
    model = Model()
    return model


async def train(model: Model, data: dict, args: argparse.Namespace) -> Model:
    """Train the model.

    :param Model model:                 Route cost model
    :param dict data:             Cost data
    :param argparse.Namespace args:     An object to take the attributes
                                            The default is a new empty Namespace object

    :return:                            Model: trained mnist model
    """
    return model


async def evaluate(model: Model, data: dict, args: argparse.Namespace):
    """
    Evaluate model results

    :param Model model:                 Trained model
    :param dict data:                   Cost data by deployment target
    :param argparse.Namespace args:     An object to take the attributes
                                            The default is a new empty Namespace object

    :return:                            None
    """
    autoroutes = model.predict(data)
    _logger.info('autoroutes: {}'.format(autoroutes))


async def _pickle_artifact(model: Model, args: argparse.Namespace) -> str:
    """
    Save the model to disk as a bz2 compressed pickled binary artifact.

    :param Model model:                 Trained model
    :param argparse.Namespace args:     An object to take the attributes
                                            The default is a new empty Namespace object

    :return:                            str path to the pickled binary artifact
    """
    compressor = 'bz2'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}_model.pkl.{}'.format(
        args.train_datetime,
        compressor
    ))

    with open(path, 'wb') as f:
        joblib.dump(model, f, compress=(compressor, 3))

    _logger.info('saved model: %s' % path)
    return path


if __name__ == '__main__':
    """ multiprocessing wants the fork to happen in a __main__ protected block """
    with open('test_request.json', 'rb') as fb:
        request_bytes = fb.read()

    loop = asyncio.get_event_loop()
    loop.run_until_complete(test_model(request_bytes))
    loop.close()

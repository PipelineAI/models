import os
import json
import logging
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import cloudpickle as pickle

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']


_labels = {
           'model_name': 'xgboost',
           'model_tag': 'v1',
           'model_type': 'xgboost',
           'model_runtime': 'python',
           'model_chip': 'cpu',
          }


def _initialize_upon_import():
    model_pkl_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model.pkl')

    # Load pickled model from model directory
    with open(model_pkl_path, 'rb') as fh:
        restored_model = pickle.load(fh)

    return restored_model


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()

_sc = StandardScaler()

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
    request_df = pd.read_csv(request)
    request_std = _sc.fit_transform(request_df.values)
    requests_dmatrix= xgb.DMatrix(data=request_std)
    return request_dmatrix

#    request_str = request.decode('utf-8')
#    request_json = json.loads(request_str)
#    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0).reshape(1, 28, 28)
#    return {"image": request_np}


def _transform_response(response):
    # Convert from tf.tensor, np.array, etc. to bytes
    # return response

    return json.dumps({"classes": response['classes'].tolist(),
                       "probabilities": response['probabilities'].tolist(),
                      })


#test_df = pd.read_csv("../input/training/test.csv")
#sc = StandardScaler()
#test_std = sc.fit_transform(test_df.values)
#d_test = xgb.DMatrix(data=test_std)

# TODO: Add test
#print(y_pred)

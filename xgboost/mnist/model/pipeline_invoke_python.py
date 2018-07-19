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
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
   
    # TODO:  Convert from json above to whatever this xgboost model needs... 
    #request_df = pd.read_csv(request)
    #request_std = _sc.fit_transform(request_df.values)
    #requests_dmatrix= xgb.DMatrix(data=request_std)
    #return request_dmatrix

    pass

def _transform_response(response):
    # Convert from xgboost to expect output json 
    # 
    #return json.dumps({"classes": response['classes'].tolist(),
    #                   "probabilities": response['probabilities'].tolist(),
    #                  })
    pass

# TODO:  Mini-test
#test_df = pd.read_csv("../input/training/test.csv")
#sc = StandardScaler()
#test_std = sc.fit_transform(test_df.values)
#d_test = xgb.DMatrix(data=test_std)
#print(y_pred)

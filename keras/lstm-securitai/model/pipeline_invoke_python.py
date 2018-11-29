import io
import os
import numpy as np
import pandas
import json
import logging                                                 #<== Optional.  Log to console, file, kafka
from pipeline_monitor import prometheus_monitor as monitor     #<== Optional.  Monitor runtime metrics
from pipeline_logger import log

import tensorflow as tf
from tensorflow.contrib import predictor
from keras.models import Sequential, load_model
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from collections import OrderedDict

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

__all__ = ['invoke']                                           #<== Optional.  Being a good Python citizen.

_labels = {                                                    #<== Optional.  Used for metrics/labels
           'name': 'injection',
           'tag': 'v1',
           'type': 'tensorflow',
           'runtime': 'python',
           'chip': 'cpu',
          }


def _initialize_upon_import():                                 #<== Optional.  Called once upon server startup
    ''' Initialize / Restore Model Object.
    '''
    model = load_model('securitai-lstm-model.h5')
    model.load_weights('securitai-lstm-weights.h5')
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


#@log(labels=_labels, logger=_logger)                           #<== Optional.  Sample and compare predictions
def invoke(request):                                           #<== Required.  Called on every prediction
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):    #<== Optional.  Expose fine-grained metrics
        transformed_request = _transform_request(request)      #<== Optional.  Transform input (json) into TensorFlow (tensor)

    with monitor(labels=_labels, name="invoke"):               #<== Optional.  Calls _model.predict()
        response = _model.predict(transformed_request)

    with monitor(labels=_labels, name="transform_response"):   #<== Optional.  Transform TensorFlow (tensor) into output (json)
        transformed_response = _transform_response(response)

    return transformed_response                                #<== Required.  Returns the predicted value(s)


def _transform_request(request):
    request_str = request.decode('utf-8')

    # tokenize the csv request and create json
    X = pandas.read_csv(io.StringIO(request_str), engine='python', quotechar='|', header=None).values[:,0]
    for index, item in enumerate(X):
        reqJson = json.loads(item, object_pairs_hook=OrderedDict)
        del reqJson['http']['timestamp']
        del reqJson['http']['headers']
        del reqJson['http']['source']
        del reqJson['http']['route']
        del reqJson['http']['responsePayload']
        X[index] = json.dumps(reqJson, separators=(',', ':'))

    tokenizer = Tokenizer(filters='\t\n', char_level=True)
    tokenizer.fit_on_texts(X)
    # this used to be [log_entry]
    seq = tokenizer.texts_to_sequences([request_str])
    max_log_length = 1024
    log_entry_processed = sequence.pad_sequences(seq, maxlen=max_log_length)

    return log_entry_processed 


def _transform_response(response):
    return response[0]


if __name__ == '__main__':
    with open('./pipeline_test_request.csv', 'rb') as fb:
        request_bytes = fb.read()
        response_bytes = invoke(request_bytes)
        print(response_bytes)

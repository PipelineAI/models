import os
import numpy as np
import json
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from pipeline_monitor import prometheus_monitor as monitor
from pipeline_logger import log

_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)


__all__ = ['invoke']


_labels = {
           'model_name': 'mnist',
           'model_tag': 'v1', 
           'model_type': 'pytorch', 
           'model_runtime': 'python', 
           'model_chip': 'cpu',
          }


# TODO:  This is a COPY/PASTE FROM pipeline_train.py until this is refactored
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def _initialize_upon_import():
    ''' Initialize / Restore Model Object.
    '''
    saved_model_path = './model.pth' 
    model = Net()
    model.load_state_dict(torch.load(saved_model_path))
    print('Loaded model from "%s": %s' % (saved_model_path, model))
    return model


# This is called unconditionally at *module import time*...
_model = _initialize_upon_import()


@log(labels=_labels, logger=_logger)
def invoke(request):
    '''Where the magic happens...'''

    with monitor(labels=_labels, name="transform_request"):
        transformed_request = _transform_request(request)

    with monitor(labels=_labels, name="invoke"):
        response = _model(transformed_request)

    with monitor(labels=_labels, name="transform_response"):
        transformed_response = _transform_response(response)

    return transformed_response


def _transform_request(request):
    request_str = request.decode('utf-8')
    request_json = json.loads(request_str)
    request_np = ((255 - np.array(request_json['image'], dtype=np.uint8)) / 255.0)
    request_np = request_np.reshape(1,1,28,28)

    request_tensor = torch.from_numpy(request_np).float()
    return Variable(request_tensor, volatile=True)


def _transform_response(response):
    print(response)
    response_np = response.data.numpy().tolist()[0]
    print(response_np)
    return json.dumps({"outputs": response_np})

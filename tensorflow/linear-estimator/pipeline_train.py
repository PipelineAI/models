import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
import tensorflow as tf
import argparse
from datetime import datetime 
import os


def main():

    # Show debugging output
    #tf.logging.set_verbosity(tf.logging.DEBUG)
    argContainer = get_commandline_args()
    dataList = generate_data(argContainer)
    create_estimator(dataList,argContainer)

def generate_data(argContainer):
    dataList = [] 
    flags, detail = argContainer.parse_known_args()
    num_samples = flags.num_samples
    x_train = np.random.rand(num_samples).astype(np.float32)
    dataList.append(x_train)
    noise = np.random.normal(scale=0.01, size=len(x_train)) 
    y_train = (0.1 * x_train) + 0.3 + noise
    dataList.append(y_train)
    x_test = np.random.rand(len(x_train)).astype(np.float32)
    dataList.append(x_test)
    y_test = (0.1 * x_test) + 0.3 + noise
    dataList.append(y_test)
    return dataList

def get_commandline_args():
   parser = argparse.ArgumentParser()
   parser.add_argument('--learning_rate', type=float, help='Learning Rate', default=0.025)  
   parser.add_argument('--num_samples', type=int, help='Num Samples', default=100000)  
   #print (parser.num_samples)
   version = int(datetime.now().strftime("%s"))
   parser.add_argument('--rundir', type=str, help='Run Directory', default='runs/%s' %version)  
   return parser


def train_input_func(dataList):
    x_train = dataList[0]
    y_train = dataList[1]
    return tf.estimator.inputs.numpy_input_fn({'x' : x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

def eval_input_func(dataList):
    x_train = dataList[2]
    y_train = dataList[3]
    return tf.estimator.inputs.numpy_input_fn({'x' : x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)

def input_fn_predict(estimator):
    return tf.estimator.inputs.numpy_input_fn({'x' :np.random.rand(10).astype(np.float32)}, shuffle=False)


def create_estimator(dataList,argContainer):

    flags,details = argContainer.parse_known_args()

    feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]

    estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols, model_dir='./' + flags.rundir)

    estimator.train(input_fn=train_input_func(dataList), steps=1000)

    train_metrics = estimator.evaluate(input_fn=train_input_func(dataList), steps=1000)

    eval_metrics = estimator.evaluate(input_fn=eval_input_func(dataList), steps=1000)

    print("train metrics: {}".format(train_metrics))

    print("eval metrics: {}".format(eval_metrics))

    list(estimator.predict(input_fn=input_fn_predict(estimator)))

    predictions = []
    for x in estimator.predict(input_fn=input_fn_predict(estimator)):
        predictions.append(x['predictions'])
    print (predictions)

if __name__ == '__main__':

    main()


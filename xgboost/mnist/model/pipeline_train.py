#!/usr/bin/env python3
# ------------- Builtin imports --------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import asyncio
from datetime import datetime
import logging
import os
from time import time

# ------------- 3rd-party imports ------------------------------------------------------------------
import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
import xgboost as xgb


_logger = logging.getLogger('pipeline-logger')
_logger.setLevel(logging.INFO)
_logger_stream_handler = logging.StreamHandler()
_logger_stream_handler.setLevel(logging.INFO)
_logger.addHandler(_logger_stream_handler)

np.set_printoptions(
    precision=20,
    threshold=5,
    edgeitems=50,
    linewidth=25000,
    formatter={
        'int_kind': lambda i: '{:d}'.format(i),
        'float_kind': lambda f: '{:20f}'.format(f)
    },
    suppress=True  # suppress scientific notation
)

pd.set_option('precision', 20)

# ------------- XGBoost Parameters ----------------------------------------------------------------
# see:  http://xgboost.readthedocs.io/en/latest/parameter.html
#
# alpha         L1 regularization term on weights
#               Increasing this value will make model more conservative
#               default=0   alias: reg_alpha
ALPHA = 0
BOOSTER = 'gbtree'
# colsample_bytree          Subsample ratio of columns when constructing each tree
#                           Subsampling will occur once in every boosting iteration
#                           default=1
#                           range: (0,1]
COLSAMPLE_BYTREE = 1
# early_stopping_rounds     activates early stopping
#                           CV error needs to decrease at least every
#                           <EARLY_STOPPING_ROUNDS> round(s) to continue
#                           Last entry in evaluation history is the one from best iteration
EARLY_STOPPING_ROUNDS = 50
# eta           the training step size shrinkage for each iteration used in
#               update to prevents overfitting
#               eta shrinks the feature weights to make the boosting process more conservative
#               GBM works by starting with an initial estimate which is updated
#               using the output of each tree
#               The learning parameter controls the magnitude of this change in the estimates
#               Lower values are generally preferred as they make the model
#               robust to the specific characteristics of tree and thus allowing
#               it to generalize well
#               Lower values would require higher number of trees to model all the relations
#               and will be computationally expensive
#               Choose a relatively high learning rate to start testing and work your way down
#               learning rate of 0.1 is a good place to start
#               expected range is somewhere between 0.05 to 0.3
#               default=0.3
#               alias: learning_rate
#               range: [0,1]
ETA = 0.1
# eval_metric   Evaluation metrics for validation data
#               a default metric will be assigned according to objective
#                   rmse for regression
#                   error for classification
#                   mean average precision for ranking
#               User can add multiple evaluation metrics
#               Python users: remember to pass the metrics in as list of parameters
#               pairs instead of map
#               mmerror   Multiclass classification error rate
#                           calculated as #(wrong cases)/#(all cases)
#               mae       mean absolute error
#               rmse      root mean square error
EVAL_METRIC = 'merror'
# gamma         smaller values like 0.1-0.2 can be a good starting place
#               Minimum loss reduction required to make a further partition on a
#               leaf node of the tree
#               The larger gamma is, the more conservative the algorithm will be
GAMMA = 0
# lambda        L2 regularization term on weights
#               Increasing this value will make model more conservative
#               default=1   alias: reg_lambda
LAMBDA = 2
# max_depth     the maximum depth of a tree.
#               Used to control over-fitting as higher depth will allow model to
#               learn relations very specific to a particular sample
#               Should be tuned using CV
#               max_depth is usually between 3-10
#               4-6 is a good place to start
MAX_DEPTH = 8
# min_child_weight  usually between 3-10
MIN_CHILD_WEIGHT = 1
N_ESTIMATORS = 10
N_JOBS = -1
# num_boost_round   int Number of boosting iterations.
#                   set NUM_BOOST_ROUND higher to improve accuracy, example 600
NUM_BOOST_ROUND = 100
# num_class     number of classes that exist in this dataset
NUM_CLASS = 10  # 0 - 9
# objective     multi:softmax: set XGBoost to do multiclass classification using softmax objective
#               multi:softprob: same as softmax, but output a vector of ndata * nclass
#                   which can be further reshaped to ndata * nclass matrix
#                   the result contains predicted probability of each data point
#                   belonging to each class
#               IMPORTANT: for softmax and softprob you need to set num_class(number of classes)
OBJECTIVE = 'multi:softprob'
# predictor     type of predictor algorithm to use
#               Provides the same results but allows the use of GPU or CPU
#               cpu_predictor: Multicore CPU prediction algorithm
#               gpu_predictor: Prediction using GPU.
#               Default when tree_method is gpu_exact or gpu_hist
PREDICTOR = 'cpu_predictor'
# subsample     fraction of observations to be selected for each tree
#               Selection is done by random sampling
#               Values slightly less than 1 make the model robust by reducing the variance
#               subsample, colsample_bytree 0.8 is a good starting place
#               expected range is between 0.5-0.9
SUBSAMPLE = 1  # 0.8
# ------------- CUDA Accelerated Tree Construction Algorithms --------------------------------------
#
# Booster params
#   n_gpus          Multiple GPUs can be used with the gpu_hist tree method using the
#                       defaults to 1
#                       -1 = use all available GPUs
#   gpu_id          used to select device ordinal, gpu device order is
#                       mod(gpu_id + i) % n_visible_devices for i=0 to n_gpus-1
#   tree_method     set to one of the values below to enable CUDA Accelerated tree construction
#
# Tree Method Algorithms
#   gpu_exact       The standard XGBoost tree construction algorithm.
#                       Performs exact search for splits.
#                       Slower and uses considerably more memory than gpu_hist.
#   gpu_hist        Equivalent to the XGBoost fast histogram algorithm.
#                       Much faster and uses considerably less memory.
#                       NOTE: Will run very slowly on GPUs older than Pascal architecture.
#   hist            CPU algorithm
#   see: http://xgboost.readthedocs.io/en/latest/gpu/index.html
TREE_METHOD = 'hist'

# ------------- XGBoost binary files ---------------------------------------------------------------
TRAIN_CSV = '../input/training/train.csv'
DTRAIN_BUFFER = '../input/training/dtrain.buffer'
DTEST_BUFFER = '../input/training/dtest.buffer'


async def main():
    # create id to uniquely identify this training session
    training_run_datetime = datetime.today().strftime('%Y%m%d%H%M%S')

    await _save_xgboost_binary_buffers()

    # ------------- XGBoost binary files -----------------------------------------------------------
    dtrain = xgb.DMatrix(DTRAIN_BUFFER)
    dtest = xgb.DMatrix(DTEST_BUFFER)

    # Booster params.
    params = {
        'alpha': ALPHA,
        'booster': BOOSTER,
        'colsample_bytree': COLSAMPLE_BYTREE,
        'eval_metric': EVAL_METRIC,
        'eta': ETA,
        'lambda': LAMBDA,
        'max_depth': MAX_DEPTH,
        'n_estimators': N_ESTIMATORS,
        'n_jobs': N_JOBS,
        'num_class': NUM_CLASS,
        'objective': OBJECTIVE,
        'predictor': PREDICTOR,
        'subsample': SUBSAMPLE,
        'tree_method': TREE_METHOD
    }

    # ------------- training and testing -----------------------------------------------------------
    # List of items to be evaluated during training, this allows user to watch
    # performance on the validation set
    evals = [(dtrain, 'train'), (dtest, 'validation')]
    model = xgb.train(
        params=params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=evals,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS
    )
    y_pred = model.predict(dtest)
    _logger.info('y_pred.shape: {}'.format(y_pred.shape))

    # ------------- extract most confident predictions ---------------------------------------------
    # output is a vector of ndata * nclass, which can be further reshaped to ndata * nclass matrix
    # probabilities contains predicted probability of each data point belonging to each class
    probabilities = y_pred.reshape(y_pred.shape[0], y_pred.shape[1])
    # classes is an array of the most confident classification predictions
    classes = np.argmax(probabilities, axis=1).tolist()

    y_pred_precision_score = precision_score(dtest.get_label(), classes, average='macro')
    _logger.info('y_pred_precision_score: %s' % y_pred_precision_score)

    # save the models for later
    await _pickle_artifact(model, training_run_datetime)

    await _cross_validate_results(params, dtrain)


async def _save_xgboost_binary_buffers():
    """
    Save DMatrix into a XGBoost binary files to improve load performance
    """
    # if test and train data is already split into separate csv files load
    # them directly into DMatrix which is an internal data structure that
    # used by XGBoost which is optimized for both memory efficiency and training speed.
    #   label_column - specifies the index of the column containing the true label
    # dtrain = xgb.DMatrix('../input/training/train.csv?format=csv&label_column=0')
    # feature_names = dtrain.feature_names
    # feature_types = dtrain.feature_types
    # _logger.info('feature_names: {}'.format(feature_names))
    # _logger.info('feature_types: {}'.format(feature_types))

    # if not os.path.exists(DTRAIN_BUFFER) or not os.path.exists(DTEST_BUFFER):

    train_df = pd.read_csv(TRAIN_CSV, encoding='utf-8', dtype=np.float64)
    train_df.fillna(np.float64(0), inplace=True)

    X = train_df.values[:, 1:]
    y = train_df.values[:, 0]
    _logger.info('X.shape: {}'.format(X.shape))
    _logger.info('y.shape: {}'.format(y.shape))

    # train_size is automatically set to the complement of the test size.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=31, shuffle=True)

    _logger.info('X_train.shape: {}'.format(X_train.shape))
    _logger.info('X_test.shape: {}'.format(X_test.shape))
    _logger.info('y_train.shape: {}'.format(y_train.shape))
    _logger.info('y_test.shape: {}'.format(y_test.shape))

    # use DMatrix for xgboost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # Saving DMatrix into a XGBoost binary file will make loading faster
    dtrain.save_binary(DTRAIN_BUFFER)
    dtest.save_binary(DTEST_BUFFER)


async def _cross_validate_results(params: dict, dtrain: xgb.DMatrix):
    """
    Cross validate results, this will print result out as [iteration]  metric_name:mean_value

    :param params:
    :param dtrain:

    :return:
    """
    _logger.info('running cross validation')

    cv_result = xgb.cv(
        params,
        dtrain,
        num_boost_round=10,
        nfold=5,
        metrics={EVAL_METRIC},
        seed=0,
        callbacks=[
            xgb.callback.print_evaluation(show_stdv=False),
            xgb.callback.early_stop(3)
        ]
    )
    _logger.info('cv_result: %s' % cv_result)


async def _pickle_artifact(model, train_datetime: str) -> str:
    """
    Save the model to disk as a bz2 compressed pickled binary artifact.

    :param model:               The object to pickle.
    :param str train_datetime:  The date and time the training session that created the artifact
                                in the format: YmdHMS

    :return:                    str pickled binary artifact path
    """
    # dump the models into a text file
    model.dump_model('{}_dump.model.raw.txt'.format(train_datetime))

    compressor = 'bz2'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '{}_model.pkl.{}'.format(
        train_datetime,
        compressor
    ))

    with open(path, 'wb') as f:
        joblib.dump(model, f, compress=(compressor, 3))

    _logger.info('saved model: %s' % path)
    return path


if __name__ == '__main__':
    """ multiprocessing wants the fork to happen in a __main__ protected block """
    t = time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main())
    loop.close()

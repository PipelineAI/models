import numpy as np
import tensorflow as tf
import os
from tensorflow.python.estimator.export.export import build_raw_serving_input_receiver_fn
from tensorflow.python.estimator.export.export_output import PredictOutput

# Parameter initialization
PARAMS = {'learning_rate': 0.01}
TRAINING_BATCH_SIZE = 100
TRAINING_EPOCHS = None
TRAINING_SHUFFLE = True
EVAL_BATCH_SIZE = 4
EVAL_EPOCHS = 100
EVAL_SHUFFLE = False
DATA_DOWNSIZE = 0.001
TRAINING_FILE_NAME = 'ml_sort_train.csv'
EVAL_FILE_NAME = 'ml_sort_eval.csv'
SIGNATURE_NAME = "serving_default"
INPUT_TENSOR_PRICE = "price"
INPUT_TENSOR_INVENTORY = "inventory"
INPUT_TENSOR_SINGLE_PREDICT = "singlep"

def model_fn(features, labels, mode, params):
    w_price_weight = tf.Variable([.1], dtype=tf.float32)
    w_inv_confidence_weight = tf.Variable([.1], dtype=tf.float32)

    price_feature = features[INPUT_TENSOR_PRICE]
    inventory_confidence_feature = features[INPUT_TENSOR_INVENTORY]

    print(features)
    print(price_feature)
    print(inventory_confidence_feature)

    target_conversion_confidence_feature = labels

    conversion_confidence_linear_model = tf.add(
        tf.multiply(w_price_weight, price_feature),
        tf.multiply(w_inv_confidence_weight, inventory_confidence_feature)
    )
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            # predictions={'relevance': tf.reshape(conversion_confidence_linear_model, -1)},
            # export_outputs={SIGNATURE_NAME: PredictOutput({"relevance": tf.reshape(conversion_confidence_linear_model, -1)})}
            predictions={'relevance': conversion_confidence_linear_model},
            export_outputs={SIGNATURE_NAME: PredictOutput({"relevance": conversion_confidence_linear_model})}
        )

    loss = tf.reduce_sum(tf.square(conversion_confidence_linear_model - target_conversion_confidence_feature))

    global_step = tf.train.get_global_step()
    learning_rate = PARAMS['learning_rate']
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train = tf.group(
        optimizer.minimize(loss),
        tf.assign_add(global_step, 1)
    )

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=conversion_confidence_linear_model,
        loss=loss,
        train_op=train
    )


def train_input_fn(training_dir, hyperparameters):
    """
    :param training_dir: The directory where the training CSV is located
    :param hyperparameters: A parameter set of the form
    {
        'batch_size': TRAINING_BATCH_SIZE,
        'num_epochs': TRAINING_EPOCHS,
        'data_downsize': DATA_DOWNSIZE
    }
    :return: A numpy_input_fn for the run
    """
    return _input_fn(
        training_dir,
        TRAINING_FILE_NAME,
        {
            'batch_size': TRAINING_BATCH_SIZE,
            'num_epochs': TRAINING_EPOCHS,
            'data_downsize': DATA_DOWNSIZE,
            'shuffle': TRAINING_SHUFFLE
        }
    )

def eval_input_fn(training_dir, hyperparameters):
    """
    :param training_dir: The directory where the training CSV is located
    :param hyperparameters: A parameter set of the form
    {
        'batch_size': TRAINING_BATCH_SIZE,
        'num_epochs': TRAINING_EPOCHS,
        'data_downsize': DATA_DOWNSIZE
    }
    :return: A numpy_input_fn for the run
    """
    return _input_fn(
        training_dir,
        EVAL_FILE_NAME,
        {
            'batch_size': EVAL_BATCH_SIZE,
            'num_epochs': EVAL_EPOCHS,
            'data_downsize': DATA_DOWNSIZE,
            'shuffle': EVAL_SHUFFLE
        }
    )


def eval_on_train_data_input_fn(training_dir, hyperparameters):
    """
    :param training_dir: The directory where the training CSV is located
    :param hyperparameters: A parameter set of the form
    {
        'batch_size': TRAINING_BATCH_SIZE,
        'num_epochs': TRAINING_EPOCHS,
        'data_downsize': DATA_DOWNSIZE
    }
    :return: A numpy_input_fn for the run
    """
    return _input_fn(
        training_dir,
        'ml_sort_train.csv',
        {
            'batch_size': EVAL_BATCH_SIZE,
            'num_epochs': EVAL_EPOCHS,
            'data_downsize': DATA_DOWNSIZE,
            'shuffle': EVAL_SHUFFLE
        }
    )


# Assumes that the data set is already sized down and appropriate
def _input_fn(training_dir, training_filename, params):
    """
    :param training_dir:
    :param training_filename:
    :param params:
    :return:
    """
    print('Reading from '+training_dir+' and '+training_filename)
    training_set = tf.contrib.learn.datasets.base.load_csv_without_header(
        filename=os.path.join(training_dir, training_filename),
        target_dtype=np.float32,
        features_dtype=np.float32)

    batch_size = params['batch_size']
    num_epochs = params['num_epochs']
    data_downsize = params['data_downsize']
    shuffle = params['shuffle']
    print(training_set.data)
    downed = np.asarray(training_set.data) * data_downsize
    features = {INPUT_TENSOR_PRICE: np.take(downed, 0, 1), INPUT_TENSOR_INVENTORY: np.take(downed, 1, 1)}
    labels = np.asarray(training_set.target) * data_downsize

    print(features)
    print(labels)

    return tf.estimator.inputs.numpy_input_fn(
        features,
        labels,
        # batch_size=batch_size,
        # num_epochs=num_epochs,
        num_epochs=None,
        shuffle=shuffle,
    )


def serving_input_fn(hyperparameters):
    feature_spec = {
        INPUT_TENSOR_PRICE: tf.placeholder(tf.float32, shape=[1]),
        INPUT_TENSOR_INVENTORY: tf.placeholder(tf.float32, shape=[1])
    }

    # These should have () after them? I don't know. Most example for input_fn show it, but my simple example
    # doesn't work with it.

    # Used in RAW for parse_input
    return build_raw_serving_input_receiver_fn(feature_spec)

    # Used in IRIS example
    # return tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec)()


# Only use estimator_fn in this scenario if you are using a pre-canned estimator
# Only use predictor_input_fn in non-hosted

                                                 	

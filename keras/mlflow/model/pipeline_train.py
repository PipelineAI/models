"""
Example of image classification with MLflow using Keras to classify flowers from photos. The data is
taken from ``http://download.tensorflow.org/example_images/flower_photos.tgz`` and may be
downloaded during running this project if it is missing.
"""
import math
import os
import time 

import click
import keras
from keras.utils import np_utils
from keras.models import Model
from keras.callbacks import Callback
from keras.applications import vgg16
from keras.layers import Input, Dense, Flatten, Lambda
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import mlflow

from image_pyfunc import decode_and_resize_image, log_model, KerasImageClassifierPyfunc

from datetime import datetime

import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

#from utils import plot_history
#import matplotlib.pyplot as plt
#
# See https://nbviewer.jupyter.org/github/WillKoehrsen/Data-Analysis/blob/master/slack_interaction/Interacting%20with%20Slack.ipynb for more details.
#
class SlackUpdate(Callback):

    """Custom Keras callback that posts to Slack while training a neural network"""

    def __init__(self, 
                 channel,
                 slack_webhook_url):

        self.channel = channel
        self.slack_webhook_url = slack_webhook_url 

    def file_upload(self,
                    path,
                    title):
        pass

    def report_stats(self, text):
        """Report training stats"""

        import subprocess
        try:
            cmd = 'curl -X POST --data-urlencode "payload={\\"unfurl_links\\": true, \\"channel\\": \\"%s\\", \\"username\\": \\"pipelineai_bot\\", \\"text\\": \\"%s\\"}" %s' % (self.channel, text, self.slack_webhook_url)

            response = subprocess.check_output(cmd, shell=True).decode('utf-8')
            return True
        except:
            return False

    def on_train_begin(self, logs={}):
        from timeit import default_timer as timer 
        self.report_stats(text=f'Training started at {datetime.now()}')

        self.start_time = timer()
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []
        self.n_epochs = 0

    def on_epoch_end(self, batch, logs={}):

        self.train_acc.append(logs.get('acc'))
        self.valid_acc.append(logs.get('val_acc'))
        self.train_loss.append(logs.get('loss'))
        self.valid_loss.append(logs.get('val_loss'))
        self.n_epochs += 1

        message = f'Epoch: {self.n_epochs} Training Loss: {self.train_loss[-1]:.4f} Validation Loss: {self.valid_loss[-1]:.4f}'

        self.report_stats(message)

    def on_train_end(self, logs={}):
        best_epoch = np.argmin(self.valid_loss)
        valid_loss = self.valid_loss[best_epoch]
        train_loss = self.train_loss[best_epoch]
        train_acc = self.train_acc[best_epoch]
        valid_acc = self.valid_acc[best_epoch]

        message = f'Trained for {self.n_epochs} epochs. Best epoch was {best_epoch + 1}.'
        self.report_stats(message)
        message = f'Best validation loss = {valid_loss:.4f} Training Loss = {train_loss:.2f} Validation accuracy = {100*valid_acc:.2f}%'
        self.report_stats(message)

    def on_train_batch_begin(self, batch, logs={}):
        pass

    def on_train_batch_end(self, batch, logs={}):
        pass


def download_input():
    import requests
    url = 'http://download.tensorflow.org/example_images/flower_photos.tgz'
    print("downloading '{}' into '{}'".format(url, os.path.abspath("flower_photos.tgz")))
    r = requests.get(url)
    with open('flower_photos.tgz', 'wb') as f:
        f.write(r.content)
    import tarfile
    print("decompressing flower_photos.tgz to '{}'".format(os.path.abspath("flower_photos")))
    with tarfile.open("flower_photos.tgz") as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, path="./")


@click.command(help="Trains an Keras model on flower_photos dataset."
                    "The input is expected as a directory tree with pictures for each category in a"
                    " folder named by the category."
                    "The model and its metrics are logged with mlflow.")
@click.option("--epochs", type=click.INT, default=1, help="Maximum number of epochs to evaluate.")
@click.option("--batch-size", type=click.INT, default=16,
              help="Batch size passed to the learning algo.")
@click.option("--image-width", type=click.INT, default=224, help="Input image width in pixels.")
@click.option("--image-height", type=click.INT, default=224, help="Input image height in pixels.")
@click.option("--seed", type=click.INT, default=97531, help="Seed for the random generator.")
@click.option("--training-data", type=click.STRING, default='./flower_photos')
@click.option("--test-ratio", type=click.FLOAT, default=0.2)
def run(training_data, test_ratio, epochs, batch_size, image_width, image_height, seed):
    image_files = []
    labels = []
    domain = {}
    print("Training model with the following parameters:")
    for param, value in locals().items():
        print("  ", param, "=", value)

    if training_data == "./flower_photos" and not os.path.exists(training_data):
        print("Input data not found, attempting to download the data from the web.")
        download_input()

    for (dirname, _, files) in os.walk(training_data):
        for filename in files:
            if filename.endswith("jpg"):
                image_files.append(os.path.join(dirname, filename))
                clazz = os.path.basename(dirname)
                if clazz not in domain:
                    domain[clazz] = len(domain)
                labels.append(domain[clazz])

    train(image_files, labels, domain,
          epochs=epochs,
          test_ratio=test_ratio,
          batch_size=batch_size,
          image_width=image_width,
          image_height=image_height,
          seed=seed)


class MLflowLogger(Callback):
    """
    Keras callback for logging metrics and final model with MLflow.

    Metrics are logged after every epoch. The logger keeps track of the best model based on the
    validation metric. At the end of the training, the best model is logged with MLflow.
    """
    def __init__(self, model, x_train, y_train, x_valid, y_valid,
                 **kwargs):
        self._model = model
        self._best_val_loss = math.inf
        self._train = (x_train, y_train)
        self._valid = (x_valid, y_valid)
        self._pyfunc_params = kwargs
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        """
        Log Keras metrics with MLflow. Update the best model if the model improved on the validation
        data.
        """
        if not logs:
            return
        for name, value in logs.items():
            if name.startswith("val_"):
                name = "valid_" + name[4:]
            else:
                name = "train_" + name
            mlflow.log_metric(name, value)
        val_loss = logs["val_loss"]
        if val_loss < self._best_val_loss:
            # Save the "best" weights
            self._best_val_loss = val_loss
            self._best_weights = [x.copy() for x in self._model.get_weights()]

    def on_train_end(self, *args, **kwargs):
        """
        Log the best model with MLflow and evaluate it on the train and validation data so that the
        metrics stored with MLflow reflect the logged model.
        """
        self._model.set_weights(self._best_weights)
        x, y = self._train
        train_res = self._model.evaluate(x=x, y=y)
        for name, value in zip(self._model.metrics_names, train_res):
            mlflow.log_metric("train_{}".format(name), value)
        x, y = self._valid
        valid_res = self._model.evaluate(x=x, y=y)
        for name, value in zip(self._model.metrics_names, valid_res):
            mlflow.log_metric("valid_{}".format(name), value)
        log_model(keras_model=self._model, **self._pyfunc_params)

    def on_train_batch_begin(self, batch, logs={}):
        pass

    def on_train_batch_end(self, batch, logs={}):
        pass


def _imagenet_preprocess_tf(x):
    return (x / 127.5) - 1


def _create_model(input_shape, classes):
    image = Input(input_shape)
    lambda_layer = Lambda(_imagenet_preprocess_tf)
    preprocessed_image = lambda_layer(image)
    model = vgg16.VGG16(classes=classes,
                        input_tensor=preprocessed_image,
                        weights=None,
                        include_top=False)

    x = Flatten(name='flatten')(model.output)
    x = Dense(4096, activation='relu', name='fc1')(x)
    x = Dense(4096, activation='relu', name='fc2')(x)
    x = Dense(classes, activation='softmax', name='predictions')(x)
    return Model(inputs=model.input, outputs=x)


def train(image_files,
          labels,
          domain,
          image_width=224,
          image_height=224,
          epochs=1,
          batch_size=16,
          test_ratio=0.2,
          seed=None):
    """
    Train VGG16 model on provided image files. This will create a new MLflow run and log all
    parameters, metrics and the resulting model with MLflow. The resulting model is an instance
    of KerasImageClassifierPyfunc - a custom python function model that embeds all necessary
    preprocessing together with the VGG16 Keras model. The resulting model can be applied
    directly to image base64 encoded image data.

    :param image_height: Height of the input image in pixels.
    :param image_width: Width of the input image in pixels.
    :param image_files: List of image files to be used for training.
    :param labels: List of labels for the image files.
    :param domain: Dictionary representing the domain of the reponse.
                   Provides mapping label-name -> label-id.
    :param epochs: Number of epochs to train the model for.
    :param batch_size: Batch size used during training.
    :param test_ratio: Fraction of dataset to be used for validation. This data will not be used
                       during training.
    :param seed: Random seed. Used e.g. when splitting the dataset into train / validation.
    """
    assert len(set(labels)) == len(domain)

    input_shape = (image_width, image_height, 3)

    #mlflow.set_tracking_uri('http://mlflow-tracking-host:port')

    # This will create and set the experiment
    mlflow.set_experiment(str(int(time.time()))[2:] + 'flower-v1')

    with mlflow.start_run() as run:
        mlflow.log_param("epochs", str(epochs))
        mlflow.log_param("batch_size", str(batch_size))
        mlflow.log_param("validation_ratio", str(test_ratio))
        if seed:
            mlflow.log_param("seed", str(seed))

        def _read_image(filename):
            with open(filename, "rb") as f:
                return f.read()

        with tf.Graph().as_default() as g:
            with tf.Session(graph=g).as_default():
                dims = input_shape[:2]
                x = np.array([decode_and_resize_image(_read_image(x), dims)
                              for x in image_files])
                y = np_utils.to_categorical(np.array(labels), num_classes=len(domain))
                train_size = 1 - test_ratio
                x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=seed,
                                                                      train_size=train_size)
                model = _create_model(input_shape=input_shape, classes=len(domain))
                model.compile(
                    optimizer=keras.optimizers.SGD(decay=1e-5, nesterov=True, momentum=.9),
                    loss=keras.losses.categorical_crossentropy,
                    metrics=["accuracy"])
                sorted_domain = sorted(domain.keys(), key=lambda x: domain[x])

                slack_update = SlackUpdate(channel='#slack-after-dark',
                                           slack_webhook_url='https://hooks.slack.com/services/T/B/G')

                history = model.fit(
                    x=x_train,
                    y=y_train,
                    validation_data=(x_valid, y_valid),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[MLflowLogger(model=model,
                                            x_train=x_train,
                                            y_train=y_train,
                                            x_valid=x_valid,
                                            y_valid=y_valid,
                                            artifact_path="model",
                                            domain=sorted_domain,
                                            image_dims=input_shape),
                                           slack_update])

                # From the following:  https://keras.io/visualization/

                # Plot training & validation accuracy values
                plt.plot(history.history['acc'])
                plt.plot(history.history['val_acc'])
                plt.title('Model accuracy')
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()
                plt.savefig('training_accuracy.png')

                # Plot training & validation loss values
                plt.plot(history.history['loss'])
                plt.plot(history.history['val_loss'])
                plt.title('Model loss')
                plt.ylabel('Loss')
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Test'], loc='upper left')
                plt.show()

                # plot_history(history.history)
                plt.savefig('training_loss.png')
                #slack_update.file_upload(path='training_charts.png',
                #                         title='Charts')
            
if __name__ == '__main__':
    run()

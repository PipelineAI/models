import tensorflow as tf

from keras.models import Model
from keras.callbacks import Callback
from keras.layers import Dense, Flatten
import mlflow
from datetime import datetime
import math
import numpy as np
from image_pyfunc import log_model

#
# See https://nbviewer.jupyter.org/github/WillKoehrsen/Data-Analysis/blob/master/slack_interaction/Interacting%20with%20Slack.ipynb for more details.
#
class SlackLogger(Callback):

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

class MLflowLogger(Callback):
    """
    Keras callback for logging metrics and final model with MLflow.

    Metrics are logged after every epoch. The logger keeps track of the best model based on the
    validation metric. At the end of the training, the best model is logged with MLflow.
    """
    def __init__(self, model, x_train, y_train, x_test, y_test,
                 **kwargs):
        self._model = model
        self._best_val_loss = math.inf
        self._train = (x_train, y_train)
        self._valid = (x_test, y_test)
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


import tensorflow as tf

#mlflow.set_tracking_uri('https://community.cloud.pipeline.ai/admin/tracking')

# This will create and set the experiment
#mlflow.set_experiment(str(int(time.time()))[2:] + 'flower-v1')
#mlflow.set_experiment('83f05e58mlflow-mnist')

with mlflow.start_run() as run:
    epochs = 10 
    batch_size = 256
    test_ratio = 0.2
    mlflow.log_param("epochs", str(epochs))
    mlflow.log_param("batch_size", str(batch_size))
    mlflow.log_param("test_ratio", str(test_ratio))

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation=tf.nn.relu),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(
                    x=x_train,
                    y=y_train,
                    validation_data=(x_test, y_test),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[MLflowLogger(model=model,
                                            x_train=x_train,
                                            y_train=y_train,
                                            x_test=x_test,
                                            y_test=y_test,
                                            domain={},
                                            artifact_path="model",
                                            image_dims=(28, 28)),
                               SlackLogger(channel='#slack-after-dark',
                                           slack_webhook_url='https://hooks.slack.com/services/T/B/G')
                              ])

    model.evaluate(x_test, y_test)

    saved_model_path = tf.contrib.saved_model.save_keras_model(model, "./pipeline_tfserving")
    print(saved_model_path)

#if __name__ == '__main__':
#    run()

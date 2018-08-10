import numpy as np
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, precision_score
import os.path
from scipy.io import loadmat
from modeldb.sklearn_native.ModelDbSyncer import *
from modeldb.sklearn_native import SyncableMetrics
from modeldb.basic.Structs import (NewOrExistingProject, ExistingProject,
     NewOrExistingExperiment, ExistingExperiment, DefaultExperiment,
     NewExperimentRun, ExistingExperimentRun, ThriftConfig, VersioningConfig,
     Dataset, ModelConfig, Model, ModelMetrics)

PATH = './model.pkl'

if __name__ == '__main__':
    syncer_obj = Syncer.create_syncer_from_config("./pipeline_syncer.json")

    print('Fetching and loading MNIST data')

    # Note:  This is the path within the Docker Container started with `train-server-start` and mounted to the host filesystem (your laptop or server) with --PIPELINE_INPUT_HOME (must add /training since only /opt/ml/input is mounted)
    mnist_path = os.path.join("./input/training", "mnist-original.mat")
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }

    print(mnist)

    x, y = mnist["data"], mnist["target"]
    x_train, x_test, y_train, y_test = cross_validation.train_test_split_sync(x / 255., y, test_size=0.25)

    print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
    print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))

    clf = None
    if os.path.exists(PATH):
        print('Loading model from file.')
        clf = joblib.load(PATH).best_estimator_
    else:
        print('Training model.')
        #params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
        params = {'hidden_layer_sizes': [(1,), (1,), (1, 1, 1,)]}
        mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
        clf = GridSearchCV(mlp, params, verbose=10, n_jobs=1, cv=2)
        clf.fit_sync(x_train, y_train)
        print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
        print('Best params appeared to be', clf.best_params_)
        joblib.dump(clf, PATH)
        y_pred = clf.predict_sync(x_test)
        score = SyncableMetrics.compute_metrics(
            clf, accuracy_score, y_test, y_pred, x_train, "", "")
        clf = clf.best_estimator_


    print('Test accuracy:', clf.score(x_test, y_test))

#    datasets = {
#        "train" : Dataset("/path/to/train", {"num_cols" : 15, "dist" : "random"}),
#        "test" : Dataset("/path/to/test", {"num_cols" : 15, "dist" : "gaussian"})
#    }
#    model = "model_obj"
#    model_type = "NN"
#    mdb_model1 = Model(model_type, model, "./model.pkl")
#    model_config1 = ModelConfig(model_type, {"l1" : 10})
#    model_metrics1 = ModelMetrics({"accuracy" : 0.8})
#    syncer_obj.sync_datasets(datasets)
#    syncer_obj.sync_model("train", model_config1, mdb_model1)
#    syncer_obj.sync_metrics("test", mdb_model1, model_metrics1)
    syncer_obj.sync()

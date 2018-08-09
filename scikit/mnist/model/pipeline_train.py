import numpy as np
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals import joblib
import os.path
from scipy.io import loadmat

# Note: This is the path within the Docker Container started with `train-server-start`and mounted on the host file system (your laptop or server) with `train-server-build` --model-path (basically same as the path with this source .py file, but i'm highlighting the full context)
PATH = 'model.pkl'

if __name__ == '__main__':
    print('Fetching and loading MNIST data')

    mnist_path = os.path.join("./input/training", "mnist-original.mat")
    mnist_raw = loadmat(mnist_path)
    mnist = {
        "data": mnist_raw["data"].T,
        "target": mnist_raw["label"][0],
        "COL_NAMES": ["label", "data"],
        "DESCR": "mldata.org dataset: mnist-original",
    }

    print(mnist)

    X, y = mnist["data"], mnist["target"]
    X_train, X_test, y_train, y_test = train_test_split(X / 255., y, test_size=0.25)

    print('Got MNIST with %d training- and %d test samples' % (len(y_train), len(y_test)))
    print('Digit distribution in whole dataset:', np.bincount(y.astype('int64')))

    clf = None
    if os.path.exists(PATH):
        print('Loading model from file.')
        clf = joblib.load(PATH).best_estimator_
    else:
        print('Training model.')
        params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
        mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
        clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=5)
        clf.fit(X_train, y_train)
        print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
        print('Best params appeared to be', clf.best_params_)
        joblib.dump(clf, PATH)
        clf = clf.best_estimator_

    print('Test accuracy:', clf.score(X_test, y_test))

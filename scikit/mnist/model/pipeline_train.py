import numpy as np
from scipy.ndimage import convolve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.externals import joblib
from os import path, environ as env
from scipy.io import loadmat
import requests

model_path = './model.pkl'

if __name__ == '__main__':
    print('Fetching and loading MNIST data')

    url = 'https://s3.amazonaws.com/datapalooza/mnist/mnist-original.mat'
    request = requests.get(url, allow_redirects=True)

    mnist_path = './mnist-original.mat'
    open(mnist_path, 'wb').write(request.content)

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
    if path.exists(model_path):
        print('Loading model from file.')
        clf = joblib.load(model_path).best_estimator_
    else:
        print('Training model.')
        params = {'hidden_layer_sizes': [(256,), (512,), (128, 256, 128,)]}
        mlp = MLPClassifier(verbose=10, learning_rate='adaptive')
        clf = GridSearchCV(mlp, params, verbose=10, n_jobs=-1, cv=2)
        clf.fit(X_train, y_train)
        print('Finished with grid search with best mean cross-validated score:', clf.best_score_)
        print('Best params appeared to be', clf.best_params_)
        joblib.dump(clf, model_path)
        clf = clf.best_estimator_

    print('Test accuracy:', clf.score(X_test, y_test))

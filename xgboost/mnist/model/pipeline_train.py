# Derived from https://www.kaggle.com/anktplwl91/mnist-xgboost

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from subprocess import check_output

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import pandas as pd
import xgboost as xgb
import cloudpickle as pickle

train_df = pd.read_csv("../input/training/train.csv")
test_df = pd.read_csv("../input/training/test.csv")

sc = StandardScaler()
X_std = sc.fit_transform(train_df.values[:, 1:])
y = train_df.values[:, 0]

test_std = sc.fit_transform(test_df.values)

X_train, X_valid, y_train, y_valid = train_test_split(X_std, y, test_size=0.1)

param_list = [("eta", 0.08), ("max_depth", 6), ("subsample", 0.8), ("colsample_bytree", 0.8), ("objective", "multi:softmax"), ("eval_metric", "merror"), ("alpha", 8), ("lambda", 2), ("num_class", 10)]
n_rounds = 10
early_stopping = 50
    
d_train = xgb.DMatrix(X_train, label=y_train)
d_val = xgb.DMatrix(X_valid, label=y_valid)
eval_list = [(d_train, "train"), (d_val, "validation")]
bst = xgb.train(param_list, d_train, n_rounds, evals=eval_list, early_stopping_rounds=early_stopping, verbose_eval=True)

# TODO:  Pickle trained model `bst`
model_pkl_path = 'model.pkl'
print("Exporting saved model...")
with open(model_pkl_path, 'wb') as fh:
    pickle.dump(model, fh)

d_test = xgb.DMatrix(data=test_std)
y_pred = bst.predict(d_test)

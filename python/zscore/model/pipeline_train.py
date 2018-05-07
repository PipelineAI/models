import argparse
import json
import pickle

def init_flags():
    global FLAGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/tmp/MNIST_data",)
    parser.add_argument("--run-dir", default="/tmp/MNIST_train")
    parser.add_argument("--batch_size", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--prepare", dest='just_data', action="store_true")
    parser.add_argument("--test", action="store_true")
    FLAGS, _ = parser.parse_known_args()

def init_data():
    pass

def init_test():
    pass

def test():
    pass

def init_train():
    init_model()

def init_model():
    global cat_mean, cat_stdv, dog_mean, dog_stdv, model, model_pkl_path
    cat_mean = 0.1
    cat_stdv = 0.20
    dog_mean = 0.3
    dog_stdv = 0.40
    model_pkl_path = 'model.pkl'

def train():
    model = {'cat_mean':cat_mean,
             'cat_stdv':cat_stdv,
             'dog_mean':dog_mean,
             'dog_stdv':dog_stdv}

def export_saved_model():
    model = {'cat_mean':cat_mean,
             'cat_stdv':cat_stdv,
             'dog_mean':dog_mean,
             'dog_stdv':dog_stdv}
    model_pkl_path = 'model.pkl'
    print("Exporting saved model")
    with open(model_pkl_path, 'wb') as fh:
        pickle.dump(model, fh)

if __name__ == "__main__":
    init_flags()
    init_data()
    if FLAGS.just_data:
        pass
    elif FLAGS.test:
        init_test()
        test()
    else:
        init_train()
        train()
        export_saved_model()

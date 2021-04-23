import sys, os
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.svm import SVR

from sklearn.model_selection import train_test_split


def load_full_dataset(fname):
    full_dataset = np.load(fname)
    X = full_dataset.f.X
    y_avg = full_dataset.f.y_avg
    y_med = full_dataset.f.y_med
    return X, y_avg, y_med


def print_stats(model_name, n_train, n_test, R2_y_avg, R2_y_med):
    print('MODEL: ', model_name)
    print('Number of train: ', n_train)
    print('Number of test: ', n_test)
    print('R2 y_avg: ', R2_y_avg)
    print('R2_y_med: ', R2_y_med, '\n')


def main(fname):
    LR = ['Linear Regression', LinearRegression()]
    SGD = ['Stochastic Gradient Descent', SGDRegressor()]
    SVMR = ['Support Vector Regression', SVR()]
    models = [LR, SGD, SVMR]

    train_percent = .8  # percentage of examples for training; complement is for testing
    X, y_avg, y_med = load_full_dataset(fname)
    n_train = int(len(X) * train_percent)
    n_test = len(X) - n_train

    split_data = train_test_split(X, y_avg, y_med, train_size = train_percent)
    train_feats, test_feats, train_y_avg, test_y_avg, train_y_med, test_y_med = split_data

    for m in models:
        model_name = m[0]
        model = m[1]

        # fit to y_avg
        model.fit(train_feats, train_y_avg)
        preds_y_avg = model.predict(test_feats)
        score_y_avg = model.score(test_feats, test_y_avg)
        # fit to y_med
        model.fit(train_feats, train_y_med)
        preds_y_med = model.predict(test_feats)
        score_y_med = model.score(test_feats, test_y_med)

        print_stats(model_name, n_train, n_test, score_y_avg, score_y_med)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        # main(DEFAULT_DATASET)
        main('persistence_image_feature/full_dataset.npz')
        exit(0)
        # quit("Missing filename as command-line argument. e.g. 'python main.py ./data/Simple/two_ortho.csv'")
    filename = sys.argv[1]
    if not os.path.exists(filename):
        quit("Filename '{}' does not exist".format(filename))
    main(filename)
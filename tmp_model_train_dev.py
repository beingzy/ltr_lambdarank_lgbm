"""develop the training process of building LTR model

   author: Yi Zhang <beingzy@gmail.com>
   date: 2018/10/01
"""
import os

import numpy as np
import pandas as pd

import sklearn as sk
import lightgbm as lgb
from sklearn.datasets import load_svmlight_file

try:
    import cPickle as pickle
except BaseException:
    import pickle


config = {
    'data_dir': './data/lgb_examples_rank',
    'models': './models',
    'output': './output',
    'model_name': 'lgb_example_ranker'
}


def get_now_str():
    from datetime import datetime
    _now = datetime.now()
    _format = "%Y%m%d_%H%M%S"
    return datetime.strftime(_now, format=_format)


# prepare data 
def load_rank_data():
    """
    """
    X_train, y_train = load_svmlight_file(
        os.path.join(config['data_dir'], 'rank.train'))
    X_test, y_test = load_svmlight_file(
        os.path.join(config['data_dir'], 'rank.test'))
    q_train = np.loadtxt(os.path.join(config['data_dir'], 'rank.train.query'))
    q_test = np.loadtxt(os.path.join(config['data_dir'], 'rank.test.query'))

    return (X_train, y_train, q_train), (X_test, y_test, q_test)


def train_lgb_rank_model(ranker, train_data, test_data):
    """
    """
    X_train, y_train, q_train = train_data
    X_test, y_test, q_test = test_data

    ranker.fit(
        X_train, 
        y_train, 
        group=q_train, 
        eval_set=[(X_test, y_test)],
        eval_group=[q_test],
        eval_at=[1, 3],
        early_stopping_rounds=5,
        verbose=True,
        callbacks=[
            lgb.reset_parameter(learning_rate=lambda x: 0.9 ** x * 0.1)
        ],
    )
    
    return ranker


if __name__ == "__main__":
    train_data, test_data = load_rank_data()

    ranker = lgb.LGBMRanker(random_state=42, silent=False)
    ranker = train_lgb_rank_model(ranker, train_data, test_data)

    # save model
    model_name = "_".join([config['model_name'], get_now_str()])
    model_file = os.path.join(config['models'], model_name + '.pkl')

    with open(model_file, 'wb') as fout:
        pickle.dump(ranker, fout)

    # test
    test_saved_model = True
    if test_saved_model:
        try:
            with open(model_file, 'rb') as fin:
                pkl_bst_ranker = pickle.load(fin)
        except ValueError:
            print("failed to load pickled model.")

"""
Main script
"""
import logging
import argparse

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb

from utils import *
from hyperspace import LRSpace, SVMSpace, XGBSpace

ESTIMATORS = ['logistic', 'svm', 'xgboost']
PREPROCESSORS = []

parser = argparse.ArgumentParser()

parser.add_argument("algorithm", help="Algorithm, choose from 'xgboost', 'svm', 'logistic'.", type=str)
parser.add_argument("dataset", help="Dataset name.", type=str)
parser.add_argument("n_configs", help="Number of hyperparameter configurations to create", type=int)

args = parser.parse_args()

_logger = logging.getLogger()
_logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(levelname)s] [%(asctime)s, %(name)s] %(message)s',
                              datefmt='%m-%d-%Y %H:%M:%S')
handler.setFormatter(formatter)
_logger.addHandler(handler)


def run_pipeline(estimator_choice, dataset, n_configs):

    df = pd.read_csv("../datasets/%s_train.data" % dataset,
                     sep=" ",
                     header=None,
                     index_col=False)
    df.dropna(axis=1, how='all', inplace=True)
    target = pd.read_csv("../datasets/%s_train.solution" % dataset,
                         sep=" ",
                         header=None,
                         index_col=False)
    _logger.info("Read dataset ``%s.''" % dataset)
    _logger.info("Metric: F1 score")

    # Transformer

    X = df.values
    y = target.values.ravel()

    test_df = pd.read_csv("../datasets/%s_test.data" % dataset,
                          sep=" ",
                          header=None,
                          index_col=False)
    test_df.dropna(axis=1, how='all', inplace=True)
    test_target = pd.read_csv("../datasets/%s_test.solution" % dataset,
                              sep=" ",
                              header=None,
                              index_col=False)
    test_X = test_df.values
    test_y = np.asarray([x[0] for x in test_target.values])

    # preprocessor.fit_transform(X, y)
    if estimator_choice == 'logistic':
        config_space = LRSpace()
        Estimator = LogisticRegression
    elif estimator_choice == 'svm':
        config_space = SVMSpace()
        Estimator = SVC
    elif estimator_choice == 'xgboost':
        config_space = XGBSpace()
        Estimator = xgb.XGBClassifier
    else:
        raise ValueError("Estimator should be one of %s" % ESTIMATORS)
    _logger.info("Estimator: %s" % estimator_choice)

    configs = config_space.grid(n_configs)

    results_header = config_space.get_params()

    results_file = "../results/%s_%s." % (dataset, estimator_choice)
    pd.DataFrame(configs).to_csv(results_file + "data",
                                 sep=" ",
                                 header=results_header,
                                 index=False)

    results = open(results_file + "solution", 'w')
    results.write("score")

    for i, config_values in enumerate(configs):
        config = dict(zip(results_header, config_values))
        # config['n_jobs'] = -1
        _logger.info("Configuration #  %d:" % i)

        _logger.info("\tTraining with %s." % config)
        estimator = Estimator(**config)
        estimator.fit(X, y)
        _logger.info("\tDone fitting.")

        predictions = estimator.predict(test_X)
        score = f1_metric(test_y, predictions)
        results.write("%f" % score)
        _logger.info("\tScore: %f" % score)

    results.close()
    _logger.info("Results saved as %s*." % results_file)

if __name__ == "__main__":
    run_pipeline(args.algorithm, args.dataset, args.n_configs)
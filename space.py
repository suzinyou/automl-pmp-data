from collections import OrderedDict
from itertools import product
import math

import pandas as pd

from hyperparameter import NumericalHyperparam, CategoricalHyperparam


class LRSpace(object):
    """
    Hyperparameter dependencies:
    
    
    solver      | allowed penalty   
    --------------------------------
    'liblinear' | 'l1', 'l2'
    'newton-cg' | 'l2'
    'sag'       | 'l2'
    'lbfgs'     | 'l2'
    
    """

    def __init__(self):
        C = NumericalHyperparam('C', -2, 4, 'log', 'float')
        solver = CategoricalHyperparam('solver', ['liblinear', 'newton-cg', 'sag', 'lbfgs'])
        penalty = CategoricalHyperparam('penalty', ['l1', 'l2'])

        self.params = OrderedDict({'C': C,
                                   'solver': solver,
                                   'penalty': penalty})
        self.param2idx = {key: i for (i, key) in enumerate(self.params.keys())}

    def get_params(self):
        return self.params.keys()

    def _forbiddens(self, df):
        return (df[self.param2idx['penalty']] == 'l1') \
               & (df[self.param2idx['solver']].isin(['newton-cg', 'lbfgs', 'sag']))

    def grid(self, n):
        if n < 5:
            self.params['C'].set_grid(1)
        else:
            self.params['C'].set_grid(n // 5)

        product_space = list(product(*[param.grid() for param in self.params.values()]))
        configs = pd.DataFrame(product_space)
        configs = configs.drop(configs[self._forbiddens(configs)].index)

        return configs.values


class SVMSpace(object):
    def __init__(self):
        C = NumericalHyperparam('C', -3, 0, 'log', 'float')
        kernel = CategoricalHyperparam('kernel', ['linear', 'rbf', 'poly', 'sigmoid'])
        degree = NumericalHyperparam('degree', 1, 5, 'linear', 'int')
        gamma = NumericalHyperparam('gamma', -5, 1, 'log', 'float')

        self.params = OrderedDict({'C': C,
                                   'kernel': kernel,
                                   'degree': degree,
                                   'gamma': gamma})
        self.param2idx = {key: i for (i, key) in enumerate(self.params.keys())}

    def get_params(self):
        return self.params.keys()

    def _forbiddens(self, df):
        return ((df[self.param2idx['kernel']] == 'linear')
                & (df[self.param2idx['degree']].isin([2, 3, 4, 5])
                & (df[self.param2idx['gamma']] > self.params['gamma'].min))) | \
               ((df[self.param2idx['kernel']].isin(['rbf', 'sigmoid']))
                & (df[self.param2idx['degree']].isin([2, 3, 4, 5])))

    def grid(self, n):
        if n < 7:
            self.params['C'].set_grid(1)
            self.params['gamma'].set_grid(1)
            self.params['degree'].set_grid(1)
        else:
            grid_n = int(math.sqrt(n / 7.))
            self.params['C'].set_grid(grid_n)
            self.params['gamma'].set_grid(grid_n)
            self.params['degree'].set_grid(5)

        product_space = list(product(*[param.grid() for param in self.params.values()]))
        configs = pd.DataFrame(product_space)
        configs = configs.drop(configs[self._forbiddens(configs)].index)
        # [configs[x].values.tolist() for x in configs.columns]

        return configs.values


class XGBSpace(object):
    def __init__(self):
        max_depth = NumericalHyperparam('max_depth', 1, 10, 'linear', 'int')
        learning_rate = NumericalHyperparam('learning_rate', -2, 0, 'log', 'float')
        n_estimators = NumericalHyperparam('n_estimators', 50, 500, 'linear', 'int')
        subsample = NumericalHyperparam('subsample', 0.01, 1.0, 'linear', 'float')
        min_child_weight = NumericalHyperparam('min_child_weight', 1, 20, 'linear', 'int')
        colsample_bytree = NumericalHyperparam('colsample_bytree', 0.5, 1.0, 'linear', 'float')
        scale_pos_weight = NumericalHyperparam('scale_pos_weight', 0.5, 2.0, 'linear', 'float')

        self.params = OrderedDict({'max_depth': max_depth,
                                   'learning_rate': learning_rate,
                                   'n_estimators': n_estimators,
                                   'subsample': subsample,
                                   'min_child_weight': min_child_weight,
                                   'colsample_bytree': colsample_bytree,
                                   'scale_pos_weight': scale_pos_weight})

    def get_params(self):
        return self.params.keys()


    def grid(self, n):
        if n < 1000:
            self.params['max_depth'].set_grid(4)
            self.params['n_estimators'].set_grid(1)
            self.params['learning_rate'].set_grid(1)
            self.params['subsample'].set_grid(1)
            self.params['min_child_weight'].set_grid(1)
            self.params['colsample_bytree'].set_grid(1)
            self.params['scale_pos_weight'].set_grid(1)
        else:
            grid_n = int((n/50)**(1./5.))
            self.params['max_depth'].set_grid(5)
            self.params['n_estimators'].set_grid(10)
            self.params['learning_rate'].set_grid(grid_n)
            self.params['subsample'].set_grid(grid_n)
            self.params['min_child_weight'].set_grid(20)
            self.params['colsample_bytree'].set_grid(grid_n)
            self.params['scale_pos_weight'].set_grid(grid_n)

        configs = list(product(*[param.grid() for param in self.params.values()]))

        return configs

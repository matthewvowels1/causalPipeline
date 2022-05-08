
from __future__ import print_function
import numpy as np
from scipy.special import digamma
from sklearn.neighbors import NearestNeighbors
from pycit import citest
from pycit import itest
from scipy import stats
from GCIT.GCIT import *

def ksg_cmi(X, Y, Z, data, **kwargs):
    k = kwargs["knn"]
    sig = kwargs["significance_level"]
    """ Adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/ksg_cmi.py
        KSG Conditional Mutual Information Estimator: I(X;Y|Z)
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        data_matrix: all data
        x: data nodes
        y: data nodes
        s: conditioning nodes
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """

    x_data = np.array(data.loc[:, X])
    y_data = np.array(data.loc[:, Y])

    if len(Z) == 0:
        p_val = itest(x_data, y_data, test_args={'statistic': 'ksg_mi', 'k': k})

    else:
        z_data = np.array(data.loc[:, list(Z)])
        p_val = citest(x_data, y_data, z_data, test_args={'statistic': 'ksg_cmi', 'k': k})

    if p_val >= sig:
        return True
    else:
        return False


def mixed_cmi(X, Y, Z, data, **kwargs):
    k = kwargs["knn"]
    sig = kwargs["significance_level"]

    """ adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/mixed_cmi.py
        KSG Conditional Mutual Information Estimator for continuous/discrete mixtures.
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        as well as: https://arxiv.org/abs/1709.06212
        data_matrix: all data
        x: data nodes
        y: data nodes
        s: conditioning nodes
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """
    x_data = np.array(data.loc[:, X])
    y_data = np.array(data.loc[:, Y])

    if len(Z) == 0:
        p_val = itest(x_data, y_data, test_args={'statistic': 'mixed_mi', 'k': k})

    else:
        z_data = np.array(data.loc[:, list(Z)])
        p_val = citest(x_data, y_data, z_data, test_args={'statistic': 'mixed_cmi', 'k': k})

    if p_val >= sig:
        return True
    else:
        return False


def gc_it(X, Y, Z, data, **kwargs):
    # https: // github.com / alexisbellot / GCIT / blob / master / Tutorial.ipynb
    k = kwargs["knn"]
    sig = kwargs["significance_level"]
    """ Adapted from https://github.com/syanga/pycit/blob/master/pycit/estimators/ksg_cmi.py
        KSG Conditional Mutual Information Estimator: I(X;Y|Z)
        See e.g. http://proceedings.mlr.press/v84/runge18a.html
        data_matrix: all data
        x: data nodes
        y: data nodes
        s: conditioning nodes
        k: number of nearest neighbors for estimation
           * k recommended to be on the order of ~ num_samples/10 for independence testing
    """

    x_data = np.array(data.loc[:, X])
    y_data = np.array(data.loc[:, Y])

    if len(Z) == 0:
        z_data = np.ones(len(x_data))
        p_val = GCIT(x_data.reshape(-1, 1), y_data.reshape(-1, 1), z_data.reshape(-1, 1), verbose=False)

    else:
        z_data = np.array(data.loc[:, list(Z)])
        p_val = GCIT(x_data.reshape(-1, 1), y_data.reshape(-1, 1), z_data.reshape(-1, 1), verbose=False)

    if p_val >= sig:
        return True
    else:
        return False
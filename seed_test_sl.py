import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.preprocessing import PolynomialFeatures#
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import ElasticNet, LinearRegression, LogisticRegression, BayesianRidge
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, AdaBoostClassifier, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from warnings import filterwarnings
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
import random
np.random.seed(0)
random.seed(0)


def fn(x, A, b):
	return np.linalg.norm(A.dot(x) - b)


''' An example dictionary of estimators can be specified as follows:
Gest_dict = {'LR': LogisticRegression(), 'SVC': SVC(probability=True),
                                 'RF': RandomForestClassifier(), 'KNN': KNeighborsClassifier(),
                                 'AB': AdaBoostClassifier(), 'poly': 'poly'}

Note that 'poly' is specified as a string because there are no default polynomial feature regressors in sklearn.
This one defaults to 2nd order features (e.g. x1*x2, x1*x3 etc...)'''



class SuperLearner(object):
	def __init__(self, X, Y, seed=0):

		# general settings
		self.seed = seed
		np.random.seed(self.seed)
		random.seed(self.seed)
		self.Q_X = X
		self.Q_Y = Y

	def fit(self):
		qslr = RandomForestClassifier(random_state=np.random.RandomState(self.seed))
		qslr.fit(X=self.Q_X, y=self.Q_Y)
		all_preds_Q = qslr.predict(self.Q_X)
		print(all_preds_Q.sum())

import numpy as np
import os
import pandas as pd
from seed_test_sl import *
import statsmodels.api as sm
from scipy.stats import norm
from scipy.special import logit, expit
from super_learner import *
import random
np.random.seed(0)
random.seed(0)

class TLP(object):
	''' Targeted Learning class.
	::param data: a pandas dataframe with outcome, treatment, and covariates
	::param cause: a string with the dataframe column name for the treatment/cause of interest
	::param outcome: a string with the dataframe column name for the outcome/effect of interest
	::param confs: a list of strings of the dataframe column names for the confounders
	::param precs: a list of strings of the dataframe columns names for precision/risk variables
	::param Q_learners: a list of strings for the abbreviations of the learners to be included in the outcome Q SL
	::param G_learners: a list of strings for the abbreviations of the learners to be included in the treatment G SL
	::param outcome_type: a string 'reg' or 'cls' incicating whether the outcome is binary or continuous
	::param: outcome_upper_bound, outcome_lower_bound: floats for the upper and lower bound of the outcomes for rescaling to [0,1]
	'''

	def __init__(self, data, cause, outcome, confs, precs, Q_learners, G_learners, outcome_type='reg',
	             outcome_upper_bound=None, outcome_lower_bound=None, seed=0):
		# general settings
		self.seed = seed
		np.random.seed(self.seed)
		random.seed(self.seed)
		self.outcome_type = outcome_type  # reg or cls
		self.outcome_upper_bound = outcome_upper_bound  # for bounded outcomes
		self.outcome_lower_bound = outcome_lower_bound  # for bounded outcomes

		# data and variable names
		self.data = data  # pd.DataFrame()
		self.n = len(self.data)
		self.cause = cause
		self.outcome = outcome
		self.confs = confs
		self.precs = precs



		self.Q_X = self.data[sorted(set(confs), key=confs.index) + sorted(set(precs), key=precs.index) + [cause]]
		self.G_X = self.data[sorted(set(confs), key=confs.index)]
		self.num_confs = len(sorted(set(confs), key=confs.index))
		self.Q_Y = self.data[outcome].astype('int') if outcome_type == 'cls' else self.data[outcome]
		self.G_Y = self.data[cause]

		self.A = self.data[cause]
		self.A_dummys = self.A.copy()
		self.A_dummys = self.A_dummys.astype('category')
		self.A_dummys = pd.get_dummies(self.A_dummys, drop_first=False)

		if (self.outcome_upper_bound is not None) and (self.outcome_type == 'reg'):
			self.Q_Y = (self.Q_Y - self.outcome_lower_bound) / (self.outcome_upper_bound - self.outcome_lower_bound)

		self.groups = np.unique(self.A)

		# Super Learners:
		self.Q_learners = Q_learners  # list of learners
		self.G_learners = G_learners  # list of learners
		self.gslr = None
		self.qslr = None

		self.Qbeta = None  # beta learner weights for Q model
		self.Gbeta = None  # beta learner weights for G model

		# targeting and prediction storage
		self.Gpreds = None
		self.QAW = None
		self.Qpred_groups = {}
		self.Gpreds = None
		self.first_estimates = {}
		self.first_effects = {}
		self.updated_estimates = {}
		self.updated_effects = {}
		self.updated_estimates_dr = {}
		self.updated_effects_dr = {}
		self.clev_covs = {}
		self.clev_covs_dr = {}
		self.epsilons = {}
		self.ses = {}
		self.ses_dr = {}
		self.ps = {}
		self.ps_dr = {}

		self.condition1s = []
		self.condition2s = []
		self.condition3s = []

		self.r_or = {}
		self.r_ps1 = {}
		self.r_ps2 = {}

	def fit(self, k):
		print(self.Q_X.iloc[0])
		self.qslr = SuperLearner(output='proba', calibration=False, learner_list=self.Q_learners, k=k,
		                         standardized_outcome=False, seed=self.seed)
		all_preds_Q, gts_Q = self.qslr.fit(x=self.Q_X, y=self.Q_Y)
		print(all_preds_Q.sum())




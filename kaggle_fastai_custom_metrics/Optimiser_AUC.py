import numpy as np

from functools import partial
from scipy.optimize import fmin
from sklearn import metrics

class OptimizeAUC:
	"""
	Class for optimising AUC.
	"""

	def __init__(self):
		self.coef_ = 0

	def _auc(self, coef, X, y):
		"""
		This  functions calculates and returns AUC.
		 Parameters
		: coef : coef list, of same length as number of models
		: X : predictions, in this case a 2d array.
		: y : targets, in our case binary 1d array.
		"""

		x_coef = X * coef
		predictions = np.sum(x_coef, axis = 1)
		auc_score = metrics.roc_auc_score(y, predictions)

		return -1.0*auc_score

	def fit(self, X, y):

		loss_partial = partial(self._auc, X = X, y = y)
		initial_coef = np.random.dirichlet(np.ones(X.shape[1]), size = 1)
		self.coef_ = fmin(loss_partial, initial_coef, disp = True)

	def predict(self, X):

		x_coef = X* self.coef_
		predictions = np.sum(x_coef, axis=1)

		return predictions


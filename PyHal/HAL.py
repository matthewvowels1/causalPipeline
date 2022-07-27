from rpy2.robjects.packages import importr
from .translator import rpy_to_py, py_to_rpy
import rpy2.robjects as ro
from math import sqrt


class Hal9001(object):
    def __init__(self, superlearner=False, site_lib: str = '/usr/local/lib/R/site-library'):

        self.base = importr('base')
        self.utils = importr('utils')

        # print(self.base._libPaths())

        # self.utils.install_packages('hal9001')

        # self.utils.install_package('tidyverse')
        # installed_packages = rpy_to_py(self.utils.installed_packages())

        # print(ro.packages.isinstalled('hal9001'))

        self.package = importr('hal9001')

        self.superlearner = superlearner
        self.site_lib = site_lib
        self.model = None

    def fit(self, X, y,
            max_degree: int = False,
            smoothness_order: int = False,
            family: str = False,
            reduce_basis: float = False,
            fit_control: dict = False):

        if max_degree is False:
            max_degree = 2 if X.shape[1] >= 20 else 3

        if smoothness_order is False:
            smoothness_order = 1

        if family is False:
            assert family in ['gaussian', 'binomial', 'poisson', 'cox']

        if reduce_basis is False:
            reduce_basis = 1/sqrt(y.shape[0])

        if fit_control:

            for key in fit_control:
                assert key in ['cv_select', 'n_fold', 'foldid', 'use_min', 'lambda.min.ratio', 'prediction_bounds']

                if isinstance(fit_control[key], bool):
                    fit_control[key] = 'FALSE' if fit_control[key] is False else 'TRUE'

            r_fit_control = ro.r(f'list(cv_select={fit_control["cv_select"]},'
                                 f' n_fold={fit_control["n_fold"]},'
                                 f' foldid={fit_control["foldid"]},'
                                 f' use_min={fit_control["use_min"]},'
                                 f' lambda.min.ratio={fit_control["lambda.min.ratio"]},'
                                 f' prediction_bounds={fit_control["prediction_bounds"]})')

        else:
            r_fit_control = ro.r('list(cv_select=TRUE,'
                                 'n_folds=10,'
                                 'foldid=NULL,'
                                 'use_min=TRUE,'
                                 'lambda.min.ratio=1e-4,'
                                 'prediction_bounds="default"')

        r_max_degree = py_to_rpy(max_degree)
        r_smoothness_order = py_to_rpy(smoothness_order)
        r_family = py_to_rpy(family)
        r_reduce_basis = py_to_rpy(reduce_basis)

        assert X.shape[0] == y.shape[0]
        rX = py_to_rpy(X)
        ry = py_to_rpy(y)

        if family is False:

            self.model = self.package.fit_hal(rX, ry, max_degree=r_max_degree, smoothness_order=r_smoothness_order,
                                              reduce_basis=r_reduce_basis, fit_control=r_fit_control)

        else:

            self.model = self.package.fit_hal(rX, ry, max_degree=r_max_degree, smoothness_order=r_smoothness_order,
                                              reduce_basis=r_reduce_basis, fit_control=r_fit_control, family=r_family)

    def predict(self, X):

        rX = py_to_rpy(X)

        pred = self.package.predict_hal9001(self.model, rX)

        return rpy_to_py(pred)


if __name__ == '__main__':

    hal = Hal9001()

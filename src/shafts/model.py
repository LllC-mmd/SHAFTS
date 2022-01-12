import os
import numpy as np
from scipy import optimize

import sklearn
from sklearn import metrics
from sklearn import ensemble
import joblib

import xgboost

from .mathexpr import *


# ************************* [1] Machine Learning Method *************************
class VVH_model(object):

    def __init__(self, gamma=5.0, a=-1.0, b=0.1, c=-1.0):
        self.gamma = gamma
        self.a = a
        self.b = b
        self.c = c

    def get_params(self, deep=True):
        return {"gamma": self.gamma}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, feature):
        vvh = get_VVH(vv_coef=feature[:, 0], vh_coef=feature[:, 1], gamma=self.gamma)
        return self.a * np.power(vvh, self.b) + self.c

    def predict_ref(self, feature, a, b, c):
        vvh = get_VVH(vv_coef=feature[:, 0], vh_coef=feature[:, 1], gamma=self.gamma)
        return a * np.power(vvh, b) + c

    def jac_vvh(self, feature, a, b, c):
        vvh = get_VVH(vv_coef=feature[:, 0], vh_coef=feature[:, 1], gamma=self.gamma)
        da = np.power(vvh, b)
        db = a * b * np.power(vvh, b-1.0)
        dc = np.ones_like(vvh)
        return np.hstack((da.reshape(-1, 1), db.reshape(-1, 1), dc.reshape(-1, 1)))

    def fit(self, feature, height):
        para_opt, para_cov = optimize.curve_fit(self.predict_ref, feature, height, p0=[-1.0, 0.1, -1.0], maxfev=5000,
                                                bounds=([-100.0, -10.0, -100.0], [100.0, 10.0, 100.0]), jac=self.jac_vvh)
        self.a = para_opt[0]
        self.b = para_opt[1]
        self.c = para_opt[2]
        print("*"*10 + "Training Results of VVH Model Fitting" + "*"*10)
        print("a = %.6f\tb = %.6f\tc = %.6f" % (self.a, self.b, self.c))
        r2 = self.evaluate(feature, height)
        print("R^2 = %.6f" % r2)
        print("*" * 40)

    def evaluate(self, feature, height_true):
        height_fit = self.predict(feature)
        r2 = metrics.r2_score(y_true=height_true, y_pred=height_fit)
        return r2

    def save_model(self, model_file):
        f = open(model_file, "w")
        f.write("a\tb\tc\n")
        f.write("%.6f\t%.6f\t%.6f\n" % (self.a, self.b, self.c))
        f.close()

    def load_model(self, model_file):
        f = open(model_file)
        para = f.readlines()[1].replace("\n", "").split("\t")
        self.a = float(para[0])
        self.b = float(para[1])
        self.c = float(para[2])


class BaggingSupportVectorRegressionModel(object):

    def __init__(self, n_svr=50, kernel="rbf", c=1.0, epsilon=0.1, max_samples=0.1, n_jobs=1):
        self.n_svr = n_svr
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon
        self.max_samples = max_samples
        self.n_jobs = n_jobs
        self.model = None

    def get_params(self, deep=True):
        return {"n_svr": self.n_svr, "kernel": self.kernel, "c": self.c, "epsilon": self.epsilon,
                "max_samples": self.max_samples}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, feature):
        if self.model is None:
            self.build_model()
        val_pred = self.model.predict(feature)
        return val_pred

    def fit(self, feature, height, sample_weight=None, evaluated=True):
        if self.model is None:
            self.build_model()
        self.model.fit(X=feature, y=height, sample_weight=sample_weight)
        if evaluated:
            print("*" * 10 + "Training Results of Bagging Support Vector Regression Model Fitting" + "*" * 10)
            r2 = self.evaluate(feature, height)
            print("R^2 = %.6f" % r2)
            print("*" * 40)

    def evaluate(self, feature, height_true):
        height_fit = self.predict(feature)
        r2 = metrics.r2_score(y_true=height_true, y_pred=height_fit)
        return r2

    def build_model(self):
        self.model = ensemble.BaggingRegressor(base_estimator=sklearn.svm.SVR(kernel=self.kernel, C=self.c, epsilon=self.epsilon),
                                               max_samples=self.max_samples, n_estimators=self.n_svr, random_state=0, n_jobs=self.n_jobs)

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)

    def load_model(self, model_file):
        self.model = joblib.load(model_file)
        self.n_svr = self.model.n_estimators
        self.kernel = self.model.base_estimator.kernel
        self.c = self.model.base_estimator.C
        self.epsilon = self.model.base_estimator.epsilon
        self.max_samples = self.model.max_samples


class SupportVectorRegressionModel(object):

    def __init__(self, kernel="rbf", c=1.0, epsilon=0.1):
        self.kernel = kernel
        self.c = c
        self.epsilon = epsilon
        self.model = None

    def get_params(self, deep=True):
        return {"kernel": self.kernel, "c": self.c, "epsilon": self.epsilon}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, feature):
        if self.model is None:
            self.build_model()
        val_pred = self.model.predict(feature)
        return val_pred

    def fit(self, feature, height, sample_weight=None, evaluated=True):
        if self.model is None:
            self.build_model()
        self.model.fit(X=feature, y=height, sample_weight=sample_weight)

        if evaluated:
            print("*" * 10 + "Training Results of Support Vector Regression Model Fitting" + "*" * 10)
            r2 = self.evaluate(feature, height)
            print("R^2 = %.6f" % r2)
            print("*" * 40)

    def evaluate(self, feature, height_true):
        height_fit = self.predict(feature)
        r2 = metrics.r2_score(y_true=height_true, y_pred=height_fit)
        return r2

    def build_model(self):
        self.model = sklearn.svm.SVR(kernel=self.kernel, C=self.c, epsilon=self.epsilon, cache_size=200)

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)

    def load_model(self, model_file):
        self.model = joblib.load(model_file)
        self.kernel = self.model.kernel
        self.c = self.model.C
        self.epsilon = self.model.epsilon


class RandomForestModel(object):

    def __init__(self, n_tree=100, max_depth=9, n_jobs=1):
        self.n_tree = n_tree
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self.model = None

    def get_params(self, deep=True):
        return {"n_tree": self.n_tree, "max_depth": self.max_depth}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, feature):
        if self.model is None:
            self.build_model()
        val_pred = self.model.predict(feature)
        return val_pred

    def fit(self, feature, height, sample_weight=None, evaluated=True):
        if self.model is None:
            self.build_model()
        self.model.fit(X=feature, y=height, sample_weight=sample_weight)
        if evaluated:
            print("*" * 10 + "Training Results of Random Forest Regression Model Fitting" + "*" * 10)
            r2 = self.evaluate(feature, height)
            print("R^2 = %.6f" % r2)
            print("*" * 40)

    def evaluate(self, feature, height_true):
        height_fit = self.predict(feature)
        r2 = metrics.r2_score(y_true=height_true, y_pred=height_fit)
        return r2

    def build_model(self):
        self.model = ensemble.RandomForestRegressor(n_estimators=self.n_tree, max_depth=self.max_depth, n_jobs=self.n_jobs)

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)

    def load_model(self, model_file):
        self.model = joblib.load(model_file)
        self.n_tree = self.model.n_estimators
        self.max_depth = self.model.max_depth


class XGBoostRegressionModel(object):

    def __init__(self, n_estimators=100, max_depth=10, gamma=0.1, learning_rate=0.1, reg_lambda=1.0, n_jobs=1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.n_jobs = n_jobs
        self.model = None

    def get_params(self, deep=True):
        return {"n_estimators": self.n_estimators, "max_depth": self.max_depth, "gamma": self.gamma,
                "learning_rate": self.learning_rate, "reg_lambda": self.reg_lambda}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def predict(self, feature):
        if self.model is None:
            self.build_model()
        val_pred = self.model.predict(feature)
        return val_pred

    def fit(self, feature, height, sample_weight=None, evaluated=True):
        if self.model is None:
            self.build_model()
        self.model.fit(X=feature, y=height, sample_weight=sample_weight)
        if evaluated:
            print("*" * 10 + "Training Results of XGBoost Regression Model Fitting" + "*" * 10)
            r2 = self.evaluate(feature, height)
            print("R^2 = %.6f" % r2)
            print("*" * 40)

    def evaluate(self, feature, height_true):
        height_fit = self.predict(feature)
        r2 = metrics.r2_score(y_true=height_true, y_pred=height_fit)
        return r2

    def build_model(self):
        self.model = xgboost.XGBRegressor(objective="reg:squarederror", n_estimators=self.n_estimators, max_depth=self.max_depth,
                                          learning_rate=self.learning_rate, gamma=self.gamma, reg_lambda=self.reg_lambda, n_jobs=self.n_jobs)

    def save_model(self, model_file):
        joblib.dump(self.model, model_file)

    def load_model(self, model_file):
        self.model = joblib.load(model_file)
        self.n_estimators = self.model.n_estimators
        self.max_depth = self.model.max_depth
        self.gamma = self.model.gamma
        self.learning_rate = self.model.learning_rate
        self.reg_lambda = self.model.reg_lambda


if __name__ == "__main__":
    pass


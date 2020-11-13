"""Implementation of LIME for Time Series."""
import logging
import numpy as np
import pandas as pd

from abc import ABC
from sklearn import linear_model
from sklearn.base import BaseEstimator
from ..perturbation import TimeSeriesPerturbation


class XAIModels:
    """Supporting Estimators for XAI.

    reference:
        - https://scikit-learn.org/stable/modules/classes.html#module-sklearn.linear_model
        - https://scikit-learn.org/stable/modules/linear_model.html#linear-model
    """
    # Regression (Forecasting)
    Lasso = linear_model.Lasso(alpha=.5, fit_intercept=True)

    # Classifier
    Ridge = linear_model.Ridge(alpha=.5, fit_intercept=True)


class LIMEAbstract(ABC):
    """Abstract module of LIME which include all methods needs to implemented."""

    def __init__(self, sample_size=100, **_kwargs):
        self.sample_size = sample_size
        self.logger = logging.getLogger(self.__class__.__name__)

        self._xai_estimator = None
        self._perturbator = None

        self._z_prime = []
        self._z = []
        self._z_hat = []
        self._sample_weight = []

    def explain(self, x, predict_fn, **kwargs):
        raise NotImplementedError()

    @property
    def xai_estimator(self):
        return self._xai_estimator

    @xai_estimator.setter
    def xai_estimator(self, v):
        if not isinstance(v, BaseEstimator) or 'fit' not in dir(v):
            raise ValueError("The estimator not supported by sklearn.")
        self._xai_estimator = v

    @property
    def perturb_obj(self):
        return self._perturbator

    @perturb_obj.setter
    def perturb_obj(self, v):
        if 'perturb' not in dir(v):
            raise ValueError("Not found perturb function in the class.")
        self._perturbator = v


class LIMETimeSeries(LIMEAbstract):
    """LIME for time series witch time slicing."""

    def __init__(self, scale='normal', perturb_method='zeros',
                 window_size=3, off_prob=0.5, sample_size=10, **kwargs):
        super().__init__(sample_size, **kwargs)

        # MTS Time Series Initialization
        self.scale = scale
        self.window_size = window_size
        self.n_steps = 0
        self.n_segments = 0
        self.n_features = 0

        # General perturbation Initialization
        self.perturb_method = perturb_method
        self.off_p = off_prob

        # Todo: Generalize
        #   - Frequency Slice
        #   - Time Slice
        #   - XAI-Estimators mapping

        self.perturb_obj = TimeSeriesPerturbation(window_size, off_prob, perturb_method)
        self.xai_estimator = XAIModels.Lasso

    def explain(self, x, predict_fn, **kwargs):
        assert np.ndim(x) == 2, \
            "Only 2 dimension accepted. If univariate time series please use np.reshape(-1, 1)"
        self.n_features, self.n_steps = x.shape
        self.n_segments = (self.n_steps // self.window_size) + int(bool(self.n_steps % self.window_size))
        samples = self._perturbator.perturb(x, n_samples=self.sample_size, **kwargs)

        xai_estimator = self.xai_estimator

        self.logger.info("Fitting xai-model.")

        # Reset samples before new explain
        self._z_prime = []
        self._z = []
        self._z_hat = []
        self._sample_weight = []

        # Todo: any way of using fit as generators to save memory?
        for _z_prime, z, pi_z in samples:
            self._z_prime.append(_z_prime)
            self._z.append(z)
            self._z_hat.append(predict_fn(z))
            self._sample_weight.append(pi_z)

        # Fit to XAI estimator
        xai_estimator.fit(np.stack(self._z_prime),
                          np.stack(self._z_hat),
                          np.stack(self._sample_weight))
        self.logger.info("Updated xai estimator.")

    def explain_instances(self, instances, predict_fn, **kwargs):
        # Todo add to use explain, and in default n_instance = 1
        #   reshape to (n_instances, n_features, n_steps)
        coef = []
        for x in instances:
            self.explain(x, predict_fn, **kwargs)
            coef.append(self.xai_estimator.coef_)
        coef = np.stack(coef)
        coef_mean = coef.mean(axis=0)
        assert self.xai_estimator.coef_.shape == coef_mean.shape, \
            "Not same shape between 2 coefficients"
        self.xai_estimator.coef_ = coef_mean

    def plot_coef(self, feature_names=None, scaler=None, **kwargs):
        coef = self.xai_estimator.coef_
        coef_df = pd.DataFrame(coef.reshape(self.n_segments, self.n_features))
        if feature_names:
            coef_df.columns = feature_names
        if scaler:
            scaler.fit(coef_df.values)
            coef_df = pd.DataFrame(data=scaler.transform(coef_df.values),
                                   columns=coef_df.columns)
        kwargs['kind'] = kwargs.get('kind') or 'bar'
        kwargs['subplots'] = kwargs.get('subplots') or 1
        coef_df.plot(**kwargs)

    def get_a_local_sample(self):
        if len(self._z_prime) > 0:
            idx = np.random.choice(self.sample_size)
            return (self._z_prime[idx].reshape(self.n_features, self.n_segments),
                    self._z[idx].reshape(self.n_features, self.n_steps),
                    self._z_hat[idx],
                    self._sample_weight[idx]
                    )

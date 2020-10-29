"""Implementation of LIME for Time Series."""
import logging
import numpy as np
from abc import ABC
from sklearn.linear_model import Lasso

from ..perturbation import TimeSeriesPerturbation


class LIMEAbstract(ABC):
    """Abstract module of LIME which include all methods needs to implemented."""

    def __init__(self, sample_size=100, **_kwargs):
        self.sample_size = sample_size
        self.logger = logging.getLogger(self.__class__.__name__)

    def __predict__(self, **kwargs):
        raise NotImplementedError()

    def explain(self, x, predict_fn=None, **kwargs):
        raise NotImplementedError()


class LIMETimeSeries(LIMEAbstract):
    """LIME for time series witch time slicing."""

    def __init__(self, scale='normal', perturb_method='zeros',
                 window_size=3, off_prob=0.5, sample_size=10, **kwargs):
        self.scale = scale
        self.window_size = window_size
        self.perturb_method = perturb_method
        self.off_p = off_prob

        # Todo: Generalize this to support also Frequency TS Slicing
        self._perturb = self._perturb = TimeSeriesPerturbation(
            window_size=self.window_size,
            replacement_method=self.perturb_method,
            off_p=self.off_p
        )

        # Todo: Generalize xai_model (can also try with LogisticRegression)
        self.xai_estimator = Lasso(fit_intercept=True)
        super().__init__(sample_size, **kwargs)

    def __predict__(self, z, **kwargs):
        # Todo: deep learning models. (if)
        return 1

    def explain(self, x, predict_fn=None, **kwargs):
        if predict_fn is None:
            predict_fn = self.__predict__
        samples = self._perturb.perturb(x, n_samples=self.sample_size, **kwargs)
        xai_model = self.xai_estimator

        self.logger.info("Fitting xai-model.")

        # Todo: any way of using fit as generators to save memory?
        z_prime = []
        z_hat = []
        sample_weight = []
        for _z_prime, z, pi_z in samples:
            z_prime.append(_z_prime)
            z_hat.append(predict_fn(z))
            sample_weight.append(pi_z)

        xai_model.fit(np.stack(z_prime),
                      np.stack(z_hat),
                      np.stack(sample_weight))
        return xai_model

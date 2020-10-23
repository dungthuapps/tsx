"""Module to implement Perturbation Strategies for time series."""
import numpy as np
from abc import ABC, abstractmethod
from pyts.utils import segmentation

__all__ = ['Perturbation', 'TimeSeriesPerturbation']


class Perturbation(ABC):
    """Base Perturbation with abstract methods."""

    def __init__(self, off_p=0.5, replacement_method='zeros'):
        """Initialize perturbation module."""
        self.off_p = off_p
        self._replacement_fn = replacement_method

        self.labels = None
        self.x_segmented = None
        self.replacements = None

        if isinstance(replacement_method, str):
            # Todo: Try to use mapping
            self._replacement_fn = eval(f"self.{replacement_method}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.labels = None
        self.x_segmented = None
        self.replacements = None

    @abstractmethod
    def __segment__(self, x):
        raise NotImplementedError()

    def __get_replacements__(self, x, **_kwargs):
        """Prepare replacement vectors corresponding to each segments/labels.

        Notice:
            - replacement r_i same shape with z'
        """
        if self.x_segmented is None:
            self.__segment__(x)

        x_segmented = self.x_segmented
        labels = self.labels
        _replacement_fn = self._replacement_fn

        r = _replacement_fn(x=x, x_segmented=x_segmented, labels=labels, **_kwargs)
        return r

    def __get_z_prime__(self, **_kwargs):
        """Sampling based on number of segments/features.

        Sampling z_comma with shape (n_samples, n_windows)
        """
        p = self.off_p
        n_segments = len(self.labels)

        z_prime = np.random.choice([0, 1], size=n_segments, p=[p, 1 - p])
        return z_prime

    def __get_z__(self, x, z_prime, replacements=None, *_args, **_kwargs):
        """Convert from z_prime to z with same format with x.

        :param z_prime: (np.array) a binary vector
            if z_prime[i] == 0, replacements[i] will be used to perturb the segment.
        """
        if self.x_segmented is None:
            self.__segment__(x)
        if replacements is None:
            replacements = self.replacements
        if replacements is None:
            replacements = self.__get_replacements__(x)

        labels = self.labels
        x_segmented = self.x_segmented
        n_segments = len(labels)

        assert len(z_prime) == len(replacements) == len(labels), \
            f"Replacements length {len(replacements)} not match with windows features {len(z_prime)}."
        assert x_segmented.shape == x.shape, \
            f"Not matching shape of segmented {x_segmented.shape} and instance x {x.shape}."

        # Todo: try to use numpy native function instead of for loop
        z = np.zeros_like(x)
        for i in range(n_segments):
            _idx = x_segmented == labels[i]
            z[_idx] = z_prime[i] * x[_idx] + replacements[i] * (1 - z_prime[i])
        return z

    def __get_pi__(self, x, z, gamma=0.01, **kwargs):
        """Calculate distance/similarity from z to x.

        Because z was built from x, hence x, and z must have same shape and length.
        We could simply use 2-norm in np.linalg.norm to calculate the distance with default
        option is to use Frobenius Norm.

        Alternatively, we could use pairwise_distance function from sklearn.metrics to calculate
        the distance but must reshape to (1, -1) for 1 sample x and z only.

        To convert to similarity, we could use np.exp given gamma. gamma ~ 1 / num_features
        Reference:
            - https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.norm.html
            - https://mathworld.wolfram.com/FrobeniusNorm.html
            - https://scikit-learn.org/stable/modules/metrics.html#metrics

        :param x: (np.array) Instance X with shape (n_features, n_steps) to be explained.
            With univariate time series it could be (n_steps, ) or (n_steps, 1)
        :param z: An perturbed sample from z. z must have same shape with x.
        :param gamma: (float) A parameter used to convert to similarity.
            one heuristic for choosing gamma is 1 / num_features
        :param kwargs: Other options in np.linalg.norm()
        """
        assert x.shape == z.shape, \
            f"Not matching shape of segmented {x.shape} and instance x {z.shape}."
        d = np.linalg.norm(x - z, **kwargs)
        pi = np.exp(-d * gamma)
        return pi

    def perturb(self, x, n_samples=10, **_kwargs):
        """Perturb x."""
        self.__segment__(x)
        replacements = self.__get_replacements__(x)

        # Todo: try to use numpy native function instead of for loop
        for i in range(n_samples):
            z_prime = self.__get_z_prime__()
            z = self.__get_z__(x, z_prime, replacements)
            pi_z = self.__get_pi__(x, z)
            yield z_prime, z, pi_z

    @staticmethod
    def zeros(labels, **_kwargs):
        return np.zeros_like(labels)

    @staticmethod
    def local_mean(x, x_segmented, labels, **_kwargs):
        n_segments = len(labels)
        r = np.zeros_like(labels)

        # Todo: try to use numpy native function instead of for loop
        for i in range(n_segments):
            _idx = x_segmented == labels[i]
            r[i] = np.average(x[_idx])
        return r


class TimeSeriesPerturbation(Perturbation):
    """Perturbation for Time Series. Supporting also multivariate time series."""

    # Todo: Implement segmentations with
    #   1. window-size: logarithm vs equally-distributed
    #   2. slicing: Frequency vs Time slice
    def __init__(self, window_size, off_p=0.5, replacement_method='zeros'):
        super().__init__(off_p=off_p, replacement_method=replacement_method)

        self.window_size = window_size

    def __segment__(self, x):
        """Segmentation instance X into segments/labels.

        Time Slices (or time slicing segmentations on normal scale)

        :param x: (np.array) x must be (n_features, n_steps)
        """
        w_size = self.window_size
        n_features, n_steps = x.shape

        # Todo: try to use numpy native function instead of for loop
        x_segmented = np.zeros_like(x)
        start, end, n_windows = segmentation(n_steps, w_size, overlapping=False)
        for i in range(n_features):
            for j in range(n_windows):
                x_segmented[i, start[j]:end[j]] = i * n_windows + j

        self.labels = np.unique(x_segmented)
        self.x_segmented = x_segmented

        # return self.labels, self.x_segmented

    def __get_pi__(self, x, z, **kwargs):
        """Override distance function for Time Series."""
        n_features, n_steps = x.shape
        gamma = 1 / n_steps
        pi = super().__get_pi__(x, z, gamma)
        return pi

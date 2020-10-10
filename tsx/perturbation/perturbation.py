"""Module to implement Perturbation Strategies for time series."""
from numpy.random import rand


class Perturbation:
    """Base Perturbation with abstract methods."""

    def __init__(self, x):
        self.x = x

    def __segment__(self):
        """Split x into segmentations."""
        pass

    def __segment_sampling__(self):
        """Sampling based on number of segments/features."""
        pass

    def __predict__(self, z, fn):
        """Return predictions given the function or model."""
        z_hat = fn(z)
        return z_hat

    def __pi__(self, z):
        """Calculate distance/similarity from z to x."""
        return 1


def test_windows_slicing():
    from pyts.utils import segmentation
    import numpy as np

    n_steps = 20
    n_cols = 1
    w_size = 3

    # shape (n_steps, n_cols) if 1d (n_steps, ) -> (n_steps, 1)
    Xt = np.arange(n_steps * n_cols).reshape(n_steps, n_cols)
    if len(Xt.shape) == 1:
        Xt = Xt.reshape(Xt.shape[0], -1)

    # each segment is a feature.
    start, end, n_features = segmentation(n_steps, w_size)

    # on-off feature vectors
    n_samples = 2
    p = 0.5  # p of off

    # (n_samples, n_features, n_cols)
    z_comma_samples = np.random.choice([0, 1], size=(n_samples, n_features, n_cols), p=[p, 1 - p])
    # For each sample, yield (z', z, pi_x(z), z_hat ~ f(z))
    #   1. samples_in_origin.
    #      a. replace off feature by "replacement-strategy"
    #      b. store the vector to feed to g(z') model later
    #   2. calculate distance to x (pi_x)
    #   3. get the z_hat

    # Todo: (1) loop / apply, which feature is nan -> apply replacement strategy
    # cannot use start, end
    # E.g (2 samples, 7 features, 1 columns) w_size = 3 but n_steps actually = 20 only
    #   - Broadcast (2, 7, 1) --> (2, 21, 1)
    _mask = z_comma_samples * np.ones(w_size)
    _mask = _mask.reshape(n_samples, n_features * w_size, n_cols)

    #   Convert (2, 21, 1) --> (2, 20, 1) by removing late redundant values
    mask = _mask[:, :n_steps, :]

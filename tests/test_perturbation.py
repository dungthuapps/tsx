import numpy as np
from tsx.perturbation import TimeSeriesPerturbation
from tsx.xai.lime import LIMETimeSeries

mts = np.array([np.arange(1, 9), np.arange(2, 10)])
uts = np.arange(1, 9).reshape(1, -1)


def test_perturbation_time_series():
    with TimeSeriesPerturbation(window_size=3, replacement_method="local_mean") as t:
        samples = list(t.perturb(mts, n_samples=5))
        z_prime, z, pi_z = samples[0]
        assert z.shape == mts.shape
        assert len(z_prime) == len(t.labels)
        assert pi_z >= 0

        # Custom replacement
        replacements = 99
        samples = list(t.perturb(mts, n_samples=5, replacements=replacements))
        z_prime, z, pi_z = samples[0]
        if 0 in z_prime:
            assert replacements in z


def test_lime_linear():
    ts_lime = LIMETimeSeries(window_size=2, sample_size=100)
    xai_model = ts_lime.explain(mts,
                                predict_fn=lambda x: np.random.randint(100))
    assert len(xai_model.coef_) == len(ts_lime._perturb.labels)

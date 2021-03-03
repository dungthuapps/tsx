import os
import numpy as np

from tsx.perturbation import (
    TimeSeriesPerturbation, 
    SyncTimeSlicer,
    ASyncTimeSlicer
    )

DATA_DIR = "tests/data" if os.path.isdir("tests") else "data"
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


def test_perturbation_sync_time_slicer():
    t = SyncTimeSlicer(window_size=3, replacement_method="local_mean")
    _slices = list(t._slices(mts))
    x_segmented = t._x_segmented(mts)
    r_zeros = t._x_replacements(mts, 'zeros')
    target = np.zeros_like(mts)
    assert np.sum(r_zeros - target) == 0
    assert r_zeros.shape == x_segmented.shape

    r_mean = t._x_replacements(mts, 'local_mean')
    target = np.array([[1.5, 1.5, 4., 4., 4., 7., 7., 7.],
                       [2.5, 2.5, 5., 5., 5., 8., 8., 8.]])
    assert np.sum(r_mean - target) == 0

    r_global_mean = t._x_replacements(mts, 'global_mean')
    target = np.array([[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                       [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]])
    assert np.sum(r_global_mean - target) == 0

    z_prime = t._z_prime(mts)
    assert len(z_prime) == len(_slices)

    z_prime = np.array([1, 0, 1])
    x_masked = t._x_masked(mts, z_prime)
    target = np.array([[1, 1, 0, 0, 0, 1, 1, 1],
                       [1, 1, 0, 0, 0, 1, 1, 1]])
    assert np.sum(x_masked < 2), "Only Binary type here"
    assert np.sum(x_masked - target) == 0

    z_prime = np.array([1, 0, 1])
    r = t._x_replacements(mts, 'local_mean')
    z = t._z(mts, z_prime, r)
    target = np.array([[1., 2., 4., 4., 4., 6., 7., 8.],
                       [2., 3., 5., 5., 5., 7., 8., 9.]])
    assert np.sum(z - target) == 0

    z_prime = np.array([0, 0, 1])
    r = t._x_replacements(mts, 'local_mean')
    z = t._z(mts, z_prime, r)
    target = np.array([[1.5, 1.5, 4., 4., 4., 6., 7., 8.],
                       [2.5, 2.5, 5., 5., 5., 7., 8., 9.]])

    assert np.sum(z - target) == 0

    z_prime = np.array([0, 0, 1])
    r = t._x_replacements(mts, 'zeros')
    z = t._z(mts, z_prime, r)
    target = np.array([[0, 0, 0, 0, 0, 6, 7, 8],
                       [0, 0, 0, 0, 0, 7, 8, 9]])
    assert np.sum(z - target) == 0

    samples = t.perturb(mts, n_samples=10)
    z_prime, z, pi_z = next(samples)
    assert len(z_prime) == len(_slices)  # for SyncTimeSlicer
    assert z.shape == mts.shape and (pi_z <= 1) and (pi_z >= 0)

    r = t._x_replacements(mts, fn='local_mean')
    mask = t._x_masked(mts, z_prime)
    target_z = mts * mask + r * (1 - mask)  # Formula for z
    assert np.sum(z - target_z) == 0

def test_perturbation_async():
    t = ASyncTimeSlicer(window_size=3, replacement_method="local_mean")

    # Assert x_segmented
    x_segmented = t._x_segmented(mts)
    target = np.array([[0, 0, 1, 1, 1, 2, 2, 2],
                       [3, 3, 4, 4, 4, 5, 6, 6]])
    print(x_segmented)
    assert np.sum(x_segmented - target) == 0

    # Assert masking
    z_prime = np.array([1, 1, 0, 0, 1, 1])
    x_masked = t._x_masked(mts, z_prime)
    target = np.array([[1, 1, 1, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 1, 1, 1, 1]])
    assert np.sum(x_masked - target) == 0

    # Assert Replacements - Zeros
    r_zeros = t._x_replacements(mts, 'zeros')
    target = np.array([[0, 0, 0, 0, 0, 0, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 0]])
    assert np.sum(r_zeros - target) == 0

    # Assert Replacements - Local Mean
    r_local_mean = t._x_replacements(mts, 'local_mean')
    target = np.array([[1.5, 1.5, 4., 4., 4., 7., 7., 7.],
                       [2.5, 2.5, 5., 5., 5., 8., 8., 8.]])
    assert np.sum(r_local_mean - target) == 0

    # Assert Replacements - Global Mean
    r_global_mean = t._x_replacements(mts, 'global_mean')
    target = np.array([[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5],
                       [5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5, 5.5]])
    assert np.sum(r_global_mean - target) == 0

    # Assert z 
    z_prime = np.array([1, 1, 0, 0, 1, 1])
    
    #   - Zeros
    r = t._x_replacements(mts, 'zeros')
    z = t._z(mts, z_prime, r)
    target = np.array([[1. , 2. , 3. , 4. , 5. , 0. , 0. , 0. ],
                        [0, 0, 4. , 5. , 6. , 7. , 8. , 9. ]])

    assert np.sum(z - target) == 0

    #   - local mean
    r = t._x_replacements(mts, 'local_mean')
    z = t._z(mts, z_prime, r)
    target = np.array([[1. , 2. , 3. , 4. , 5. , 7. , 7. , 7. ],
                        [2.5, 2.5, 4. , 5. , 6. , 7. , 8. , 9. ]])

    assert np.sum(z - target) == 0

    #   - Global mean
    r = t._x_replacements(mts, 'global_mean')
    z = t._z(mts, z_prime, r)
    target = np.array([[1. , 2. , 3. , 4. , 5. , 4.5, 4.5, 4.5],
                        [5.5, 5.5, 4. , 5. , 6. , 7. , 8. , 9. ]])

    assert np.sum(z - target) == 0

def test_to_original():
    t = ASyncTimeSlicer(window_size=3)
    
    # ------ UTS -------
    z_prime = np.array([1, 0, 1])
    # uts - zeros
    z = t.to_original(uts, z_prime, repl_fn='zeros')
    target = np.array([[1, 2, 0, 0, 0, 6, 7, 8]])
    assert np.sum(z - target) == 0

    # uts - local_mean
    z = t.to_original(uts, z_prime, repl_fn='local_mean')
    target = np.array([[1, 2, 4, 4, 4, 6, 7, 8]])
    assert np.sum(z - target) == 0

    # uts - global_mean
    z = t.to_original(uts, z_prime, repl_fn='global_mean')
    target = np.array([[1, 2, 4.5, 4.5, 4.5, 6, 7, 8]])
    assert np.sum(z - target) == 0

    # ------ MTS -------
    z_prime = np.array([1, 1, 0, 0, 1, 1])
    
    # mts - Zeros
    z = t.to_original(mts, z_prime, repl_fn='zeros')
    target = np.array([[1. , 2. , 3. , 4. , 5. , 0. , 0. , 0. ],
                        [0, 0, 4. , 5. , 6. , 7. , 8. , 9. ]])

    assert np.sum(z - target) == 0

    # mts - local_mean
    z = t.to_original(mts, z_prime, repl_fn='local_mean')
    target = np.array([[1. , 2. , 3. , 4. , 5. , 7. , 7. , 7. ],
                        [2.5, 2.5, 4. , 5. , 6. , 7. , 8. , 9. ]])

    assert np.sum(z - target) == 0

    # mts - global_mean
    z = t.to_original(mts, z_prime, repl_fn='global_mean')
    target = np.array([[1. , 2. , 3. , 4. , 5. , 4.5, 4.5, 4.5],
                        [5.5, 5.5, 4. , 5. , 6. , 7. , 8. , 9. ]])

    assert np.sum(z - target) == 0
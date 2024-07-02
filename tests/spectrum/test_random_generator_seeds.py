import numpy as np

import ramanchada2 as rc2


def test_add_baseline():
    spe = rc2.spectrum.from_delta_lines(deltas={100: 1, 200: 2}, nbins=800)

    spen1 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=None)
    spen2 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=None)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=123)
    spen2 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=124)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=123)
    spen2 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=123)
    assert np.allclose(spen1.y, spen2.y)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    spen1 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=state0)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    state1 = rng.bit_generator.state
    spen1 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=state0)
    assert state0['state']['state'] != state1['state']['state']
    spen2 = spe.add_baseline(n_freq=40, amplitude=1, rng_seed=state1)
    assert state0['state']['state'] == state1['state']['state']
    assert np.allclose(spen1.y, spen2.y)


def test_add_gaussian_noise_drift():
    spe = rc2.spectrum.from_delta_lines(deltas={100: 1, 200: 2}, nbins=800
                                        ).add_baseline(n_freq=40, amplitude=1, rng_seed=None)

    spen1 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=None)
    spen2 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=None)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=123)
    spen2 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=124)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=123)
    spen2 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=123)
    assert np.allclose(spen1.y, spen2.y)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    spen1 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=state0)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    state1 = rng.bit_generator.state
    spen1 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=state0)
    assert state0['state']['state'] != state1['state']['state']
    spen2 = spe.add_gaussian_noise_drift(sigma=.1, coef=.1, rng_seed=state1)
    assert state0['state']['state'] == state1['state']['state']
    assert np.allclose(spen1.y, spen2.y)


def test_add_gaussian_noise():
    spe = rc2.spectrum.from_delta_lines(deltas={100: 1, 200: 2}, nbins=800
                                        ).add_baseline(n_freq=40, amplitude=1, rng_seed=None)

    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=None)
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=None)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=123)
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=124)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=123)
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=123)
    assert np.allclose(spen1.y, spen2.y)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=state0)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    state1 = rng.bit_generator.state
    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=state0)
    assert state0['state']['state'] != state1['state']['state']
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=state1)
    assert state0['state']['state'] == state1['state']['state']
    assert np.allclose(spen1.y, spen2.y)


def test_add_poisson_noise():
    spe = rc2.spectrum.from_delta_lines(deltas={100: 1, 200: 2}, nbins=800
                                        ).add_baseline(n_freq=40, amplitude=1, rng_seed=None)

    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=None)
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=None)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=123)
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=124)
    assert not np.allclose(spen1.y, spen2.y)

    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=123)
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=123)
    assert np.allclose(spen1.y, spen2.y)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=state0)

    rng = np.random.default_rng()
    state0 = rng.bit_generator.state
    state1 = rng.bit_generator.state
    spen1 = spe.add_gaussian_noise(sigma=.1, rng_seed=state0)
    assert state0['state']['state'] != state1['state']['state']
    spen2 = spe.add_gaussian_noise(sigma=.1, rng_seed=state1)
    assert state0['state']['state'] == state1['state']['state']
    assert np.allclose(spen1.y, spen2.y)

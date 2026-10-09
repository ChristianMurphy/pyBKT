"""Fitting and predicting work for a student with a very long history."""
import numpy as np
import pytest

from helpers import IS_COMPILED
from pyBKT.fit import EM_fit, predict_onestep
from pyBKT.generate import random_model_uni

# The compiled E-step once kept each student's working arrays on the stack, so a
# few hundred thousand attempts from one student overflowed it and crashed.
pytestmark = pytest.mark.skipif(not IS_COMPILED, reason="too slow on the pure-Python build")

ATTEMPTS = 400_000


@pytest.fixture(scope="module")
def one_long_student():
    rng = np.random.default_rng(0)
    return {
        "data": rng.integers(1, 3, size=(1, ATTEMPTS)).astype(np.int32),
        "resources": np.ones(ATTEMPTS, dtype=np.int64),
        "starts": np.array([1], dtype=np.int64),
        "lengths": np.array([ATTEMPTS], dtype=np.int64),
    }


@pytest.mark.parametrize("parallel", [False, True])
def test_em_fits_one_long_student(one_long_student, parallel):
    start = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    fitted, log_likelihoods = EM_fit.EM_fit(start, one_long_student, tol=-1, maxiter=2, parallel=parallel)
    assert np.isfinite(log_likelihoods).all()
    assert 0 <= fitted["learns"][0] <= 1


def test_predict_handles_one_long_student(one_long_student):
    model = random_model_uni.random_model_uni(1, 1, rand=np.random.RandomState(0))
    correct, state = predict_onestep.run(model, one_long_student, parallel=False)
    assert correct.shape == (ATTEMPTS,)
    assert state.shape == (2, ATTEMPTS)
    np.testing.assert_allclose(state.sum(axis=0), 1)

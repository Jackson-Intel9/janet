"""
"""
import os
from ..train_network import train_photometry_network

NBATCH = int(os.environ.get("JANET_NBATCH", 20))
NSTEPS = int(os.environ.get("JANET_NSTEPS", 20))


def test_train_photometry_network():
    n_batch, n_steps = NBATCH, NSTEPS
    state, loss_history, p_best, loss_min = train_photometry_network(n_batch, n_steps)
    assert loss_min < loss_history[0]

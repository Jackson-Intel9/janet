"""
"""
from ..train_network import train_photometry_network


def test_train_photometry_network():
    n_batch, n_steps = 20, 20
    state, loss_history, p_best, loss_min = train_photometry_network(n_batch, n_steps)
    assert loss_min < loss_history[0]

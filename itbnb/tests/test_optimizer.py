import numpy as np
from sklearn.datasets import make_classification

from itbnb import TbNB, ThresholdOptimizer


def test_threshold_optimizer_fit():

    X, y = make_classification(
        n_samples=80,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_classes=2,
        random_state=0,
    )

    estimator_params = dict(
        optimize_threshold=False,
        iterative=False,
    )

    tau_grid = np.linspace(-5.0, 5.0, 20)

    optimizer = ThresholdOptimizer(
        tau_grid=tau_grid,
        estimator_class=TbNB,
        estimator_params=estimator_params,
        K=3,
        random_state=0,
    )

    optimizer.fit(X, y)

    assert hasattr(optimizer, "best_tau_")
    assert "accuracy" in optimizer.best_tau_

    tau = optimizer.best_tau_["accuracy"]
    assert np.isscalar(tau)

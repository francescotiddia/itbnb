from dataclasses import dataclass

import numpy as np
from scipy import optimize
from scipy.stats import gaussian_kde, norm


@dataclass
class Decision:
    iteration: int
    start: float
    end: float
    tau: float
    x_max_pos: float
    x_max_neg: float
    direction: str


def _intersect_threshold(scores_pos, scores_neg):
    """
    Estimate the intersection point (new threshold) between two class densities.

    Parameters
    ----------
    scores_pos : array-like of shape (n_pos,)
        Scores associated with positive class instances.
    scores_neg : array-like of shape (n_neg,)
        Scores associated with negative class instances.

    Returns
    -------
    tau : float
        Intersection point of the two class densities (refined threshold).
    x_max_pos : float
        Score corresponding to the argmax of the positive class density.
    x_max_neg : float
        Score corresponding to the argmax of the negative class density.

    Raises
    ------
    InterruptedError
    If the two densities do not intersect (no sign change in their difference).
    """
    f_pos = gaussian_kde(scores_pos)
    f_neg = gaussian_kde(scores_neg)

    low = min(scores_pos.min(), scores_neg.min())
    high = max(scores_pos.max(), scores_neg.max())

    try:
        tau = optimize.brentq(lambda x: f_pos(x) - f_neg(x), low, high)
    except Exception as e:
        raise InterruptedError from e

    x_max_pos = optimize.minimize_scalar(
        lambda x: -f_pos(x), bounds=(low, high), method="bounded"
    ).x

    x_max_neg = optimize.minimize_scalar(
        lambda x: -f_neg(x), bounds=(low, high), method="bounded"
    ).x

    return tau, x_max_pos, x_max_neg


def predict_from_decisions(X, decisions, default_threshold=0.0):
    """
    Generate class predictions using a sequence of refined decision rules.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Decision scores to classify.
    decisions : list of Decision
        Sequence of iterative threshold refinement steps.
    default_threshold : float, default=0.0
        Base threshold used for classification outside any refined regions.

    Returns
    -------
    y_pred : ndarray of shape (n_samples,)
        Predicted class labels (0 or 1) after applying all iterative decisions.
    """

    X = np.asarray(X)

    y_pred = (X > default_threshold).astype(int)

    decisions = sorted(decisions, key=lambda d: d.iteration)

    for d in decisions:
        mask = (X >= d.start) & (X <= d.end)
        if not np.any(mask):
            continue

        if d.direction == "r":
            y_pred[mask & (X > d.tau)] = 1
            y_pred[mask & (X <= d.tau)] = 0
        else:
            y_pred[mask & (X < d.tau)] = 1
            y_pred[mask & (X >= d.tau)] = 0

    return y_pred


def _intersect_threshold_clt(scores_pos, scores_neg, n_boot=200, eps=1e-12):
    """
    Estimate the intersection threshold between two class score distributions
    using a Central Limit Theorem (CLT) approximation.

    The method approximates the distributions of class-specific scores by
    Normal densities obtained from bootstrap sample means. For each class,
    repeated bootstrap samples are drawn and their means are used to estimate
    the parameters of a Normal distribution, as motivated by the Central Limit
    Theorem. The refined threshold is then computed as the intersection point
    of the two Normal density functions.

    Parameters
    ----------
    scores_pos : array-like of shape (n_pos,)
        Decision scores associated with positive class instances.

    scores_neg : array-like of shape (n_neg,)
        Decision scores associated with negative class instances.

    n_boot : int, default=200
        Number of bootstrap resamples used to estimate the distribution
        of the sample means.


    eps : float, default=1e-12
        Small positive constant used to prevent degenerate standard deviations.

    Returns
    -------
    tau : float
        Estimated intersection point of the two Normal density functions,
        representing the refined decision threshold.

    x_max_pos : float
        Location of the maximum of the positive class density, corresponding
        to the estimated mean of the positive class distribution.

    x_max_neg : float
        Location of the maximum of the negative class density, corresponding
        to the estimated mean of the negative class distribution.

    Raises
    ------
    InterruptedError
        If the two Normal densities do not intersect within the range of the
        observed scores.
    """

    sample_size_pos = len(scores_pos)
    sample_size_neg = len(scores_neg)

    pos_means = []
    neg_means = []

    for _ in range(n_boot):
        pos_means.append(
            np.mean(np.random.choice(scores_pos, size=sample_size_pos, replace=True))
        )
        neg_means.append(
            np.mean(np.random.choice(scores_neg, size=sample_size_neg, replace=True))
        )

    mu_pos, sigma_pos = np.mean(pos_means), max(np.std(pos_means), eps)
    mu_neg, sigma_neg = np.mean(neg_means), max(np.std(neg_means), eps)

    low = min(scores_pos.min(), scores_neg.min())
    high = max(scores_pos.max(), scores_neg.max())

    g = lambda x: norm.pdf(x, mu_pos, sigma_pos) - norm.pdf(x, mu_neg, sigma_neg)

    if np.sign(g(low)) == np.sign(g(high)):
        raise InterruptedError

    tau = optimize.brentq(g, low, high)
    x_max_pos = mu_pos
    x_max_neg = mu_neg
    return tau, x_max_pos, x_max_neg


def iterate_threshold(
    X, Y, tau, p=0.2, s=20, i=0, epsilon=1e-3, mode="kde", clt_boot=500, clt_sample=30
):
    """
    Perform iterative local refinement of the decision threshold around
    regions of classification uncertainty.

    This procedure implements the iterative Threshold-based Naive Bayes
    (iTb-NB) refinement step. Starting from an initial global threshold `tau`,
    the algorithm identifies an uncertainty region around the current threshold
    by selecting a fraction of samples closest to it from each class. A new,
    refined threshold is then estimated by intersecting the class-specific
    score densities within this region. The process is repeated iteratively
    until a stopping criterion is met.

    At each iteration, a single continuous window Omega_k containing the current
    threshold is constructed, and a local decision rule is stored. The final
    output is a sequence of decision rules that can be applied sequentially
    to refine predictions near the decision boundary.

    Parameters
    ----------
    X : array-like of shape (n_samples,)
        Continuous decision scores for all samples.

    Y : array-like of shape (n_samples,)
        Binary class labels encoded as {0, 1}.

    tau : float
        Initial decision threshold used to start the iterative refinement.

    p : float, default=0.2
        Fraction of samples closest to the current threshold selected from
        each class to define the local uncertainty region.

    s : int, default=20
        Minimum number of samples required to perform a refinement step.
        - When ``mode="kde"``: minimum **total** samples in uncertainty window
        - When ``mode="clt"``: minimum samples **per class** (ensures CLT validity)
        Recommended: >= 15 for KDE, >= 30 for CLT.

    i : int, default=0
        Initial iteration index. This parameter is used internally to label
        refinement steps and should typically be left unchanged.

    epsilon : float, default=1e-3
        Minimum distance between the modes of the positive and negative
        class score densities required to continue the refinement. If the
        modes become indistinguishable, the iteration stops.

    mode : {"kde", "clt"}, default="kde"
        Method used to estimate class score densities within the uncertainty
        region:
        - "kde": Gaussian kernel density estimation
        (non-parametric, works with fewer samples)
        - "clt": Normal approximation via Central Limit Theorem
        (parametric, more stable, requires more data per class)

    clt_boot : int, default=500
        Number of bootstrap resamples used when ``mode="clt"``.
        Higher values (e.g., 500-1000) provide more stable parameter estimates
        but increase computational cost. Ignored when ``mode="kde"``.

    clt_sample : int, default=30
        Size of each bootstrap sample when ``mode="clt"``. This is the "n" in
        the CLT approximation. Must be >= 30 for valid Normal approximation.
        This parameter controls the variance of the bootstrap means: smaller
        values give wider distributions, larger values give narrower ones.
        Ignored when ``mode="kde"``.

    Returns
    -------
    decisions : list of Decision
        Ordered list of local decision rules generated at each iteration.
        Each element encodes the refinement window, the refined threshold,
        and the direction of the decision rule.

    Notes
    -----
    The iterative refinement does not re-estimate feature likelihoods.
    Only the decision rule is updated locally around the threshold.
    The algorithm typically converges in a small number of iterations
    (often fewer than three), as observed in empirical studies.

    When using mode="clt", sufficient samples per class are required for
    valid Normal approximation. The parameter `s` controls this minimum:
    it should be at least 30 for reliable CLT-based inference, though
    smaller values (15-20) may work for KDE mode.

    Raises
    ------
    InterruptedError
        If a valid refined threshold cannot be estimated at any iteration,
        or if insufficient samples are available for the chosen mode.
    """

    decisions = []

    curr_x = np.asarray(X)
    curr_y = np.asarray(Y)

    while True:
        i += 1

        x_pos = curr_x[curr_y == 1]
        x_neg = curr_x[curr_y == 0]

        if len(x_pos) < 2 or len(x_neg) < 2:
            break

        d_pos = np.abs(x_pos - tau)
        d_neg = np.abs(x_neg - tau)
        k_pos = max(1, int(np.ceil(p * len(x_pos))))
        k_neg = max(1, int(np.ceil(p * len(x_neg))))

        idx_pos = np.argsort(d_pos)
        idx_neg = np.argsort(d_neg)

        omega_pos = x_pos[idx_pos[:k_pos]]
        omega_neg = x_neg[idx_neg[:k_neg]]

        start = min(omega_pos.min(), omega_neg.min())
        end = max(omega_pos.max(), omega_neg.max())

        mask = (curr_x >= start) & (curr_x <= end)
        omega_x = curr_x[mask]
        omega_y = curr_y[mask]

        pos_in_win = omega_x[omega_y == 1]
        neg_in_win = omega_x[omega_y == 0]

        if mode == "kde":
            if len(omega_x) < s:
                break

            if len(pos_in_win) < 2 or len(neg_in_win) < 2:
                break
        elif mode == "clt":
            if len(pos_in_win) < s or len(neg_in_win) < s:
                break
        else:
            raise ValueError(f"mode must be 'kde' or 'clt', got '{mode}'")
        try:
            if mode == "kde":
                tau_new, x_max_pos, x_max_neg = _intersect_threshold(
                    pos_in_win, neg_in_win
                )
            else:
                tau_new, x_max_pos, x_max_neg = _intersect_threshold_clt(
                    pos_in_win, neg_in_win, n_boot=clt_boot, sample_size=clt_sample
                )
        except InterruptedError:
            break
        except Exception:
            break

        # Check if modes are distinguishable
        if abs(x_max_pos - x_max_neg) < epsilon:
            break

        direction = "r" if x_max_pos > x_max_neg else "l"

        decisions.append(
            Decision(
                iteration=i,
                start=start,
                end=end,
                tau=tau_new,
                x_max_pos=x_max_pos,
                x_max_neg=x_max_neg,
                direction=direction,
            )
        )

        # Update for next iteration
        tau = tau_new
        curr_x = omega_x
        curr_y = omega_y

    return decisions

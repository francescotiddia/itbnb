import warnings
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.model_selection import StratifiedKFold

from .utils.confusion import all_metrics
from .utils.validation import check_y


class ThresholdOptimizer(BaseEstimator):
    """
    Cross-validated threshold optimizer for the Threshold-based Naïve Bayes (Tb-NB) classifier.

    The `ThresholdOptimizer` class estimates the optimal decision threshold (τ)
    that minimizes or maximizes a relevant metric. It can be applied to any estimator implementing
    a `fit(X, y)` and `decision_function(X)` interface.

   Parameters
    ----------
    tau_grid : array-like of shape (n_tau,)
        Candidate thresholds to evaluate.

    estimator_class : class
        Estimator class implementing ``fit(X, y)`` and at least one of
        ``decision_function(X)`` or ``predict_proba(X)``.

    estimator_params : dict, optional
        Parameters passed to the estimator constructor.

    fit_params : dict, optional
        Additional keyword arguments passed to ``estimator.fit``.

    K : int, default=5
        Number of cross-validation folds.

    random_state : int, default=42
        Random seed used for the stratified K-fold split.

    validate_inputs : bool, default=True
        Whether to validate and sanitize the target vector ``y``.

    Attributes
    ----------
    tau_grid_ : ndarray of shape (n_tau,)
        Sorted array of thresholds actually used.

    metric_mats_ : dict
        Dictionary mapping each metric name to an array of shape
        (K, n_tau) containing cross-validated scores.

    best_tau_ : dict
        Dictionary mapping each metric name (and ``balanced_error``)
        to the optimal threshold τ.

    folds_ : list of tuples
        Train/test indices for each cross-validation fold.

    _is_fitted : bool
        Flag indicating whether ``fit`` has been called.
    """

    available_metrics = ['precision',
                         'recall',
                         'specificity',
                         'fpr',
                         'f1',
                         'mcc',
                         'misclassification_error',
                         'fnr',
                         'accuracy']

    def __init__(self, tau_grid, estimator_class, estimator_params=None, fit_params=None, K=5, random_state=42,
                 validate_inputs=True):
        self.estimator_class = estimator_class
        self.K = K
        self.random_state = random_state
        self.fit_params = fit_params
        self.estimator_params = estimator_params
        self.tau_grid = tau_grid
        self.validate_inputs = validate_inputs
        self._is_fitted = False

    def _best(self, mat, maximize=True):
        """
        Select the best threshold τ for a given metric matrix.

        Parameters
        ----------
        mat : ndarray of shape (K, n_tau)
            Cross-validated metric values for each fold and τ.

        maximize : bool, default=True
            If True, selects τ maximizing the metric; otherwise minimizes.

        Returns
        -------
        tau : float
            Best threshold according to the mean metric across folds.
        """

        if mat is None or self.tau_grid is None:
            return None
        mean = np.mean(mat, axis=0)
        idx = np.nanargmax(mean) if maximize else np.nanargmin(mean)
        return self.tau_grid_[idx]

    def fit(self, X, y):

        """
            Perform K-fold cross-validation to estimate relevant metrics
                for each candidate threshold τ.

                Parameters
                ----------
                X : ndarray of shape (n_samples, n_features)
                y : ndarray of shape (n_samples,)
                    Binary class labels

                Returns
                -------
                self : ThresholdOptimizer
                    The fitted optimizer containing cross-validated error matrices.
        """
        if self.validate_inputs:
            y = check_y(y)

        self.fit_params_, self.tau_grid_ = self._validate_input()

        skf = StratifiedKFold(
            n_splits=self.K,
            shuffle=True,
            random_state=self.random_state
        )
        self.folds_ = list(skf.split(X, y))

        n_folds = len(self.folds_)
        n_tau = len(self.tau_grid_)

        self.metric_mats_ = {m: np.zeros((n_folds, n_tau), dtype=np.float32)
                             for m in self.available_metrics}

        for k, (train_idx, test_idx) in enumerate(self.folds_):
            X_train, y_train = X[train_idx], y[train_idx]
            X_test, y_test = X[test_idx], y[test_idx]

            estimator_params = {} if self.estimator_params is None else self.estimator_params
            try:
                model = self.estimator_class(**estimator_params)
            except TypeError as e:
                raise TypeError(f"Invalid estimator_params passed to estimator_class: {e}")

            try:
                model.fit(X_train, y_train, **self.fit_params_)
            except TypeError as e:
                raise TypeError(f"Invalid fit_params passed to model.fit: {e}")

            scores = self._get_scores(model, X_test)

            TP, TN, FP, FN = self.confusion_counts(scores, y_test, self.tau_grid_)
            self._compute_fold_metrics(k, TP, TN, FP, FN)

        self._is_fitted = True
        self._compute_taus()
        return self

    def _compute_fold_metrics(self, fold_idx, TP, TN, FP, FN):
        metrics = all_metrics(TP, FP, TN, FN)
        for m in self.available_metrics:
            self.metric_mats_[m][fold_idx, :] = metrics[m]

    @staticmethod
    def _get_scores(model, X):
        """
        Retrieve continuous scores from a fitted model.

        Tries the following methods in order:
        1. ``decision_function(X)``
        2. ``predict_proba(X)`` (returns column 1 if shape is (n, 2))

        Parameters
        ----------
        model : object
            Fitted estimator.

        X : ndarray of shape (n_samples, n_features)
            Test data.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Continuous decision scores suitable for thresholding.

        Raises
        ------
        AttributeError
            If none of the supported scoring methods are available.
        """

        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
                return np.asarray(scores).ravel()
            except Exception:
                pass

        if hasattr(model, "predict_proba"):
            try:
                proba = np.asarray(model.predict_proba(X))

                if proba.ndim == 2:
                    if proba.shape[1] != 2:
                        raise ValueError(
                            f"predict_proba must return an array of shape (n_samples, 2) "
                            f"for binary classification, but got shape {proba.shape}"
                        )
                    return proba[:, 1]
                elif proba.ndim == 1:
                    return proba

                else:
                    raise ValueError(
                        f"predict_proba returned an array with invalid shape {proba.shape}"
                    )
            except Exception:
                pass

        if hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X)
                return np.asarray(scores).ravel()
            except Exception:
                pass

        raise AttributeError(
            "The model does not provide any usable scoring method among: "
            "decision_function, predict_proba (validated), decision_function."
        )

    @staticmethod
    def confusion_counts(scores, y, tau_grid):
        """
            Compute TP, TN, FP, FN for each threshold τ in a vectorized manner.

            The algorithm sorts scores once and uses cumulative counts to derive confusion
            components for all

            Parameters
            ----------
            scores : ndarray of shape (n_samples,)
                Continuous model scores.

            y : ndarray of shape (n_samples,)
                Binary labels {0, 1}.

            tau_grid : ndarray of shape (n_tau,)
                Candidate thresholds.

            Returns
            -------
            TP : ndarray of shape (n_tau,)
            TN : ndarray of shape (n_tau,)
            FP : ndarray of shape (n_tau,)
            FN : ndarray of shape (n_tau,)
            """
        sort_idx = np.argsort(scores)
        scores_sorted = scores[sort_idx]
        y_sorted = y[sort_idx]
        y_int = (y_sorted == 1).astype(np.int8)

        cum_pos = np.cumsum(y_int)
        cum_neg = np.arange(1, len(y_int) + 1) - cum_pos

        total_pos = cum_pos[-1]
        total_neg = cum_neg[-1]

        split_idx = np.searchsorted(scores_sorted, tau_grid, side='left')
        split_idx = np.clip(split_idx, 0, len(scores_sorted))

        FN = np.where(split_idx > 0, cum_pos[split_idx - 1], 0)
        TN = np.where(split_idx > 0, cum_neg[split_idx - 1], 0)
        TP = total_pos - FN
        FP = total_neg - TN

        return TP, TN, FP, FN

    def _validate_input(self):

        """
        Validate and sanitize user-provided input parameters.

        Returns
        -------
        fit_params : dict
            Validated ``fit_params`` dictionary.

        tau_grid : ndarray
            Sorted threshold grid. Defaults to ``np.arange(-3, 3, 0.1)`` if none was provided.

        Raises
        ------
        ValueError
            If ``K < 2``.
        """

        if self.fit_params is None:
            fit_params = {}
        else:
            fit_params = self.fit_params

        if self.tau_grid is None:
            warnings.warn("tau_grid was not provided. Defaulting to [-3,3].")
            tau_grid = np.arange(-3, 3, 0.1)
        else:
            tau_grid = np.sort(np.asarray(self.tau_grid))

        if self.K < 2:
            raise ValueError("K must be >= 2.")

        return fit_params, tau_grid

    def _compute_taus(self):
        """
       Compute the optimal τ for each metric after cross-validation.

       For each metric matrix in ``metric_mats_``, selects τ maximizing or
       minimizing the metric as appropriate. Also computes the threshold
       minimizing the balanced error rate.

       Returns
       -------
       self : ThresholdOptimizer
       """
        self.best_tau_ = {}
        for m in self.available_metrics:
            mat = self.metric_mats_[m]
            maximize = not (m in ["fnr", "fpr", "misclassification_error"])
            self.best_tau_[m] = self._best(mat, maximize=maximize)
        fpr_mat = self.metric_mats_["fpr"]
        fnr_mat = self.metric_mats_["fnr"]
        self.best_tau_["balanced_error"] = self.tau_grid_[np.argmin(np.mean(fpr_mat + fnr_mat, axis=0))]
        return self

    def summary(self):
        """
        Summarize cross-validated performance at the best threshold for each metric.

        Returns
        -------
        summary : pandas.DataFrame
            A dataframe with columns:
            - ``metric`` : metric name
            - ``tau_best`` : selected threshold
            - ``mean_at_best`` : mean metric value at ``tau_best`` across folds
            - ``std_at_best`` : standard deviation across folds

        Raises
        ------
        RuntimeError
            If the optimizer has not been fitted.
        """

        if not self._is_fitted:
            raise RuntimeError("Call fit() before summary().")

        records = []

        for m in self.available_metrics + ["balanced_error"]:
            tau_best = self.best_tau_[m]
            idx = np.argmin(np.abs(self.tau_grid_ - tau_best))

            if m == "balanced_error":
                mat = (self.metric_mats_["fpr"] + self.metric_mats_["fnr"]) / 2
            else:
                mat = self.metric_mats_[m]

            mean_at_best = np.mean(mat[:, idx])
            std_at_best = np.std(mat[:, idx])

            records.append({
                "metric": m,
                "tau_best": tau_best,
                "mean_at_best": mean_at_best,
                "std_at_best": std_at_best,
            })

        return pd.DataFrame(records)

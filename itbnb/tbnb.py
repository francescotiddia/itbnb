import pickle
import warnings
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, ClassifierMixin

from .threshold import ThresholdOptimizer
from .utils.decision import *
from .utils.validation import (
    _validate_predict_inputs,
    _validate_fit_inputs,
    check_priors,
)


class TbNB(ClassifierMixin, BaseEstimator):
    """
    Threshold-based Naïve Bayes (Tb-NB) classifier.

    This classifier implements a Naïve Bayes model in which the
    final decision is taken by comparing the log-likelihood ratio score
    against an optimized threshold τ. Threshold optimization can be performed
    via cross-validation, and an optional iterative refinement procedure
    (iTb-NB) can be enabled.

    Parameters
    ----------
    fit_prior : bool, default=True
        Whether to estimate class prior probabilities from the training data.
        If ``False``, `class_prior` must be provided.

    class_prior : array-like of shape (2,), optional
        Prior probabilities for the classes ``[P(y=0), P(y=1)]``.
        Used only when ``fit_prior=False``.

    alpha : float, default=1
        Additive (Laplace) smoothing parameter applied to conditional
        feature counts.

    iterative : bool, default=False
        If ``True``, applies the iterative thresholding procedure (iTb-NB).
        Requires a fitted threshold.

    optimize_threshold : bool, default=True
        Whether to perform threshold optimization during method .fit()

    criterion : string
        Metric used to select the optimal decision threshold. Must be one
        self.AVAILABLE_CRITERIA

    tau_grid : {"observed", tuple of float, array-like}, default="observed"
        If ``"observed"``, threshold candidates are generated from empirical
        score percentiles (1st–99th).
        If a tuple ``(low, high)``, thresholds are generated uniformly from
        that interval.
        If array-like, the values are interpreted as an explicit list of
        thresholds to evaluate.

    K : int, default=5
        Number of cross-validation folds used for threshold optimization.

    n_tau : int, default=50
        Number of threshold candidates to evaluate.

    random_state : int, default=42
        Random seed used for cross-validation splits.

    p_iter : float, default=0.2
        Parameter controlling the proportion of samples around the current
        threshold used to define the uncertainty region at each iteration.

    s_iter : int, default=20
        Minimum number of observations required for an iterative refinement step.
        When ``mode="kde"``, this is the minimum total samples in the window.
        When ``mode="clt"``, this is interpreted as the minimum samples **per class**
        to ensure CLT validity. Recommended: >= 15 for KDE, >= 30 for CLT.

    mode : {"kde", "clt"}, default="kde"
        Method used to estimate class score densities within the uncertainty
        region during iterative refinement:
        - "kde": Gaussian kernel density estimation (non-parametric, works with fewer samples)
        - "clt": Normal approximation via Central Limit Theorem (parametric, more stable, requires more data)

    clt_n_boot : int, default=500
        Number of bootstrap resamples used when ``mode="clt"``.
        Only relevant for iterative refinement with CLT-based density estimation.
        Higher values (e.g., 500-1000) provide more stable estimates but increase
        computational cost. Ignored when ``mode="kde"``.

    clt_sample_size : int, default=30
        Size of each bootstrap sample when ``mode="clt"``. This is the sample size
        used in each bootstrap iteration to compute means. Must be >= 30 for valid
        CLT approximation. Controls the variance of bootstrap means: smaller values
        give wider Normal distributions, larger values give narrower ones.
        Ignored when ``mode="kde"``.

    Attributes
    ----------
    classes_ : ndarray of shape (2,)
        Class labels seen during .fit()

    n_features_in_ : int
        Number of features in the input dataset.

    class_prior_ : ndarray of shape (2,)
        Estimated or user-provided class prior probabilities.

    class_occurrences_ : ndarray of shape (2,)
        Number of samples observed for each class.

    feature_counts_ : ndarray of shape (2, n_features)
        Feature occurrence counts conditional on class.

    log_conditional_pres_ : ndarray of shape (2, n_features)
        Log-odds of feature presence given each class.

    log_conditional_abs_ : ndarray of shape (2, n_features)
        Log-probabilities of feature absence given each class.

    threshold_ : float
        Selected decision threshold τ. Available only if
        ``optimize_threshold=True``

    optimizer_ : ThresholdOptimizer
        Fitted threshold optimizer (when threshold optimization is enabled).

    decisions_ : list of Decision
        Sequence of iterative threshold refinement steps, available
        only when ``iterative=True``.

    References
    ----------
    Romano, M., Zammarchi, G., & Conversano, C. (2024).
    *Iterative Threshold-Based Naïve Bayes Classifier*.
    Statistical Methods & Applications, 33, 235–265.
    https://doi.org/10.1007/s10260-023-00721-1
    """

    AVAILABLE_CRITERIA = ['precision',
                          'recall',
                          'specificity',
                          'fpr',
                          'f1',
                          'mcc',
                          'misclassification_error',
                          'fnr',
                          'accuracy',
                          'balanced_error']

    def __init__(
        self,
        fit_prior=True,
        class_prior=None,
        alpha=1,
        iterative=False,
        optimize_threshold=True,
        criterion="balanced_error",
        tau_grid="observed",
        K=5,
        n_tau=50,
        random_state=42,
        p_iter=0.2,
        s_iter=20,
        mode="kde",
        clt_n_boot=500,
        clt_sample_size=30
    ):
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.alpha = alpha
        self.iterative = iterative
        self.optimize_threshold = optimize_threshold
        self.criterion = criterion
        self.tau_grid = tau_grid
        self.K = K
        self.n_tau = n_tau
        self.random_state = random_state
        self.p_iter = p_iter
        self.s_iter = s_iter
        self.mode = mode
        self.clt_n_boot = clt_n_boot
        self.clt_sample_size = clt_sample_size

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.classifier_tags.multi_class = False
        tags.classifier_tags.multi_label = False
        tags.input_tags.sparse = True
        tags.target_tags.required = True
        tags.classifier_tags.poor_score = True
        return tags

    def best_tau(self, metric):
        if not hasattr(self, "optimizer_"):
            raise RuntimeError("Threshold has not been optimized yet. Call fit(..., optimize_threshold=True).")
        return self.optimizer_.best_tau_[metric]

    @property
    def lw_present_(self):
        return self.log_conditional_pres_[1] - self.log_conditional_pres_[0]

    @property
    def lw_absent_(self):
        return self.log_conditional_abs_[1] - self.log_conditional_abs_[0]

    @staticmethod
    def _estimate_prior(Y: np.ndarray):
        """
        Estimate class occurrence counts and prior probabilities.

        Parameters
        ----------
        Y : one-hot encoded ndarray of shape (n_samples, 2)

        Returns
        -------
        class_occurrences : ndarray of shape (2,)
            Count of samples per class.
        class_prior : ndarray of shape (2,)
            Empirical class prior probabilities.
        """
        class_occurrences = Y.sum(axis=0)
        n = class_occurrences.sum()
        class_prior = class_occurrences / n
        return class_occurrences, class_prior

    @staticmethod
    def _estimate_feature_counts(X: csr_matrix, Y: np.ndarray):
        """
        Compute feature counts conditional on class.

        Parameters
        ----------
        X : csr_matrix of shape (n_samples, n_features)
            Binary Bag-of-Words matrix (presence/absence of terms).
        Y : ndarray of shape (n_samples, 2)
            Binary indicator matrix of class labels.

        Returns
        -------
        feature_counts : ndarray of shape (2, n_features)
            Count of each feature across positive and negative samples.
        """

        feature_counts = Y.T @ X
        smooth_fc = np.asarray(feature_counts)
        return smooth_fc

    def _estimate_log_conditional_probabilities(self, feature_counts, alpha=1.0):
        """
        Estimate log-conditional probabilities with Laplace smoothing.

        For each class, computes:
        P(w | class) = (count + α) / (N_class + 2α)

        The logarithms of both the presence and absence likelihoods are returned,
        which are later used to compute log-likelihood ratios.

        Parameters
        ----------
        feature_counts : ndarray of shape (2, n_features)
            Class-specific feature occurrence counts.
        alpha : float, default=1.0
            Smoothing parameter to prevent zero probabilities.

        Returns
        -------
        log_likelihood_pres : ndarray of shape (2, n_features)
            Log-probabilities of word presence given each class.
        log_likelihood_abs : ndarray of shape (2, n_features)
            Log-probabilities of word absence given each class.
        """

        pres_counts = feature_counts
        abs_counts = self.class_occurrences_[:, None] - feature_counts

        pres_likelihood = (pres_counts + alpha) / (self.class_occurrences_[:, None] + 2 * alpha)
        abs_likelihood = (abs_counts + alpha) / (self.class_occurrences_[:, None] + 2 * alpha)

        log_likelihood_pres = np.log(pres_likelihood)
        log_likelihood_abs = np.log(abs_likelihood)

        return log_likelihood_pres, log_likelihood_abs

    def fit(self, X, y):
        """
        Fit the Tb-NB classifier.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Bag-of-Words feature matrix.

        y : array-like of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : TbNB
            Fitted estimator.
        """
        if self.criterion not in self.AVAILABLE_CRITERIA:
            raise ValueError("{} is not an available criterion.".format(self.criterion))

        result = _validate_fit_inputs(self, X, y)
        X = result[0]
        y_data = result[1]

        if y_data is None:
            return self

        y, classes = y_data

        self.classes_ = classes
        self.n_features_in_ = X.shape[1]

        alpha = self.alpha

        Y = np.column_stack((1 - y, y))

        if self.fit_prior:
            self.class_occurrences_, self.class_prior_ = self._estimate_prior(Y)
        else:
            check_priors(self.class_prior)
            self.class_occurrences_ = Y.sum(axis=0)
            self.class_prior_ = self.class_prior

        self.feature_counts_ = self._estimate_feature_counts(X, Y)
        self.log_conditional_pres_, self.log_conditional_abs_ = \
            self._estimate_log_conditional_probabilities(self.feature_counts_, alpha)

        scores = self.decision_function(X)

        if isinstance(self.tau_grid, str) and self.tau_grid == "observed":
            low, high = np.percentile(scores, [1, 99])
            tau_grid = np.linspace(low, high, self.n_tau)
        elif isinstance(self.tau_grid, tuple) and len(self.tau_grid) == 2:
            tau_grid = np.linspace(self.tau_grid[0], self.tau_grid[1], self.n_tau)
        elif isinstance(self.tau_grid, (list, np.ndarray)):
            tau_grid = np.sort(np.asarray(self.tau_grid))
        else:
            raise ValueError("tau_grid must be 'observed', a (low, high) tuple, or array-like.")

        if self.optimize_threshold:
            optimizer = ThresholdOptimizer(
                tau_grid=tau_grid,
                estimator_class=TbNB,
                estimator_params={
                    "fit_prior": self.fit_prior,
                    "class_prior": self.class_prior,
                    "alpha": self.alpha,
                    "iterative": False,
                    "optimize_threshold": False,
                    "criterion": self.criterion,
                    "tau_grid": self.tau_grid,
                    "K": self.K,
                    "n_tau": self.n_tau,
                    "random_state": self.random_state,
                },
                K=self.K,
                random_state=self.random_state,
            )

            optimizer.fit(X, y)
            self.optimizer_ = optimizer
            self.threshold_ = self.optimizer_.best_tau_[self.criterion]

        if self.iterative:
            self.decisions_ = iterate_threshold(
                X=scores,
                Y=y,
                tau=self.threshold_,
                p=self.p_iter,
                s=self.s_iter,
                mode=self.mode,
                clt_boot=self.clt_n_boot,
                clt_sample=self.clt_sample_size
            )

        return self

    def save_model(self, filename):
        """
        Save the fitted model to disk using pickle.

        Parameters
        ----------
        filename : str
            Path to the output file.
        """

        with open(filename, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """
        Load a pickled Tb-NB model from disk.

        Parameters
        ----------
        filename : str
            Path to the pickle file.

        Returns
        -------
        model : TbNB
            Loaded classifier instance.
        """
        with open(filename, "rb") as f:
            return pickle.load(f)

    def decision_function(self, X: csr_matrix):
        """
        Compute log-likelihood ratio scores for samples.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Log-likelihood ratio scores used for thresholding.
        """
        X = _validate_predict_inputs(self, X)
        Lw = self.lw_present_
        Lnw = self.lw_absent_

        delta = Lw - Lnw
        bias = np.sum(Lnw)

        scores = X @ delta
        scores = np.asarray(scores).ravel() + bias

        return scores

    def predict(self, X, threshold=None, iterative=None):
        """
        Predict class labels for samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input samples.

        threshold : float, optional
            Decision threshold τ.
            If ``None``, uses ``self.threshold_`` estimated during ``fit()``.
            If the model was fitted with ``optimize_threshold=False`` and
            no threshold is passed, a ``ValueError`` is raised.

        iterative : bool, optional
            If ``None``, uses the ``iterative`` setting defined at initialization
            (i.e., ``self.iterative``).
            If provided, overrides the model's default iterative mode without
            modifying the internal state.
            This allows performing standard TbNB predictions even when the model
            was fitted with ``iterative=True``.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted class labels.
        """

        scores = self.decision_function(X)

        use_iterative = self.iterative if iterative is None else iterative

        tau = threshold if threshold is not None else getattr(self, "threshold_", None)
        if tau is None:
            raise ValueError("No threshold available. Fit the model with optimize_threshold=True"
                             " or provide a threshold explicitly.")

        pred = (scores >= tau).astype(int)

        if not use_iterative:
            return self.classes_[pred]

        if not hasattr(self, "decisions_"):
            warnings.warn("Iterative mode enabled but no decisions_ found.")
            return self.classes_[pred]

        pred_int = predict_from_decisions(scores, self.decisions_, default_threshold=tau)
        return self.classes_[pred_int]
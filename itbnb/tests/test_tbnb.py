"""
Comprehensive test suite for TbNB (Threshold-based Naive Bayes) classifier.

This module tests the core functionality, parameter validation, iterative
refinement, and edge cases of the TbNB classifier.
"""

import numpy as np
import pytest
import warnings
from scipy.sparse import csr_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.utils._testing import assert_allclose, assert_array_equal
from itbnb import TbNB


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def binary_classification_data():
    """Generate simple binary classification dataset."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
        flip_y=0.1
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def sparse_bow_data():
    """Generate sparse bag-of-words data."""
    corpus = [
                 "great movie loved acting",
                 "terrible film waste time",
                 "excellent performance amazing",
                 "boring predictable disappointing",
                 "fantastic brilliant masterpiece",
                 "awful horrible regret watching",
             ] * 50  # Repeat for more samples

    labels = [1, 0, 1, 0, 1, 0] * 50

    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(corpus)
    y = np.array(labels)

    return train_test_split(X, y, test_size=0.2, random_state=42)


@pytest.fixture
def small_dataset():
    """Generate very small dataset for edge case testing."""
    X = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0], [0, 0, 1]])
    y = np.array([1, 0, 1, 0])
    return X, y


# ============================================================================
# Basic Functionality Tests
# ============================================================================

def test_tbnb_initialization():
    """Test TbNB initialization with default parameters."""
    clf = TbNB()

    assert clf.fit_prior is True
    assert clf.alpha == 1
    assert clf.iterative is False
    assert clf.optimize_threshold is True
    assert clf.criterion == "balanced_error"
    assert clf.K == 5
    assert clf.n_tau == 50
    assert clf.p_iter == 0.2
    assert clf.s_iter == 20
    assert clf.mode == "kde"
    assert clf.clt_n_boot == 500
    assert clf.clt_sample_size == 30


def test_tbnb_fit_basic(binary_classification_data):
    """Test basic fitting functionality."""
    X_train, X_test, y_train, y_test = binary_classification_data

    clf = TbNB(optimize_threshold=True, iterative=False)
    clf.fit(X_train, y_train)

    # Check that model is fitted
    assert hasattr(clf, "classes_")
    assert hasattr(clf, "n_features_in_")
    assert hasattr(clf, "class_prior_")
    assert hasattr(clf, "class_occurrences_")
    assert hasattr(clf, "feature_counts_")
    assert hasattr(clf, "log_conditional_pres_")
    assert hasattr(clf, "log_conditional_abs_")
    assert hasattr(clf, "threshold_")
    assert hasattr(clf, "optimizer_")

    # Check shapes
    assert clf.classes_.shape == (2,)
    assert clf.n_features_in_ == X_train.shape[1]
    assert clf.class_prior_.shape == (2,)
    assert clf.feature_counts_.shape == (2, X_train.shape[1])


def test_tbnb_predict(binary_classification_data):
    """Test prediction functionality."""
    X_train, X_test, y_train, y_test = binary_classification_data

    clf = TbNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Check prediction shape and type
    assert y_pred.shape == (X_test.shape[0],)
    assert y_pred.dtype == y_test.dtype
    assert set(y_pred).issubset(set(clf.classes_))

    # Check that predictions are reasonable (> 50% accuracy)
    accuracy = np.mean(y_pred == y_test)
    assert accuracy > 0.5


def test_tbnb_decision_function(binary_classification_data):
    """Test decision function outputs."""
    X_train, X_test, y_train, y_test = binary_classification_data

    clf = TbNB()
    clf.fit(X_train, y_train)
    scores = clf.decision_function(X_test)

    # Check shape and type
    assert scores.shape == (X_test.shape[0],)
    assert isinstance(scores, np.ndarray)

    # Check that scores are continuous
    assert len(np.unique(scores)) > 2


def test_tbnb_sparse_input(sparse_bow_data):
    """Test TbNB with sparse matrices."""
    X_train, X_test, y_train, y_test = sparse_bow_data

    assert isinstance(X_train, csr_matrix)

    clf = TbNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert y_pred.shape == (X_test.shape[0],)
    assert np.mean(y_pred == y_test) > 0.5


# ============================================================================
# Iterative Mode Tests
# ============================================================================

def test_tbnb_iterative_kde(binary_classification_data):
    """Test iterative refinement with KDE mode."""
    X_train, X_test, y_train, y_test = binary_classification_data

    clf = TbNB(iterative=True, mode="kde", s_iter=15)
    clf.fit(X_train, y_train)

    # Check that decisions were created
    assert hasattr(clf, "decisions_")
    assert isinstance(clf.decisions_, list)

    # Predict with iterative mode
    y_pred = clf.predict(X_test)
    assert y_pred.shape == (X_test.shape[0],)


def test_tbnb_iterative_clt(binary_classification_data):
    """Test iterative refinement with CLT mode."""
    X_train, X_test, y_train, y_test = binary_classification_data

    clf = TbNB(
        iterative=True,
        mode="clt",
        s_iter=30,
        clt_n_boot=100,  # Reduced for speed
        clt_sample_size=30
    )
    clf.fit(X_train, y_train)

    # Check that decisions were created
    assert hasattr(clf, "decisions_")
    assert isinstance(clf.decisions_, list)

    # Each decision should have proper structure
    if len(clf.decisions_) > 0:
        decision = clf.decisions_[0]
        assert hasattr(decision, "iteration")
        assert hasattr(decision, "start")
        assert hasattr(decision, "end")
        assert hasattr(decision, "tau")
        assert hasattr(decision, "x_max_pos")
        assert hasattr(decision, "x_max_neg")
        assert hasattr(decision, "direction")
        assert decision.direction in ["r", "l"]


def test_iterative_improves_or_maintains_performance(binary_classification_data):
    """Test that iterative mode doesn't significantly degrade performance."""
    X_train, X_test, y_train, y_test = binary_classification_data

    # Standard TbNB
    clf_standard = TbNB(iterative=False)
    clf_standard.fit(X_train, y_train)
    y_pred_standard = clf_standard.predict(X_test)
    acc_standard = np.mean(y_pred_standard == y_test)

    # Iterative TbNB
    clf_iterative = TbNB(iterative=True, mode="kde")
    clf_iterative.fit(X_train, y_train)
    y_pred_iterative = clf_iterative.predict(X_test)
    acc_iterative = np.mean(y_pred_iterative == y_test)

    # Iterative should not be significantly worse (allow 5% tolerance)
    assert acc_iterative >= acc_standard - 0.05


def test_predict_with_iterative_override(binary_classification_data):
    """Test prediction with iterative mode override."""
    X_train, X_test, y_train, y_test = binary_classification_data

    clf = TbNB(iterative=True, mode="kde")
    clf.fit(X_train, y_train)

    # Predict with iterative=False override
    y_pred_no_iter = clf.predict(X_test, iterative=False)

    # Predict with iterative=True (default)
    y_pred_iter = clf.predict(X_test, iterative=True)

    # Predictions may differ
    assert y_pred_no_iter.shape == y_pred_iter.shape


# ============================================================================
# Parameter Validation Tests
# ============================================================================

def test_invalid_criterion_raises_error(binary_classification_data):
    """Test that invalid criterion raises ValueError."""
    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB(criterion="invalid_metric")

    with pytest.raises(ValueError, match="not an available criterion"):
        clf.fit(X_train, y_train)


def test_invalid_mode_raises_error(binary_classification_data):
    """Test that invalid mode raises ValueError."""
    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB(iterative=True, mode="invalid_mode")

    with pytest.raises(ValueError, match="mode must be"):
        clf.fit(X_train, y_train)


def test_predict_without_fit_raises_error(binary_classification_data):
    """Test that predict without fit raises appropriate error."""
    _, X_test, _, _ = binary_classification_data

    clf = TbNB()

    with pytest.raises(AttributeError):
        clf.predict(X_test)


def test_predict_without_threshold_raises_error(binary_classification_data):
    """Test that predict without threshold optimization raises error."""
    X_train, X_test, y_train, _ = binary_classification_data

    clf = TbNB(optimize_threshold=False)
    clf.fit(X_train, y_train)

    with pytest.raises(ValueError, match="No threshold available"):
        clf.predict(X_test)


def test_custom_threshold_prediction(binary_classification_data):
    """Test prediction with custom threshold."""
    X_train, X_test, y_train, _ = binary_classification_data

    clf = TbNB(optimize_threshold=False)
    clf.fit(X_train, y_train)

    # Should work with explicit threshold
    y_pred = clf.predict(X_test, threshold=0.0)
    assert y_pred.shape == (X_test.shape[0],)


# ============================================================================
# Threshold Optimization Tests
# ============================================================================

def test_threshold_optimization_criteria(binary_classification_data):
    """Test different optimization criteria."""
    X_train, X_test, y_train, y_test = binary_classification_data

    criteria = ["precision", "recall", "f1", "accuracy", "balanced_error"]

    for criterion in criteria:
        clf = TbNB(criterion=criterion)
        clf.fit(X_train, y_train)

        assert hasattr(clf, "threshold_")
        assert hasattr(clf, "optimizer_")

        y_pred = clf.predict(X_test)
        assert y_pred.shape == (X_test.shape[0],)


def test_best_tau_method(binary_classification_data):
    """Test best_tau method for different metrics."""
    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB()
    clf.fit(X_train, y_train)

    # Should be able to retrieve best tau for any metric
    for metric in clf.AVAILABLE_CRITERIA:
        tau = clf.best_tau(metric)
        assert isinstance(tau, (int, float))


def test_best_tau_without_optimization_raises_error(binary_classification_data):
    """Test that best_tau raises error when threshold not optimized."""
    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB(optimize_threshold=False)
    clf.fit(X_train, y_train)

    with pytest.raises(RuntimeError, match="Threshold has not been optimized"):
        clf.best_tau("f1")


def test_tau_grid_types(binary_classification_data):
    """Test different tau_grid specifications."""
    X_train, _, y_train, _ = binary_classification_data

    # Test "observed"
    clf1 = TbNB(tau_grid="observed")
    clf1.fit(X_train, y_train)
    assert hasattr(clf1, "threshold_")

    # Test tuple
    clf2 = TbNB(tau_grid=(-5.0, 5.0))
    clf2.fit(X_train, y_train)
    assert hasattr(clf2, "threshold_")

    # Test array
    clf3 = TbNB(tau_grid=np.linspace(-3, 3, 20))
    clf3.fit(X_train, y_train)
    assert hasattr(clf3, "threshold_")


# ============================================================================
# Prior and Smoothing Tests
# ============================================================================

def test_fit_prior_true(binary_classification_data):
    """Test with fit_prior=True."""
    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB(fit_prior=True)
    clf.fit(X_train, y_train)

    # Prior should be estimated from data
    assert hasattr(clf, "class_prior_")
    assert_allclose(clf.class_prior_.sum(), 1.0)


def test_fit_prior_false(binary_classification_data):
    """Test with fit_prior=False and custom priors."""
    X_train, _, y_train, _ = binary_classification_data

    custom_prior = np.array([0.3, 0.7])
    clf = TbNB(fit_prior=False, class_prior=custom_prior)
    clf.fit(X_train, y_train)

    assert_array_equal(clf.class_prior_, custom_prior)


def test_different_alpha_values(binary_classification_data):
    """Test with different smoothing parameters."""
    X_train, X_test, y_train, y_test = binary_classification_data

    alphas = [0.01, 0.1, 1.0, 10.0]

    for alpha in alphas:
        clf = TbNB(alpha=alpha)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        # Should produce valid predictions
        assert y_pred.shape == (X_test.shape[0],)
        assert set(y_pred).issubset(set([0, 1]))


# ============================================================================
# Edge Cases and Robustness Tests
# ============================================================================

def test_small_dataset(small_dataset):
    """Test TbNB on very small dataset."""
    X, y = small_dataset
    clf = TbNB(iterative=False, n_tau=5, K=2)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    assert y_pred.shape == (X.shape[0],)


def test_balanced_vs_imbalanced_data():
    """Test TbNB on imbalanced dataset."""
    X, y = make_classification(
        n_samples=400,
        n_features=10,
        n_classes=2,
        weights=[0.9, 0.1],
        random_state=42
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    clf = TbNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    assert len(np.unique(y_pred)) <= 2


def test_deterministic_with_random_state(binary_classification_data):
    """Test that results are deterministic with fixed random_state."""
    X_train, X_test, y_train, _ = binary_classification_data

    clf1 = TbNB(random_state=42)
    clf1.fit(X_train, y_train)
    y_pred1 = clf1.predict(X_test)

    clf2 = TbNB(random_state=42)
    clf2.fit(X_train, y_train)
    y_pred2 = clf2.predict(X_test)

    assert_array_equal(y_pred1, y_pred2)


def test_single_feature(binary_classification_data):
    """Test TbNB with single feature."""
    X_train, X_test, y_train, y_test = binary_classification_data

    X_train_single = X_train[:, :1]
    X_test_single = X_test[:, :1]

    clf = TbNB()
    clf.fit(X_train_single, y_train)
    y_pred = clf.predict(X_test_single)

    assert y_pred.shape == (X_test_single.shape[0],)


def test_all_same_class():
    """Test behavior when all samples are same class."""
    X = np.random.randn(50, 10)
    y = np.ones(50, dtype=int)  #

    clf = TbNB(n_tau=5)

    # Should handle gracefully (may raise warning but shouldn't crash)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            clf.fit(X, y)
            # If fit succeeds, predictions should all be the same class
            y_pred = clf.predict(X)
            assert len(np.unique(y_pred)) == 1
        except ValueError:
            # It's also acceptable to raise ValueError for degenerate case
            pass


# ============================================================================
# Properties and Attributes Tests
# ============================================================================

def test_lw_properties(binary_classification_data):
    """Test lw_present_ and lw_absent_ properties."""
    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB()
    clf.fit(X_train, y_train)

    # Check properties exist and have correct shape
    assert clf.lw_present_.shape == (X_train.shape[1],)
    assert clf.lw_absent_.shape == (X_train.shape[1],)

    # Check they're computed correctly
    expected_lw_pres = (
            clf.log_conditional_pres_[1] - clf.log_conditional_pres_[0]
    )
    expected_lw_abs = (
            clf.log_conditional_abs_[1] - clf.log_conditional_abs_[0]
    )

    assert_allclose(clf.lw_present_, expected_lw_pres)
    assert_allclose(clf.lw_absent_, expected_lw_abs)


def test_sklearn_tags():
    """Test sklearn tags are properly set."""
    clf = TbNB()
    tags = clf.__sklearn_tags__()

    assert tags.classifier_tags.multi_class is False
    assert tags.classifier_tags.multi_label is False
    assert tags.input_tags.sparse is True


# ============================================================================
# Serialization Tests
# ============================================================================

def test_save_and_load_model(binary_classification_data, tmp_path):
    """Test model serialization and deserialization."""
    X_train, X_test, y_train, _ = binary_classification_data

    # Train and save model
    clf = TbNB(iterative=True, mode="kde")
    clf.fit(X_train, y_train)

    model_path = tmp_path / "tbnb_model.pkl"
    clf.save_model(str(model_path))

    # Load model
    clf_loaded = TbNB.load(str(model_path))

    # Check predictions match
    y_pred_original = clf.predict(X_test)
    y_pred_loaded = clf_loaded.predict(X_test)

    assert_array_equal(y_pred_original, y_pred_loaded)

    # Check attributes preserved
    assert clf_loaded.threshold_ == clf.threshold_
    assert clf_loaded.mode == clf.mode


# ============================================================================
# Integration Tests
# ============================================================================

def test_pipeline_compatibility(binary_classification_data):
    """Test TbNB in sklearn pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    X_train, X_test, y_train, _ = binary_classification_data

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", TbNB())
    ])

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)

    assert y_pred.shape == (X_test.shape[0],)


def test_cross_validation_compatibility(binary_classification_data):
    """Test TbNB with cross-validation."""
    from sklearn.model_selection import cross_val_score

    X_train, _, y_train, _ = binary_classification_data

    clf = TbNB()
    scores = cross_val_score(clf, X_train, y_train, cv=3)

    assert scores.shape == (3,)
    assert all(0 <= score <= 1 for score in scores)


# ============================================================================
# CLT-specific Tests
# ============================================================================

def test_clt_bootstrap_parameters():
    """Test CLT mode with different bootstrap parameters."""
    X, y = make_classification(
        n_samples=500,
        n_features=20,
        n_informative=15,
        random_state=42
    )

    # Test with different n_boot values
    clf1 = TbNB(iterative=True, mode="clt", clt_n_boot=100)
    clf1.fit(X, y)

    clf2 = TbNB(iterative=True, mode="clt", clt_n_boot=1000)
    clf2.fit(X, y)

    # Both should work
    assert hasattr(clf1, "decisions_")
    assert hasattr(clf2, "decisions_")


def test_clt_sample_size_constraint():
    """Test that CLT respects minimum sample size."""
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        random_state=42
    )

    # With s_iter=30 and mode="clt", needs 30 samples per class in window
    clf = TbNB(
        iterative=True,
        mode="clt",
        s_iter=30,
        clt_sample_size=30,
        p_iter=0.1  # Small window
    )
    clf.fit(X, y)

    # Should complete without error
    assert hasattr(clf, "decisions_")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
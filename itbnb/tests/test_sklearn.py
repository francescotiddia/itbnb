from sklearn.utils.estimator_checks import parametrize_with_checks

from itbnb import TbNB


@parametrize_with_checks([TbNB()])
def test_my_estimator(estimator, check):
    check(estimator)

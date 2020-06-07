from linreg import linear_regression
import random

def test_linregression():
    def target(x): return 5 * x - 3.2
    X = [random.uniform(0, 1) for _ in range(500)]
    y = [target(x) + random.uniform(-1, 1) for x in X]

    slope, intercept = linear_regression(X, y)
    assert (slope - 5) <= 1.e-3
    assert (intercept + 3.2) <= 0.1

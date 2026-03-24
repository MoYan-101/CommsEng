"""
models/model_svm.py
Multi-output SVM regression wrapper using scikit-learn.
"""

from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor


class SVMRegression:
    """Multi-output SVM regressor (one SVR per target)."""

    def __init__(self, kernel="rbf", C=1.0, epsilon=0.1, gamma="scale",
                 degree=3, coef0=0.0, max_iter=-1, n_jobs=-1):
        base = SVR(
            kernel=kernel,
            C=C,
            epsilon=epsilon,
            gamma=gamma,
            degree=degree,
            coef0=coef0,
            max_iter=max_iter
        )
        self.model = MultiOutputRegressor(base, n_jobs=n_jobs)

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

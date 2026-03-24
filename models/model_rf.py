"""
models/model_rf.py

A Random Forest regressor wrapper using scikit-learn.
"""

from sklearn.ensemble import RandomForestRegressor

class RFRegression:
    """
    A simple regression model using RandomForest from scikit-learn.
    """

    def __init__(self, n_estimators=100, max_depth=10, random_state=42,
                 ccp_alpha=0.0, min_samples_leaf=1, n_jobs=-1):
        """
               :param n_estimators: number of trees in the forest
               :param max_depth: maximum depth of the trees
               :param random_state: seed for reproducibility
               """
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            ccp_alpha=ccp_alpha,
            min_samples_leaf=min_samples_leaf,
            n_jobs=n_jobs
        )

    def fit(self, X, Y):
        """
        Train the Random Forest model.
        :param X: training inputs, shape (N, input_dim)
        :param Y: training targets, shape (N, output_dim)
        """
        self.model.fit(X, Y)

    def predict(self, X):
        """
        Predict using the trained Random Forest model.
        :param X: inputs, shape (N, input_dim)
        :return: predictions, shape (N, output_dim)
        """
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        """
        Convenience proxy for external access via rf_model.feature_importances_.
        """
        return self.model.feature_importances_

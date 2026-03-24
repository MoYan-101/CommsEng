from sklearn.tree import DecisionTreeRegressor

class DTRegression:
    """
    Decision Tree with optional ccp_alpha (cost-complexity pruning).
    """
    def __init__(self, max_depth=None, random_state=42, ccp_alpha=0.0):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state,
            ccp_alpha=ccp_alpha
        )

    def fit(self, X, Y):
        self.model.fit(X, Y)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

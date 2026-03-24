"""
models/model_catboost.py
"""

from catboost import CatBoostRegressor
from catboost import Pool

class CatBoostRegression:
    """
    CatBoost with l2_leaf_reg controlling L2 regularization strength.
    """
    def __init__(self, iterations=100, learning_rate=0.1, depth=6,
                 random_seed=42, l2_leaf_reg=3.0, thread_count=-1,
                 random_strength=1.0, bagging_temperature=0.0,
                 rsm=1.0):
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            random_seed=random_seed,
            verbose=0,
            loss_function="MultiRMSE",
            l2_leaf_reg=l2_leaf_reg,
            thread_count=thread_count,
            random_strength=random_strength,
            bagging_temperature=bagging_temperature,
            rsm=rsm
        )

    def fit(self, X, Y, **fit_kwargs):
        self.model.fit(X, Y, **fit_kwargs)

    def predict(self, X, **kwargs):
        return self.model.predict(X, **kwargs)

    # @property
    # def feature_importances_(self):
    #     return self.model.get_feature_importance(type="PredictionValuesChange")

    @property
    def feature_importances_(self):
        """
        Returns normalized feature importance values for CatBoost, ensuring
        the sum of all importances equals 1, making it comparable to
        feature importances from XGBoost, RandomForest, and DecisionTree.
        """
        importances = self.model.get_feature_importance(type="PredictionValuesChange")

        # Avoid division-by-zero errors.
        total_importance = sum(importances)
        if total_importance > 0:
            return importances / total_importance  # Normalize so the sum is 1
        else:
            return importances  # If all values are zero, return them as is

    def get_shap_values(self, X):
        pool = Pool(X)
        shap_values = self.model.get_feature_importance(type="ShapValues", data=pool)
        # If the result is multi-output (3D), for example (n_samples, n_outputs, n_features+1)
        if len(shap_values.shape) == 3:
            # Remove the last baseline column for each output if cols == n_features + 1.
            outputs = []
            for i in range(shap_values.shape[1]):
                if shap_values.shape[2] == X.shape[1] + 1:
                    outputs.append(shap_values[:, i, :-1])
                else:
                    outputs.append(shap_values[:, i, :])
            return outputs
        # If the result is 2D (n_samples, n_features+1)
        elif len(shap_values.shape) == 2:
            if shap_values.shape[1] == X.shape[1] + 1:
                shap_values = shap_values[:, :-1]
            return shap_values

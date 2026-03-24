"""
models/model_xgb.py
Thin wrapper around the scikit-learn API of XGBoost.
Uses the officially recommended early_stopping_rounds=set_params style;
compatible with Python 3.8/3.9.
"""

from typing import Optional, Any
from xgboost import XGBRegressor


class XGBRegression:
    def __init__(self,
                 n_estimators: int = 100,
                 learning_rate: float = 0.1,
                 max_depth: int = 6,
                 random_state: int = 42,
                 reg_alpha: float = 0.0,
                 reg_lambda: float = 1.0,
                 min_child_weight: float = 1.0,
                 gamma: float = 0.0,
                 subsample: float = 1.0,
                 colsample_bytree: float = 1.0,
                 n_jobs: int = -1,
                 early_stopping_rounds: Optional[int] = None
                 ):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            verbosity=0,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            n_jobs=n_jobs,
            objective="reg:squarederror",
        )

        # Officially recommended: call set_params after construction.
        if early_stopping_rounds is not None:
            self.model.set_params(early_stopping_rounds=early_stopping_rounds)

    # ------------------------------------------------------------------ #
    # scikit-learn-compatible interface
    # ------------------------------------------------------------------ #
    def set_params(self, **params: Any):
        """Transparent passthrough so callers can set early_stopping_rounds and similar params."""
        self.model.set_params(**params)
        return self

    def get_params(self, deep: bool = True):
        return self.model.get_params(deep=deep)

    def fit(self, X, y, **fit_kwargs):
        # Accept standard fit kwargs such as eval_set / sample_weight.
        eval_set = fit_kwargs.pop("eval_set", None)
        self.model.fit(X, y, eval_set=eval_set, **fit_kwargs)

    def predict(self, X):
        return self.model.predict(X)

    @property
    def feature_importances_(self):
        return self.model.feature_importances_

def train_sklearn_model(model,
                        X_train, Y_train,
                        X_val=None, Y_val=None,
                        enable_early_stop: bool = False,
                        es_rounds: int = 50):
    """
    Train a sklearn-style model (RF / DT) or a custom thin wrapper
    (CatBoostRegression / XGBRegression).
    """

    if enable_early_stop and X_val is not None:
        base_name = model.__class__.__name__
        if hasattr(model, "model"):
            base_name = model.model.__class__.__name__

        fit_kwargs = dict(
            eval_set=[(X_val, Y_val)],
            verbose=False
        )

        if base_name.startswith("CatBoost"):
            fit_kwargs["use_best_model"] = True
            fit_kwargs["early_stopping_rounds"] = es_rounds
        elif base_name.startswith("XGB"):
            # In XGBoost 3.x, early_stopping_rounds is set via set_params; fit does not accept it.
            pass
        else:
            fit_kwargs["early_stopping_rounds"] = es_rounds

        model.fit(X_train, Y_train, **fit_kwargs)
    else:
        model.fit(X_train, Y_train)

    return model

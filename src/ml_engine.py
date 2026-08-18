import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def train_pilot_model(X, y, train_fraction=0.2):
    """
    KEY REMINDERS:
    1.  Trains a personalized Random Forest model for one subject, using a
    stratified, per-condition chronological split.
    2.  The first 'train_fraction' of EACH workload condition's own epochs is
    used for training and not train_fraction of the whole concatenated
    sequence.
    3.  X and y here come from the same features/epochs already loaded in the GUI
    (self.features).

    Parameters:
        X: DataFrame of features, one row per epoch (e.g. self.features)
        y: array-like of the true condition label per row, same order as X
        train_fraction: fraction of EACH condition's own epochs used for
            training (default 0.2 - train on the first 20% of every
            condition, applied on the remaining 80%)

    Returns:
        model: fitted RandomForestClassifier, or None if calibration failed
        is_train_row: boolean numpy array, same length and order as X -
            True for every row used in training. The GUI should never
            display a live prediction for a row where this is True, since
            the model has already seen it.
        status: string describing what happened, for the status label
    """

    y = pd.Series(y).reset_index(drop=True)
    X = X.reset_index(drop=True)

    is_train_row = pd.Series(False, index=X.index)

    for condition in sorted(y.unique()):
        condition_idx = y[y == condition].index
        # max(1, ...) so a small condition doesn't silently contribute zero
        # training rows purely from rounding down so that every condition is present
        # should get at least one training example if it has any epochs at all.
        cutoff = max(1, int(len(condition_idx) * train_fraction))
        is_train_row.loc[condition_idx[:cutoff]] = True

    if is_train_row.sum() == 0:
        return None, None, "Not enough data to calibrate on"

    X_train = X[is_train_row.values]
    y_train = y[is_train_row.values]

    if y_train.nunique() < 2:
        return (
            None,
            None,
            "Calibration data doesn't cover enough distinct workload levels",
        )

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced", random_state=42
    )
    model.fit(X_train, y_train)

    return model, is_train_row.values, "Success"

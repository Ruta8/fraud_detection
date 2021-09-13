import numpy as np
import pandas as pd

from sklearn.metrics import (
    recall_score,
    precision_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import RandomizedSearchCV
from sklearn.compose import ColumnTransformer, make_column_selector


def perform_randomized_search(
    model,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    sample_weights: np.ndarray,
    parameter_grid: dict,
    scoring: str,
    cv: int,
):
    """Performs randomizes hyperparameter search for provided model.

    Args:
        model (sklearn.base.BaseEstimator): sklearn model.
        X_train (pd.DataFrame): train dataset.
        y_train (np.ndarray): train label.
        sample_weights (np.ndarray): sample weights.
        parameter_grid (dict): dictionary with parameters names (str) as keys and distributions or lists of parameters to try.
        scoring (str): a strategy to evaluate the performance of the cross-validated model on the test set.
        cv (int): determines the cross-validation splitting strategy.

    Returns:
        [type]: best parameters for the provided model
    """

    encoder = ColumnTransformer(
        transformers=[
            (
                "one_hot",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                make_column_selector(dtype_exclude=np.number),
            ),
        ],
        remainder="passthrough",
    )

    model_pipeline = Pipeline(steps=[("encoder", encoder), ("model", model),])

    randomized_search_model = RandomizedSearchCV(
        model_pipeline,
        param_distributions=parameter_grid,
        scoring=scoring,
        cv=cv,
        n_iter=60,
        random_state=42,
        n_jobs=4,
        verbose=4,
    )

    kwargs = {model_pipeline.steps[-1][0] + "__sample_weight": sample_weights}

    randomized_search_model.fit(X_train, y_train, **kwargs)

    return randomized_search_model


def evaluate_model(
    fraud_probabilities: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds_list: list,
) -> pd.DataFrame:
    """Computes threshold-based metrics for all given thresholds.
    Args:
        fraud_probabilities (np.ndarray): model predictions of fraud probability to be added as a column.
        X_test (np.ndarray): test dataset.
        y_test (np.ndarray): test labels.
        thresholds_list (list): a list of thresholds values to be iterated over.

    Returns:
        pd.DataFrame: a data frame with metric evaluations for every threshold.
    """

    results = []

    for threshold in thresholds_list:
        predicted_classes = [
            0 if fraud_probability < threshold else 1
            for fraud_probability in fraud_probabilities
        ]

        recall = recall_score(y_test, predicted_classes)
        precision = precision_score(y_test, predicted_classes)

        (TN, FP, FN, TP) = confusion_matrix(y_test, predicted_classes).ravel()
        total_workload = TP + FP

        X_test["predicted_classes"] = list(predicted_classes)
        X_test["label"] = y_test

        saved_money = round(
            X_test[X_test.label == 1][
                X_test.predicted_classes == 1
            ].transaction_amount.sum()
        )
        defrauded_money = round(X_test[X_test.label == 1].transaction_amount.sum())
        saved_money_perc = round((saved_money / defrauded_money) * 100)

        results.append(
            {
                "Theshold": threshold,
                "Recall": recall,
                "Precision": precision,
                "TP": TP,
                "FP": FP,
                "TN": TN,
                "FN": FN,
                "Total Workload": total_workload,
                "Saved Money": saved_money,
                "Defrauded Money": defrauded_money,
                "Saved Money %": saved_money_perc,
            }
        )

    return pd.DataFrame(results)


def choose_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    max_case_coverage: int = 400,
    increment: float = 0.001,
) -> float:
    """Choose the classification threshold for the classifier.

    Args:
       y_true (pd.Series): true dependant variable labels (0=non fraud, 1=fraud).
       y_pred_proba (pd.Series): probabilistic predict.
       max_case_coverage (int, optional): max number of cases handled in a month.
       increment (float, optional): value by which to modify threshold on each iteration.

    Returns:
       threshold (float): the optimal threshold to maximise recall given max_case_coverage.

    """
    assert len(y_true) == len(y_pred_proba), (
        "y_true and y_pred_proba of unequal length (%d and %d)"
        % len(y_true, len(y_pred_proba))
    )

    threshold = 0.5
    y_pred = (y_pred_proba > threshold).astype(int)
    assert (
        sum(y_pred) < max_case_coverage
    ), "0.5 threshold already gives more than max_case_coverage positive predictions."

    _, FP, _, TP = confusion_matrix(y_true, y_pred).ravel()
    num_positives = TP + FP

    while num_positives <= max_case_coverage:
        threshold -= increment
        y_pred = (y_pred_proba > threshold).astype(int)
        _, FP, _, TP = confusion_matrix(y_true, y_pred).ravel()
        num_positives = TP + FP

    threshold += increment
    return threshold


def return_test_with_predictions(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    best_threshold: float,
) -> pd.DataFrame:
    """Merges X_test and y_test back to a full test dataframe and adds columns for predictions.

    Args:
        X_test (pd.DataFrame): test set without the label.
        y_test (np.ndarray): test label or target.
        y_pred_proba (np.ndarray): fraud probabilities.
        best_threshold (float): the optimal decision value in a classification task, used to create binary prediction column.

    Returns:
        pd.DataFrame: a test dataframe with binary and probabilistc predictions.
    """
    results_df = X_test.copy()

    results_df["label"] = y_test
    results_df["fraud_probability"] = y_pred_proba
    results_df.loc[
        results_df["fraud_probability"] >= best_threshold, "predicted_class"
    ] = 1
    results_df.loc[
        results_df["fraud_probability"] < best_threshold, "predicted_class"
    ] = 0

    return results_df

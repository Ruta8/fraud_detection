import pandas as pd
import numpy as np


def add_account_spending_behaviour_features(
    account_transactions: pd.DataFrame,
    windows_size_in_days: list,
    transaction_time_col_name: str,
    transaction_amount_col_name: str,
    unique_transaction_id_col_name: str,
) -> pd.DataFrame:
    """Adds features that describe account spending behaviour. 
    The first half of the features describe the number of transactions by the customer in the
    last n day(s), for n in windows_size_in_days. The second half of the feature describes the
    average spending amount in the last n day(s), for n in windows_size_in_days.

    Args:
        account_transactions (pd.DataFrame): a set of transactions for an account number.
        windows_size_in_days (list):  a list of integer values representing a number of days for window size.
        transaction_time_col_name (str): non-local transaction time column name.
        transaction_amount_col_name (str): a column containing transaction amounts.
        unique_transaction_id_col_name (str): a column containing unique transaction ids.

    Returns:
        pd.DataFrame: data frame for an account number with additional features.
        The number of features is equal to windows_size_in_days x 2.
    """

    account_transactions = account_transactions.sort_values(transaction_time_col_name)
    account_transactions.index = account_transactions[transaction_time_col_name]

    for window_size in windows_size_in_days:
        sum_amount_transaction_window = (
            account_transactions[transaction_amount_col_name]
            .rolling(str(window_size) + "d")
            .sum()
        )
        nb_transaction_window = (
            account_transactions[transaction_amount_col_name]
            .rolling(str(window_size) + "d")
            .count()
        )

        avg_amount_transaction_window = (
            sum_amount_transaction_window / nb_transaction_window
        )

        account_transactions[
            "acccount_id_nb_tx_" + str(window_size) + "day_window"
        ] = list(nb_transaction_window)
        account_transactions[
            "account_id_avg_amout_" + str(window_size) + "day_window"
        ] = list(avg_amount_transaction_window)

    account_transactions.index = account_transactions[unique_transaction_id_col_name]
    account_transactions[
        account_transactions.select_dtypes(include=[np.number]).columns
    ] = account_transactions.select_dtypes(include=[np.number]).fillna(0)

    return account_transactions


def add_transaction_stats_features(
    account_transactions: pd.DataFrame,
    transaction_amount_col_name: str,
    transaction_time_col_name: str,
) -> pd.DataFrame:
    """The function takes as inputs the set of transactions for an account number.
    The function returns five new statistical features, calculated based on the transaction amount.

    Args:
        account_transactions (pd.DataFrame): a set of transactions for an account number.
        transaction_amount_col_name (str): a column containing transaction amounts.
        transaction_time_col_name (str): non-local transaction time column name.

    Returns:
        pd.DataFrame: data frame for an account number with 5 additional features.
    """

    account_transactions = account_transactions.sort_values(transaction_time_col_name)
    account_transactions["mean_transaction_amount"] = (
        account_transactions[transaction_amount_col_name].expanding().mean()
    )

    account_transactions["median_transaction_amount"] = (
        account_transactions[transaction_amount_col_name].expanding().median()
    )

    account_transactions["std_transaction_amount"] = (
        account_transactions[transaction_amount_col_name].expanding().std()
    )

    account_transactions["z_score_transaction_amount"] = (
        account_transactions[transaction_amount_col_name]
        - account_transactions.mean_transaction_amount
    ) / account_transactions.std_transaction_amount

    account_transactions["transaction_amount_delta"] = (
        account_transactions[transaction_amount_col_name]
        - account_transactions.mean_transaction_amount
    )

    return account_transactions


def add_transaction_time_diff_features(
    account_transactions: pd.DataFrame,
    transaction_time_col_name: str,
    unique_transaction_id_col_name: str,
) -> pd.DataFrame:
    """The function takes as inputs the set of transactions for an account number. 
    It returns a data frame with the five new statistical features, based on transaction time.

    Args:
        account_transactions (pd.DataFrame): a set of transactions for an account number.
        transaction_time_col_name (str): non-local transaction time column name.
        unique_transaction_id_col_name (str): a column containing unique transaction ids.

    Returns:
        pd.DataFrame: data frame for an account number with 5 additional features.
    """

    account_transactions = account_transactions.sort_values(transaction_time_col_name)
    account_transactions["transaction_hours_diff"] = (
        account_transactions[transaction_time_col_name].diff().dt.total_seconds()
    ) / 3600
    account_transactions[
        "mean_transaction_freq"
    ] = account_transactions.transaction_hours_diff.expanding().mean()
    account_transactions[
        "std_transaction_freq"
    ] = account_transactions.transaction_hours_diff.expanding().std()
    account_transactions["transaction_freq_delta"] = (
        account_transactions.transaction_hours_diff
        - account_transactions.mean_transaction_freq
    )
    account_transactions["z_score_transaction_freq"] = (
        account_transactions.transaction_hours_diff
        - account_transactions.mean_transaction_freq
    ) / account_transactions.std_transaction_freq

    account_transactions.index = account_transactions[unique_transaction_id_col_name]

    return account_transactions

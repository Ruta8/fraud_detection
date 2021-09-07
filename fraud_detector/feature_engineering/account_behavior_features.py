import pandas as pd
import numpy as np


def add_account_spending_behaviour_features(
    account_transactions: pd.DataFrame,
    windows_size_in_days: list,
    transaction_time_col_name: str,
    transaction_amount_col_name: str,
    unique_transaction_id_col_name: str,
) -> pd.DataFrame:

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

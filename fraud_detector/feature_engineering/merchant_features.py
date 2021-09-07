import pandas as pd


def get_fraud_per_reported_date(
    merchant_transactions: pd.DataFrame,
) -> pd.DataFrame:
    fraudulent_merchant_transactions = merchant_transactions[
        merchant_transactions.label == 1
    ]

    fraud_per_reported_date = pd.DataFrame(
        fraudulent_merchant_transactions.groupby(by=["reported_date"])
        .size()
        .expanding()
        .sum()
    )

    fraud_per_reported_date.rename(
        columns={0: "merchant_fraud_transactions_count"}, inplace=True
    )

    fraud_per_reported_date = fraud_per_reported_date.reset_index()

    return fraud_per_reported_date


def get_fraud_before_transaction_date(
    merchant_transactions: pd.DataFrame, fraud_per_reported_date: pd.DataFrame
) -> pd.DataFrame:

    fraud_before_transaction_date = pd.merge(
        merchant_transactions[["transaction_date"]],
        fraud_per_reported_date,
        how="left",
        left_on="transaction_date",
        right_on="reported_date",
    )

    fraud_before_transaction_date.drop_duplicates(
        subset=["transaction_date"], inplace=True
    )

    fraud_before_transaction_date["merchant_fraud_transactions_count"] = (
        fraud_before_transaction_date[["merchant_fraud_transactions_count"]]
        .shift()
        .expanding()
        .sum()
    )

    fraud_before_transaction_date["merchant_fraud_transactions_count"].fillna(
        0, inplace=True
    )

    fraud_before_transaction_date.drop(columns=["reported_date"], inplace=True)

    return fraud_before_transaction_date


def add_merchant_risk_features(merchant_transactions: pd.DataFrame) -> pd.DataFrame:

    merchant_transactions = merchant_transactions.sort_values("transaction_time")

    fraud_per_reported_date = get_fraud_per_reported_date(merchant_transactions)

    fraud_before_transaction_date = get_fraud_before_transaction_date(
        merchant_transactions, fraud_per_reported_date
    )

    merchant_transactions = pd.merge(
        merchant_transactions,
        fraud_before_transaction_date,
        how="left",
        on="transaction_date",
    )

    merchant_transactions["merchant_transaction_count"] = (
        merchant_transactions["event_id"].expanding().count()
    )

    merchant_transactions["merchant_risk_score"] = (
        merchant_transactions.merchant_fraud_transactions_count
        / merchant_transactions.merchant_transaction_count
    )

    return merchant_transactions

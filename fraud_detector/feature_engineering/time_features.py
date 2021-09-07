import pandas as pd
from pytz import country_timezones


def add_label(
    df: pd.DataFrame, label_col_name: str, reported_fraud_time_col_name: str
) -> pd.DataFrame:
    df[label_col_name] = [
        1 if pd.notnull(x) else 0 for x in df[reported_fraud_time_col_name]
    ]

    return df


def add_localized_transaction_time(
    df: pd.DataFrame, transaction_time_col_name: str
) -> pd.DataFrame:
    df["localized_transaction_time"] = df.apply(
        lambda x: x[transaction_time_col_name].tz_convert(
            country_timezones(x["alpha_two"])[0]
        ),
        axis=1,
    )

    df["localized_transaction_time"] = pd.to_datetime(
        df["localized_transaction_time"], utc=True
    )

    return df


def add_localized_transaction_time_features(
    df: pd.DataFrame,
    localized_time_features_list: list,
    localized_transaction_time_col_name: str,
) -> pd.DataFrame:
    for feature in localized_time_features_list:
        df[f"local_{feature}"] = getattr(
            df[localized_transaction_time_col_name].dt, feature
        )
    return df


def add_reported_time_features(
    df: pd.DataFrame,
    reported_time_features_list: list,
    reported_fraud_time_col_name: str,
) -> pd.DataFrame:
    for feature in reported_time_features_list:
        df[f"reported_{feature}"] = getattr(
            df[reported_fraud_time_col_name].dt, feature
        )
    return df


def add_transaction_time_features(
    df: pd.DataFrame,
    transaction_time_features_list: list,
    transaction_time_col_name: str,
) -> pd.DataFrame:
    for feature in transaction_time_features_list:
        df[f"transaction_{feature}"] = getattr(
            df[transaction_time_col_name].dt, feature
        )
    return df

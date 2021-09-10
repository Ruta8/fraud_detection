import pandas as pd
from pytz import country_timezones


def add_localized_transaction_time(
    df: pd.DataFrame, transaction_time_col_name: str
) -> pd.DataFrame:
    """Adds a column with local transaction time.

    Args:
        df (pd.DataFrame): main dataframe.
        transaction_time_col_name (str): column that has non-local
        transaction time.

    Returns:
        pd.DataFrame: the same dataframe with aditional local transaction
        time column.
    """

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


def add_time_property_features(
    df: pd.DataFrame,
    time_properties_list: list,
    time_feature_col_name: str,
    time_property_feature_prefix: str,
) -> pd.DataFrame:
    """Takes datetime column, extracts required time properties, and sets then as new
    time-based features.

    Args:
        df (pd.DataFrame): the main data frame.
        time_properties_list (list): a list of time properties to extract.
        time_feature_col_name (str): datetime column from which to extract 
        time_property_feature_prefix (str): prefix for new columns.

    Returns:
        pd.DataFrame: main dataframe with aditional time-based features.
    """

    for time_property in time_properties_list:
        df[f"{time_property_feature_prefix}_{time_property}"] = getattr(
            df[time_feature_col_name].dt, time_property
        )

    return df

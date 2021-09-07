"""Functions to clean data"""
import pandas as pd
import numpy as np


def non_alphanumerics_to_nans(
    df: pd.DataFrame, alphanumeric_cols: list
) -> pd.DataFrame:
    for column in alphanumeric_cols:
        df.replace(
            {
                column: {
                    "^[0-9]*$": np.nan,
                    "[^a-zA-Z0-9]": np.nan,
                    "^[a-zA-Z]+$": np.nan,
                }
            },
            inplace=True,
            regex=True,
        )
        return df


def validate_merchant_zip_codes(
    df: pd.DataFrame, zip_codes_df: np.ndarray, merchant_zip_col_name: str
) -> pd.DataFrame:

    df[merchant_zip_col_name] = df[merchant_zip_col_name].str.upper()
    df["merchant_zip_areas"] = (
        df[merchant_zip_col_name].str[:2].replace("(\d)", "", regex=True)
    )
    valid_merchant_zip_areas = zip_codes_df.iloc[:, 0].unique()
    non_valid_merchant_zips = list(
        df[~df["merchant_zip_areas"].isin(valid_merchant_zip_areas)][
            merchant_zip_col_name
        ].unique()
    )
    df[merchant_zip_col_name] = df[merchant_zip_col_name].replace(
        non_valid_merchant_zips, np.nan
    )
    df.drop(columns=["merchant_zip_areas"], inplace=True)

    return df


def drop_rows_from_columns_list(df: pd.DataFrame, cols_list: list) -> pd.DataFrame:
    for column in cols_list:
        df = df.dropna(subset=[column])

    return df

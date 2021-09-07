import pandas as pd


def fill_categorical_column_na(df: pd.DataFrame) -> pd.DataFrame:
    categorical_features = df.select_dtypes(include="object").columns
    df[categorical_features] = df[categorical_features].fillna("missing_value")

    return df


def reduce_feature_cardinality(
    df: pd.DataFrame, feature: str, nb_of_categories: int
) -> pd.DataFrame:

    current_nb_of_categories = df[feature].nunique()
    assert (
        current_nb_of_categories > nb_of_categories
    ), "New number of features cannot be larger than current number of features."
    top_categories = (
        df[feature]
        .value_counts()
        .sort_values(ascending=False)
        .nlargest(nb_of_categories)
        .index
    ).to_list()

    df[feature] = df[feature].apply(
        lambda category: category if category in top_categories else "OTHER"
    )

    return df

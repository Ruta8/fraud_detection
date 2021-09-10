import pandas as pd


def reduce_feature_cardinality(
    df: pd.DataFrame, feature: str, nb_of_categories: int
) -> pd.DataFrame:
    """Reduces cardinality for a feature to the number of categories given.

    Args:
        df (pd.DataFrame): oridinal dataframe.
        feature (str): feature name with high cardinality.
        nb_of_categories (int): number of prefered categories in the high
        cardinality feature.

    Returns:
        pd.DataFrame: the same dataframe with the reduced cardinality feature.
    """

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

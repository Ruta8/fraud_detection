""" Main function triggers the complete workflow."""

import pandas as pd
import numpy as np
import pickle
from betacal import BetaCalibration
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from fraud_detector.data_cleaning.clean_data import (
    non_alphanumerics_to_nans,
    validate_merchant_zip_codes,
    drop_rows_from_columns_list,
)


from fraud_detector.feature_engineering.time_features import (
    add_localized_transaction_time,
    add_time_property_features,
)


from fraud_detector.feature_engineering.account_behavior_features import (
    add_account_spending_behaviour_features,
    add_transaction_time_diff_features,
    add_transaction_stats_features,
)


from fraud_detector.feature_engineering.merchant_features import (
    add_merchant_risk_features,
)


from fraud_detector.feature_engineering.other import reduce_feature_cardinality


from fraud_detector.modeling.modeling import (
    evaluate_model,
    choose_threshold,
    return_test_with_predictions,
    perform_randomized_search,
)


def main(randomized_search=False):
    df = pd.read_csv(
        "./data/raw/transactions.csv",
        parse_dates=["transaction_time"],
        dtype={
            "mcc": str,
            "merchant_country": str,
            "pos_entry_mode": str,
            "merchant_zip": str,
        },
    )
    labels_df = pd.read_csv("./data/raw/labels.csv", parse_dates=["reported_time"])
    mcc_descriptions_df = pd.read_csv(
        "./data/external/mcc_descriptions.csv", dtype={"mcc": str}
    )
    country_codes_df = pd.read_csv(
        "./data/external/country_codes.csv", dtype={"merchant_country": str}
    )
    uk_zip_codes_df = pd.read_csv("./data/external/uk_postcodes.csv")

    df = pd.merge(df, mcc_descriptions_df, how="left", on="mcc")
    df = pd.merge(df, country_codes_df, how="left", on="merchant_country")
    df = pd.merge(df, labels_df, how="left", on="event_id")

    df = non_alphanumerics_to_nans(df, ["merchant_df", "account_number"])
    df = validate_merchant_zip_codes(df, uk_zip_codes_df, "merchant_zip")
    df = drop_rows_from_columns_list(df, ["account_number", "merchant_id"])
    df = df.drop_duplicates(subset=["transaction_time", "event_id"], keep="first")
    df = df[
        df["pos_entry_mode"].isin(["0", "1", "2", "5", "7", "80", "81", "90", "91"])
    ]
    df = df[df.transaction_amount > 0]

    df["label"] = [1 if pd.notnull(x) else 0 for x in df["reported_time"]]

    df = add_localized_transaction_time(df, "transaction_time")
    df = add_time_property_features(
        df, ["day", "hour", "month"], "localized_transaction_time", "local"
    )
    df = add_time_property_features(df, ["date"], "reported_time", "reported")
    df = add_time_property_features(df, ["date"], "transaction_time", "transaction")

    categorical_features = df.select_dtypes(include="object").columns
    df[categorical_features] = df[categorical_features].fillna("missing_value")

    df = df.groupby("account_number").apply(
        lambda account_transactions: add_account_spending_behaviour_features(
            account_transactions,
            [1, 7, 14, 20, 30],
            "transaction_time",
            "transaction_amount",
            "event_id",
        )
    )
    df = df.sort_values("transaction_time").reset_index(drop=True)

    df = df.groupby("account_number").apply(
        lambda account_transactions: add_transaction_stats_features(
            account_transactions, "transaction_amount", "transaction_time",
        )
    )
    df = df.sort_values("transaction_time").reset_index(drop=True)

    df = df.groupby("account_number").apply(
        lambda account_transactions: add_transaction_time_diff_features(
            account_transactions, "transaction_time", "event_id",
        )
    )
    df = df.sort_values("transaction_time").reset_index(drop=True)

    df = df.groupby("merchant_id").apply(lambda x: add_merchant_risk_features(x))

    df["sample_weight"] = np.ceil(df["transaction_amount"]).astype(int)
    df.loc[df["label"] == 0, "sample_weight"] = 1

    df["transaction_during_night"] = np.where(
        (df["localized_transaction_time"]).dt.hour <= 6, 1, 0
    )
    df["flag_fraud_experienced_merchant"] = np.where(
        df["merchant_risk_score"] > 0, 1, 0
    )

    df[df.select_dtypes(include=[np.number]).columns] = df.select_dtypes(
        include=[np.number]
    ).fillna(0)

    df = reduce_feature_cardinality(df, "mcc", 15)
    df = reduce_feature_cardinality(df, "merchant_country", 15)
    df = reduce_feature_cardinality(df, "merchant_zip", 15)

    preprocessed_df = df.sort_values("transaction_time").reset_index(drop=True)
    preprocessed_df.to_csv("./data/processed/preprocessed.csv")

    preprocessed_df = preprocessed_df.drop(
        columns=[
            "transaction_time",
            "event_id",
            "account_number",
            "merchant_id",
            "alpha_two",
            "reported_time",
            "localized_transaction_time",
            "reported_date",
            "transaction_date",
        ]
    )

    encoded_df = pd.get_dummies(preprocessed_df, drop_first=True)
    encoded_df.to_csv("./data/processed/encoded_df.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        encoded_df.drop(columns=["label"]),
        encoded_df.label,
        test_size=1 / 13,
        stratify=encoded_df.label,
        random_state=42,
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=1 / 12, stratify=y_train, random_state=42
    )

    if randomized_search == True:
        print(f"Performing randomized search for hyperparameters.")
        randomized_search_model = perform_randomized_search(
            RandomForestClassifier(),
            X_train.drop(columns=["sample_weight"]),
            y_train,
            X_train.sample_weight,
            {
                "model__max_depth": list(np.random.randint(3, 100, 1000)),
                "model__n_estimators": list(np.random.randint(100, 800, 1000)),
                "model__min_samples_split": [2, 3, 4, 5, 6,],
                "model__max_features": ["auto", "sqrt", "log2"],
                "model__min_samples_leaf": [1, 2, 4, 6],
                "model__criterion": ["gini", "entropy"],
                "model__random_state": [42],
                "model__class_weight": ["balanced"],
            },
            "neg_log_loss",
            3,
        )
        with open(
            "./fraud_detector/models/randomized_search_model.pkl", "wb",
        ) as handle:
            pickle.dump(randomized_search_model, handle)

        rf_model = RandomForestClassifier()
        rf_model.set_params(**randomized_search_model.best_params_)

        print(f"Randomized search parameters: {randomized_search_model.best_params_}")

    else:
        rf_model = RandomForestClassifier(
            n_estimators=700,
            min_samples_split=4,
            min_samples_leaf=1,
            max_features="auto",
            max_depth=89,
            criterion="entropy",
            random_state=42,
            class_weight="balanced",
        )

    rf_model.fit(
        X_train.drop(columns=["sample_weight"]), y_train, X_train.sample_weight
    )
    with open("./fraud_detector/models/rf_model.pkl", "wb",) as handle:
        pickle.dump(rf_model, handle)

    uncalibrated_val_predictions = rf_model.predict_proba(
        X_val.drop(columns=["sample_weight"])
    )[:, 1]

    calibrated_rf_model = BetaCalibration(parameters="abm")
    calibrated_rf_model.fit(uncalibrated_val_predictions, y_val)
    with open("./fraud_detector/models/calibrated_rf_model.pkl", "wb",) as handle:
        pickle.dump(calibrated_rf_model, handle)

    calibrated_rf_test_predictions = calibrated_rf_model.predict(
        rf_model.predict_proba(X_test.drop(columns=["sample_weight"]))[:, 1]
    )

    thresholds_list = list(np.arange(0.00, 1.05, 0.005))
    evaluation_metrics_df = evaluate_model(
        calibrated_rf_test_predictions,
        X_test.drop(columns=["sample_weight"]),
        y_test,
        thresholds_list,
    )
    evaluation_metrics_df.to_csv("./results/evaluation_metrics_df.csv")

    best_threshold = choose_threshold(y_test, calibrated_rf_test_predictions)
    test_with_predictions = return_test_with_predictions(
        X_test.drop(columns=["sample_weight"]),
        y_test,
        calibrated_rf_test_predictions,
        best_threshold,
    )
    test_with_predictions.to_csv("./results/test_with_predictions.csv")

    return


if __name__ == "__main__":
    main()

"""Register, prepare, and train the SuperKart sales model."""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET = "Product_Store_Sales_Total"
NUMERIC = ["Product_Weight", "Product_Allocated_Area", "Product_MRP", "Store_Establishment_Year"]
CATEGORICAL = ["Product_Sugar_Content", "Product_Type", "Store_Id", "Store_Size",
               "Store_Location_City_Type", "Store_Type"]
FEATURES = NUMERIC + CATEGORICAL
SOURCE = Path("data/SuperKart.csv")
SPLITS = Path("splits")
MODEL = Path("models/superkart.joblib")


def load_data():
    frame = pd.read_csv(SOURCE)
    expected = set(FEATURES + ["Product_Id", TARGET])
    if set(frame.columns) != expected:
        raise ValueError(f"Missing columns: {expected - set(frame)}; extra columns: {set(frame) - expected}")
    for column in NUMERIC + [TARGET]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if frame.empty or frame.isna().any().any():
        raise ValueError("Expected a nonempty dataset with complete rows.")
    return frame


def validate():
    frame = load_data()
    print(f"Rows: {len(frame):,} | Columns: {len(frame.columns)}")
    print(f"Stores: {frame.Store_Id.nunique()} | Products: {frame.Product_Id.nunique()}")
    print(f"Duplicate rows: {frame.duplicated().sum()}")
    print(f"Repeated product-store pairs: {frame.duplicated(['Product_Id', 'Store_Id']).sum()}")
    print(frame.isna().sum().to_string())
    print(frame[TARGET].describe().to_string())


def prepare():
    frame = load_data().drop_duplicates()
    frame[CATEGORICAL] = frame[CATEGORICAL].apply(lambda column: column.str.strip())
    frame["Product_Sugar_Content"] = frame["Product_Sugar_Content"].replace({"reg": "Regular"})
    if frame.Product_Id.duplicated().any():
        raise ValueError("Repeated products require a grouped holdout and cross-validation.")
    train, test = train_test_split(frame[FEATURES + [TARGET]], test_size=0.2,
                                   random_state=42, stratify=frame.Store_Id)
    SPLITS.mkdir(exist_ok=True)
    train.to_csv(SPLITS / "train.csv", index=False)
    test.to_csv(SPLITS / "test.csv", index=False)
    print(f"Training rows: {len(train):,} | Test rows: {len(test):,}")


def metrics(actual, predictions):
    return {"MAE": mean_absolute_error(actual, predictions),
            "RMSE": root_mean_squared_error(actual, predictions),
            "R2": r2_score(actual, predictions)}


def train():
    training = pd.read_csv(SPLITS / "train.csv")
    test = pd.read_csv(SPLITS / "test.csv")
    x_train, y_train = training[FEATURES], training[TARGET]
    x_test, y_test = test[FEATURES], test[TARGET]
    preprocessing = ColumnTransformer([
        ("numbers", "passthrough", NUMERIC),
        ("categories", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    estimator = Pipeline([
        ("features", preprocessing),
        ("model", RandomForestRegressor(random_state=42)),
    ])
    search = GridSearchCV(
        estimator, {"model__n_estimators": [40, 80], "model__min_samples_leaf": [2, 5]},
        scoring="neg_mean_absolute_error", cv=KFold(3, shuffle=True, random_state=42),
        n_jobs=2, error_score="raise",
    )
    search.fit(x_train, y_train)
    baseline = DummyRegressor(strategy="median").fit(x_train, y_train)
    results = {"best_parameters": search.best_params_,
               "baseline": metrics(y_test, baseline.predict(x_test)),
               "random_forest": metrics(y_test, search.predict(x_test))}
    MODEL.parent.mkdir(exist_ok=True)
    trials = pd.DataFrame(search.cv_results_)[["param_model__n_estimators",
        "param_model__min_samples_leaf", "mean_test_score", "std_test_score", "rank_test_score"]]
    trials["cv_mae"] = -trials.pop("mean_test_score")
    trials = trials.rename(columns={"std_test_score": "cv_mae_std"})
    trials.to_csv(MODEL.parent / "experiments.csv", index=False)
    print(trials.to_string(index=False))
    print(json.dumps(results, indent=2))
    (MODEL.parent / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    categories = {column: sorted(training[column].unique().tolist()) for column in CATEGORICAL}
    example = {column: float(training[column].median()) for column in NUMERIC}
    joblib.dump({"pipeline": search.best_estimator_, "categories": categories,
                 "defaults": example}, MODEL, compress=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "prepare", "train"])
    stage = parser.parse_args().stage
    {"validate": validate, "prepare": prepare, "train": train}[stage]()

import pickle
import pandas as pd
from prefect import flow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.test_suite import TestSuite
from evidently.test_preset import DataDriftTestPreset
from evidently.ui.workspace import Workspace
from preprocessing import read_data, feature_engineering


@flow
def main_monitoring():
    target = "Count"
    cat_features = ["Weather", "Day", "Round", "Dir", "Mode", "season"]
    num_features = None
    with open("models/poi_reg.bin", "rb") as f_in:
        reg = pickle.load(f_in)
    with open("models/dv.bin", "rb") as f_in:
        dv = pickle.load(f_in)
    train_df = []
    val_df = []
    train_pair = [(2022, "spring"), (2022, "autumn")]
    val_pair = [(2023, "spring"), (2023, "autumn")]
    for year, season in train_pair:
        df = read_data(year, season)
        train_df.append(df)
    train = pd.concat(train_df, axis=0, ignore_index=True)
    train[cat_features] = train[cat_features].astype(str).apply(lambda x: x.str.lower())
    train.fillna("N", inplace=True)
    for year, season in val_pair:
        df = read_data(year, season)
        val_df.append(df)
    val = pd.concat(val_df, axis=0, ignore_index=True)
    val[cat_features] = val[cat_features].astype(str).apply(lambda x: x.str.lower())
    val.fillna("N", inplace=True)
    X_train, y_train, dv = feature_engineering(train, cat_features, target, dv)
    X_val, y_val, dv = feature_engineering(val, cat_features, target, dv)
    train["prediction"] = reg.predict(X_train)
    val["prediction"] = reg.predict(X_val)

    column_mapping = ColumnMapping(
        prediction="prediction",
        numerical_features=num_features,
        categorical_features=cat_features,
        target=target
    )

    test_suite = TestSuite(tests = [DataDriftTestPreset()])
    test_suite.run(reference_data=train, current_data=val, column_mapping=column_mapping)
    report = Report(metrics = [DataDriftPreset()])
    report.run(reference_data=train, current_data=val, column_mapping=column_mapping)

    ws = Workspace("workspace")
    project = ws.create_project("Santander bicycle rentals in London")
    project.save()
    ws.add_test_suite(project.id, test_suite)
    ws.add_report(project.id, report)

if __name__ == '__main__':
    main_monitoring()
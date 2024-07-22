import pickle
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
from prefect import task, flow
from preprocessing import read_data, feature_engineering

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Santander bicycle rentals in London")


@task(log_prints=True)
def run_train(X_train, y_train, X_val, y_val):

    mlflow.sklearn.autolog(log_datasets=False)
    with mlflow.start_run():
        params = {"alpha": 0.1}
        reg = PoissonRegressor(**params)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_val)
        mlflow.log_metric("mae", mean_absolute_error(y_val, y_pred))
    return reg


@flow
def main_train(train_pair, val_pair, feature_list, target):
    train_df = []
    val_df = []
    for year, season in train_pair:
        df = read_data(year, season)
        train_df.append(df)
    train = pd.concat(train_df, axis=0, ignore_index=True)
    for year, season in val_pair:
        df = read_data(year, season)
        val_df.append(df)
    val = pd.concat(val_df, axis=0, ignore_index=True)
    X_train, y_train, dv = feature_engineering(train, feature_list, target)
    X_val, y_val, dv = feature_engineering(val, feature_list, target, dv)
    reg = run_train(X_train, y_train, X_val, y_val)
    with open('models/dv.bin', 'wb') as f_out:
        pickle.dump(dv, f_out)
    with open('models/poi_reg.bin', 'wb') as f_out:
        pickle.dump(reg, f_out)


train_pair = [(2022, "spring"), (2022, "autumn")]
val_pair = [(2023, "spring"), (2023, "autumn")]
feature_list = ["Weather", "Day", "Round", "Dir", "Mode", "season"]
target = "Count"


if __name__ == '__main__':
    main_train(train_pair, val_pair, feature_list, target)
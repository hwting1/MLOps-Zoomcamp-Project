import numpy as np
import pandas as pd
import mlflow
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
from prefect import task, flow
from preprocessing import read_data, feature_engineering

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("poisson-regression-hyperopt")


@task(log_prints=True)
def run_optimization(X_train, y_train, X_val, y_val, num_trials):

    def objective(params):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)

            reg = PoissonRegressor(**params)
            reg.fit(X_train, y_train)
            y_pred = reg.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            mlflow.log_metric("mae", mae)

        return {'loss': mae, 'status': STATUS_OK}

    search_space = {
        "alpha": scope.float(hp.loguniform("alpha", 0, np.log(100)))
    }

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


@flow
def main_hpo(train_pair, val_pair, feature_list, target, num_trials=20):
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
    run_optimization(X_train, y_train, X_val, y_val, num_trials)


train_pair = [(2022, "spring"), (2022, "autumn")]
val_pair = [(2023, "spring"), (2023, "autumn")]
feature_list = ["Weather", "Day", "Round", "Dir", "Mode", "season"]
target = "Count"


if __name__ == '__main__':
    main_hpo(train_pair, val_pair, feature_list, target)
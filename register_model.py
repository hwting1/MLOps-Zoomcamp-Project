import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error
import mlflow
from mlflow import MlflowClient
from mlflow.entities import ViewType
from prefect import task, flow
from preprocessing import read_data, feature_engineering

HPO_EXPERIMENT_NAME = "poisson-regression-hyperopt"
EXPERIMENT_NAME = "poisson-regression-best-models"
REG_PARAMS = ["alpha"]

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()


@task
def train_and_log_model(X_train, y_train, X_val, y_val, params):

    with mlflow.start_run():
        new_params = {}
        for param in REG_PARAMS:
            new_params[param] = float(params[param])

        reg= PoissonRegressor(**new_params)
        reg.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, reg.predict(X_val))
        mlflow.log_metric("val_mae", val_mae)


@flow
def main_register_model(train_pair, val_pair, feature_list, target, top_n):

    client = MlflowClient()
    experiment = client.get_experiment_by_name(HPO_EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.mae ASC"]
    )
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


    for run in runs:
        train_and_log_model(X_train, y_train, X_val, y_val, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(experiment_ids = experiment.experiment_id,
                                  run_view_type=ViewType.ACTIVE_ONLY,
                                  max_results=1,
                                  order_by=["metrics.val_mae ASC"]
                                  )[0]

    # Register the best model
    model_uri = f"runs:/{best_run.info.run_id}/model"
    mlflow.register_model(model_uri=model_uri, name="BestPoissonRegressionModel")


train_pair = [(2022, "spring"), (2022, "autumn")]
val_pair = [(2023, "spring"), (2023, "autumn")]
feature_list = ["Weather", "Day", "Round", "Dir", "Mode", "season"]
target = "Count"
top_n = 5

if __name__ == '__main__':
    main_register_model(train_pair, val_pair, feature_list, target, top_n)
# Final Project for 'MLOps-Zoomcamp'

## Santander Bicycle Rentals in London Prediction
This project aims to predict the number of bicycle rentals in London using the Santander bicycle rental dataset. Accurate predictions of bicycle rental demand can greatly enhance the efficiency of bike-sharing systems, ensuring that bicycles are available where and when they are needed most. By leveraging machine learning techniques, we can uncover patterns and trends in the rental data, helping city planners and the bike-sharing service to optimize their operations.

The dataset contains historical rental data, including information on the number of bicycles rented, weather conditions, and other relevant features. This project involves a series of steps, from data preprocessing and feature engineering to model training and deployment, all orchestrated using MLOps practices. The goal is to create a robust and scalable pipeline that can handle data processing, model training, and prediction in a seamless manner.

## Solution

The solution involves the following steps for code execution:

### Set up

1. Create a Python virtual environment and execute `pip install -r requirements.txt` to install the required dependencies.

### Step 1: Start MLflow and Prefect Services

2. Start the MLflow UI with a backend store using SQLite:
    ```sh
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```

3. In a separate terminal, start the Prefect server:
    ```sh
    prefect server start
    ```
These commands will initiate the necessary services to monitor and manage machine learning workflows effectively.

### Step 2: Hyperparameter Optimization and Model Registration

4. Run the following command to perform hyperparameter optimization:
    ```sh
    python hpo.py
    ```

5. Run the following command to register the best model into the MLflow service:
    ```sh
    python register_model.py
    ```

6. Run the following command to train and export the model locally:
    ```sh
    python train.py
    ```

### Step 3: Batch Deployment

7. Run the following command to perform batch deployment, ensuring that the corresponding files are present in the `data` folder:
    ```sh
    python batch_predict.py {year} {season}
    ```

This command will generate batch predictions based on the specified year and season, using the prepared data files in the `data` folder. The prediction results will be exported to the `output` folder.

### Step 4: Monitoring

8. Start the Evidently UI:
    ```sh
    evidently ui
    ```

9. Run the following command to perform monitoring:
    ```sh
    python `monitoring.py`
    ```

You can check the results at [http://0.0.0.0:8000](http://0.0.0.0:8000).

These steps will set up the monitoring service, allowing you to visualize and track the performance of your machine learning models in real-time.


## Services
- http://127.0.0.1:4200 - Prefect server;
- http://127.0.0.1:5000 - MLflow server;
- http://0.0.0.0:8000 - Evidently server.



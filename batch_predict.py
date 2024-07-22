import sys, pickle
from prefect import flow
from preprocessing import read_data, feature_engineering


@flow
def main_predict():
    year = int(sys.argv[1])
    season = sys.argv[2]
    output_path = f"output/{year}-{season}-Cycleways-predict.csv"
    feature_list = ["Weather", "Day", "Round", "Dir", "Mode", "season"]
    predict_df = read_data(year, season)
    with open('models/dv.bin', 'rb') as f_in:
        dv = pickle.load(f_in)
    X_test, _ = feature_engineering(predict_df, feature_list, dv=dv)

    with open('models/poi_reg.bin', 'rb') as f_in:
        reg = pickle.load(f_in)
    y_test = reg.predict(X_test)
    df_predict = predict_df[["Date", "Time"]].copy()
    df_predict["predict_Count"] = y_test
    df_predict.to_csv(output_path, index=False)


if __name__ == '__main__':
    main_predict()



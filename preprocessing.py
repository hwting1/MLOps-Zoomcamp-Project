import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from prefect import task


@task
def read_data(year, season):
    df = pd.read_csv(f"data/{year}-{season}-Cycleways.csv")
    df["season"] = season
    return df


@task
def feature_engineering(df, feature_list, target=None, dv=None):
    df[feature_list] = df[feature_list].astype(str).apply(lambda x: x.str.lower())
    df.fillna("N", inplace=True)
    df_dict = df.to_dict(orient="records")
    if dv:
        X = dv.transform(df_dict)
    else:
        dv = DictVectorizer()
        X = dv.fit_transform(df_dict)

    if target:
        y = df[target].values
        return X, y, dv
    else:
        return X, dv



import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingRegressor
import joblib

import warnings
warnings.filterwarnings('ignore')

# import sys
# sys.path.insert(1, 'scripts/')
import scripts.preprocess as prep

def mape(y_obs, y_pred):

    return round(np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100, 2)


def prep_data(df: pd.DataFrame):

    n_lag = 3
    values = df.values
    reframed_df = prep.series_to_supervised(values, n_lag, 1)

    values = reframed_df.values
    size = (int)(values.shape[0] * 0.2)

    train = values[:-size, :]
    test = values[-size:, :]

    X_train, y_train = train[:, :-1], train[:, -1]
    X_test, y_test = test[:, :-1], test[:, -1]

    return X_train, y_train, X_test, y_test


def build_model(X_train, y_train):

    gbr_model = GradientBoostingRegressor(random_state=42)
    gbr_model.fit(X_train, y_train)

    return gbr_model


if __name__ == "__main__":

    # 1. load data & create dataset for mmodel
    train_file = "/home/arkii/_Codes/estimate_project/FC1302/Deployment/data/fc_1302_train.csv"
    df, mm_scaler = prep.load_data_csv(train_file)
    X_train, y_train, X_test, y_test = prep_data(df)

    # 2. model build
    gbr_model = build_model(X_train, y_train)

    # 3. predict
    gbr_pred  = gbr_model.predict(X_test)
    gbr_mape = mape(y_test, gbr_pred)

    print(f'Gradient Boosting Regressor: {gbr_mape}%')

    # 4. dump model (using joblib)
    with open("model.joblib", "wb") as f:
        joblib.dump(gbr_model, f)   

    with open("scaler.joblib", "wb") as f:
        joblib.dump(mm_scaler, f)




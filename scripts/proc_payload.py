import preprocess
import numpy as np
import pandas as pd
import joblib

import os, sys
from datetime import datetime, timedelta


import warnings
warnings.filterwarnings('ignore')

def create_data(argv):

    if len(argv) != 3:
        print("Error")
        exit()

    filename = argv[1]
    target_date_str = argv[2]

    # 1. load data
    payload_file = os.path.join("data/", filename)
    df, mm_scaler = preprocess.load_data_csv(payload_file)

    # 2. get raw data (target date 3 days ago ~ target date)
    n_lag = 3

    target_date = datetime.fromisoformat(target_date_str)
    leg_date = target_date - timedelta(days = n_lag)

    leg_date_str = leg_date.strftime("%Y-%m-%d")

    df = df.loc[leg_date_str:target_date_str]

    # 3. change raw data to payload data set
    values = df.values
    reframed_df = preprocess.series_to_supervised(values, n_lag, 1)    

    payload_values = reframed_df.values

    X_payload = payload_values[:, :-1]
    y_obs = payload_values[:, -1]

    return X_payload, y_obs, mm_scaler


def save_obj(obj, file_name="obj.pkl"):
    try:
        joblib.dump(obj, file_name)
    except Exception as e:
        print(e)
        exit()
    

if __name__ == "__main__":
    
    # 1. create payload data
    X_payload, y_obs, mm_scaler = create_data(sys.argv)

    # 2. save Payload data 
    payload_path = "data/X_payload_{}.csv".format(sys.argv[2])
    np.savetxt(payload_path, X_payload, delimiter=",")

    y_obs_path = "data/y_obs_{}.csv".format(sys.argv[2])
    inv_y_obs = mm_scaler.inverse_transform(y_obs.reshape(-1, 1))
    np.savetxt(y_obs_path, inv_y_obs, delimiter=",")

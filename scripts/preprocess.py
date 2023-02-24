import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	
    n_vars = 1 if type(data) is list else data.shape[1]

    df = pd.DataFrame(data)
    cols, names = list(), list()

	# input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]

    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


def load_obj(file_path):
    try:
        with open(file_path, "rb") as f:
            obj = joblib.load(f)
    except Exception as e:
        print(e)
        exit()
    
    return obj
    


def load_data_csv(file_path):
    df = pd.read_csv(file_path, parse_dates=["BKG_DATE"]).set_index("BKG_DATE")

    mm_scaler = MinMaxScaler()
    mm_scaler = load_obj("scripts/mm-scaler.pkl")

    df["ITEM_CD_COUNT"] = mm_scaler.fit_transform(df[["ITEM_CD_COUNT"]])

    return df, mm_scaler


def mape(y_obs, y_pred):
    return round(np.mean(np.abs((y_obs - y_pred) / y_obs)) * 100, 2)

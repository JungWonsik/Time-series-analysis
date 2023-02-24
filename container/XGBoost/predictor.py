import os
import json
import joblib
import flask
import boto3
import time
import pyarrow
from pyarrow import feather
import modin.pandas as pd

import logging

logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler("xgb_deploy.log")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logging.info("============= Started ===============")

#Define the path
prefix = "/opt/ml/"
model_path = os.path.join(prefix, "model")
logging.info("Model Path: " + str(model_path))

# # Load the model components
xgb_model = joblib.load(os.path.join(model_path, "xgb_model.pkl"))
logging.info("XGBoost: " + str(xgb_model))

# The flask app for serving predictions
app = flask.Flask(__name__)
@app.route('/ping', methods=['GET'])
def ping():
    # Check if the classifier was loaded correctly
    try:
        #regressor
        status = 200
        logging.info("Status : 200")
    except:
        status = 400
    return flask.Response(response= json.dumps(' '), status=status, mimetype='application/json' )

@app.route('/invocations', methods=['POST'])
def transformation():
    # Get input JSON data and convert it to a DF
    input_json = flask.request.get_json()
    input = input_json['input']['exp1']
    predictions = float(xgb_model.predict([[input]]))

    # Transform predictions to JSON
    result = {
        'output': predictions
        }

    resultjson = json.dumps(result)
    return flask.Response(response=resultjson, status=200, mimetype='application/json')
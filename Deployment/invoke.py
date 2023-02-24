import pandas as pd
import boto3
import json

runtime_client = boto3.client('sagemaker-runtime')
content_type = "application/json"

file_path = "data/X_payload.csv"

df = pd.read_csv(file_path, header=None)
input_value = df.values.tolist()

request_body = {"Input": input_value}

data = json.loads(json.dumps(request_body))
payload = json.dumps(data)
endpoint_name = "sklearn-local-ep2022-11-15-09-12-26"

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType=content_type,
    Body=payload)
result = json.loads(response['Body'].read().decode())['Output']
print(result)
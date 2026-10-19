import requests
from flask import jsonify

host = 'churn-serving-env.eba-tiiv8kaq.eu-central-1.elasticbeanstalk.com'
url = f'http://{host}/predict'

customer = {
  "gender": "female",
  "seniorcitizen": 0,
  "partner": "yes",
  "dependents": "no",
  "phoneservice": "no",
  "multiplelines": "no_phone_service",
  "internetservice": "dsl",
  "onlinesecurity": "no",
  "onlinebackup": "yes",
  "deviceprotection": "no",
  "techsupport": "no",
  "streamingtv": "no",
  "streamingmovies": "no",
  "contract": "month-to-month",
  "paperlessbilling": "yes",
  "paymentmethod": "electronic_check",
  "tenure": 12,
  "monthlycharges": 29.85,
  "totalcharges": 12*29.85
}

response = requests.post(url, json=customer)
print(response.json())

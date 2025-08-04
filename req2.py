import requests
import json

url = "http://localhost:8000/batch_predict"
with open("json/batch.json", "r") as f:
    data = json.load(f)

response = requests.post(url, json=data)

print(response.status_code)
print(response.json())
print(response)


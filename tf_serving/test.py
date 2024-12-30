import requests

endpoint = 'http://localhost:8001/predict'
url = "https://bit.ly/al-gaib"
data = {'url':  url}

response = requests.post(endpoint, json=data).json()
print(response)
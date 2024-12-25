import requests

docker_url = 'http://localhost:8080/api/clothing_model'
az_local_url = 'http://localhost:7071/api/clothing_model'
az_url = "https://harkonnen.niceground-4d5ca68b.eastus.azurecontainerapps.io/api/clothing_model"
img_url = {'url': 'http://bit.ly/al-gaib'}

# Send the request with JSON content type
result = requests.post(az_url, json=img_url)

print(result.status_code)
print(result.json())
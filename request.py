import requests

response = requests.post(
    'http://127.0.0.1:5000/predict', 
    json = {
        "input": [
            [0.0019, 0.0894], 
            [0.0019, 0.0894], 
            [0.0019, 0.0894], 
            [0.0019, 0.0894], 
            [0.0019, 0.0894], 
            [0.0019, 0.0902]
        ]
    }
)
print(response.json())

import requests

url = "http://localhost:9696/predict"

client = {
    "reports" : 0,
    "share" : 0.245,
    "expenditure" : 3.438,
    "owner" : "yes"
}

response = requests.post(url, json=client).json()

card_prob = round(response["card_probability"], 3)

print(f"Probability of client getting a credit card: {card_prob}")

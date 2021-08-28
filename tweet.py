import json
import requests
from config import token

keyword1 = '(America OR USA OR United States of America)'
keyword2 =  ' -is:retweet -is:reply lang:en'
max_results = 10

search_url = "https://api.twitter.com/2/tweets/search/recent"

# US used as an example query as it is a required field
query_params = {'query': keyword1+keyword2,
                'max_results': max_results,
               }


def bearer_oauth(r):
    r.headers["Authorization"] = f"Bearer {token}"
    return r

def connect_to_endpoint(url, params):
    response = requests.get(url, auth=bearer_oauth, params=params)
    print(response.status_code)
    if response.status_code != 200:
        raise Exception(response.status_code, response.text)
    return response.json()

def api_call():
    tweets = []
    json_response = connect_to_endpoint(search_url, query_params)
    for response in json_response['data']:
        tweets.append( {'id': response['id'], 'tweet': response['text'], 'sentiment': ''}) 
    return tweets



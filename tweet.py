import json
import requests
from config import token
from sqlalchemy import Table, Column, String, MetaData, Date, create_engine, insert, Float, SmallInteger
from sqlalchemy.ext.declarative import declarative_base

rds_connection_string = "postgres:postgres@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')
conn = engine.connect()
Base = declarative_base()
Base.metadata.create_all(conn)
meta = MetaData()
tweet_data = Table(
   'tweet_data', meta, 
   Column('id', String, primary_key = True), 
   Column('tweet', String), 
   Column('sentiments', SmallInteger),
   Column('predicted_sentiments', Float),
   Column('time_data_inserted', Date) 
)
meta.create_all(engine)

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
        conn = engine.connect()

        with conn:
            conn.execute(insert(tweet_data),[{"id":response['id'],
                "tweet":response['text'],
                "sentiments":9,
                "predicted_sentiments":9,
                "time_data_inserted":'1/1/01'}]) 
    return tweets


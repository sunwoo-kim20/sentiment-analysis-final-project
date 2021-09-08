import json
import requests
from flask_sqlalchemy import sqlalchemy
from config import token, user, password, host, port, database
from sqlalchemy import Table, Column, String, MetaData, Date, create_engine, insert, Float, SmallInteger, update
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from sqlalchemy import func
import pandas as pd
from v_functions import batch_strings, batch_ints, lema_tweet



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
    rds_connection_string = "postgres:postgres@npl-instance-1.cnrgtjkaikng.us-east-2.rds.amazonaws.com:5432/sentiment_db"
    engine = create_engine(f'postgresql://{rds_connection_string}')
    conn = engine.connect()

    metadata = MetaData(engine)
    tweet_data = Table('tweet_data', metadata, autoload=True, autoload_with=engine) 

    json_response = connect_to_endpoint(search_url, query_params)
    conn = engine.connect()
    session = Session(bind=engine)
    
    modded = 1
    holder = 0
    if session.query(tweet_data).count() == 0:
        batch = batch_strings['0']
    else:
        batch_df = pd.read_sql_query('select batch from tweet_data ORDER BY batch DESC LIMIT 1', con=engine)
        holder = batch_df['batch'].max()
        batch = batch_strings[holder]
        modded = batch_ints[batch]    
        d = tweet_data.delete().where(tweet_data.c.holder == 0)
        d.execute()
        Conn = engine.connect()
    

    
    
    
    if modded%10 == 0:
        neg = len(pd.read_sql_query('select sentiments from stats_data WHERE stats_data.sentiments = 0 LIMIT 1250', con=engine))
        pos = len(pd.read_sql_query('select sentiments from stats_data WHERE stats_data.sentiments = 1 LIMIT 1250', con=engine))
        if(neg == 1250) and (pos == 1250):
            import updateModel

    samples = []
    rez = json_response['data']
    for i in rez:
        samples.append(i['text'])
    df_holder = pd.DataFrame()
    df_holder['tweet'] = samples
    df_clean_holder = lema_tweet(df_holder,'tweet')
    length = len(rez)

    for i in range(length):
        conn = engine.connect()
        session = Session(bind=engine)
        unique_tweet = session.query(tweet_data).filter(tweet_data.c.id == rez[i]['id']).count()
        session.close()


        if(unique_tweet == 0):
            tweets.append( {'id': rez[i]['id'], 'tweet': rez[i]['text'],})
        with conn:
            conn.execute(insert(tweet_data),[{
                "id":rez[i]['id'],
                "batch":batch,
                "tweet":rez[i]['text'],
                "joined_lemm":df_clean_holder['joined_lemm'][i],
                "holder":1,
                }])
    return tweets
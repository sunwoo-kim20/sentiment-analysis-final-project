import csv
import os
import pandas as pd
from sqlalchemy import create_engine

read_csv = pd.read_csv('Resources/data/imdb_reviews_data.csv')
df = pd.DataFrame(read_csv)

df['sentiments'] = df.sentiment.apply(lambda x: 1 if x in ['positive'] else 0)
df.drop('Unnamed: 0',axis=1,inplace=True)

rds_connection_string = "postgres:password@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

df.to_sql(name='sentiment_data', con=engine, if_exists='append', index=False)
engine.table_names()
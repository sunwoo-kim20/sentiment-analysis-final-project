import csv
import os
import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData, Date
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, BigInteger
from sqlalchemy import create_engine

rds_connection_string = "postgres:Stardrive!1@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

conn = engine.connect()
Base = declarative_base()
Base.metadata.create_all(conn)
meta = MetaData()

read_csv = pd.read_csv('imdb_reviews_data.csv')

df['sentiments'] = df.sentiment.apply(lambda x: 1 if x in ['positive'] else 0)
df.drop(['Unnamed: 0','sentiment'],axis=1,inplace=True)

sentiment_data = Table(
   'sentiment_data', meta, 
   Column('id', Integer, primary_key = True), 
   Column('text', String, default='NaN'), 
   Column('sentiments', Integer, default='NaN'), 
)

tweet = Table(
   'tweet', meta, 
   Column('id', BigInteger, primary_key = True), 
   Column('tweet', String, default='NaN'), 
   Column('sentiments', Integer, default='NaN'),
   Column('predicted_sentiments', Integer, default='NaN'),
   Column('time_data_insterted', Date, default='NaN') 
)

meta.create_all(engine)
df.to_sql(name='sentiment_data', con=engine, if_exists='append', index=False)

import csv
import os
import pandas as pd
from sqlalchemy import Table, Column, Integer, String, MetaData, Date
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, BigInteger
from sqlalchemy import create_engine

rds_connection_string = "postgres:postgres@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

df = pd.read_sql_query('select * from tweet', con=engine)

def return_data():
    sentiments = df['sentiments']
    predicted_sentiments = df['predicted_sentiments']
    date = df['time_data_insterted']
    return sentiments,predicted_sentiments,date
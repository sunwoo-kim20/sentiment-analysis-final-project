from sqlalchemy import Table, Column, Integer, String, MetaData, Date
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, SmallInteger, BigInteger
from sqlalchemy import create_engine
import pandas as pd
from v_functions import lema
rds_connection_string = "postgres:postgres@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')
conn = engine.connect()
Base = declarative_base()
Base.metadata.create_all(conn)
meta = MetaData()

if not engine.has_table(engine, "stats_data"):  # If table don't exist, Create.
    meta = MetaData()
    # Create a table with the appropriate Columns"
    stats = Table("stats_data", meta,
          Column('Id', String, primary_key=True, nullable=False), 
          Column('Date', Date), 
          Column('Precision_rd', Float),
          Column('Recall_rd', Float),
          Column('Precision_twt', Float),
          Column('Recall_twt', Float),
          Column('Precision_com', Float),
          Column('Recall_rd_com', Float)
                 )
    # Implement the creation
    meta.create_all(engine)

if not engine.has_table(engine, "sentiment_data"):  # If table don't exist, Create.
    meta = MetaData()
    read_csv = pd.read_csv('Resources/data/imdb_reviews_data.csv')
    df = pd.DataFrame(read_csv)
    df['sentiments'] = df.sentiment.apply(lambda x: 1 if x in ['positive'] else 0)
    df.drop(['Unnamed: 0','sentiment'],axis=1,inplace=True)
    df = lema(df,'text')
    sentiment_data = Table('sentiment_data', meta, 
                    Column('text', String), 
                    Column('sentiments', Integer),
                    Column('joined_lemm', String), 
                    )
    meta.create_all(engine)
    df.to_sql(name='sentiment_data', con=engine, if_exists='append', index=False)
    

    
if not engine.has_table(engine, "tweet_data"):  # If table don't exist, Create.
    tweet_data = Table(
        'tweet_data', meta, 
        Column('id', String, primary_key = True),
        Column("batch", String), 
        Column('tweet', String), 
        Column('sentiments', SmallInteger),
        Column('predicted_sentiments_rd', Float),
        Column('predicted_sentiments_twt', Float),
        Column('predicted_sentiments_com', Float),
        Column('time_data_inserted', Date),
        Column('joined_lemm', String), 
    )
    meta.create_all(engine)    
from sqlalchemy import Table, Column, Integer, String, MetaData, Date
from sqlalchemy import MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, BigInteger
from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sqlalchemy import insert

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    steve = []
    steve.append(f'(True Negatives): {cm[0][0]}')
    steve.append(f'(False Negatives): {cm[1][0]}')
    steve.append(f'(True Positives): {cm[1][1]}')
    steve.append(f'(False Positives): {cm[0][1]}')
    steve.append(f'TotalTransactions: {np.sum(cm[1])}')
    steve.append(f'precision: {cm[1][1]/(cm[1][1] + cm[0][1])}')
    steve.append(f'recall: {cm[1][1]/(cm[1][1] + cm[1][0])}')

    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(steve)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('static/images/figure_1.png', dpi=300)

    
    
rds_connection_string = "postgres:postgres@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')
conn = engine.connect()
Base = declarative_base()
Base.metadata.create_all(conn)
meta = MetaData()
stats = Table("stats_data", meta,
          Column('Id', String, primary_key=True, nullable=False), 
          Column('Date', Date),  
          Column('Precision', Float),
          Column('Recall', Float)
                 )
df = pd.read_sql_query('select * from tweet_data WHERE tweet_data.sentiments != 9', con=engine)
df_stats = pd.read_sql_query('select * from stats_data', con=engine)
x_predict_test = df['predicted_sentiments']
y_actual = df['sentiments']

plot_cm(y_actual,x_predict_test)
cm = confusion_matrix(y_actual,x_predict_test > .5)
precision = cm[1][1]/(cm[1][1] + cm[0][1])
recall = cm[1][1]/(cm[1][1] + cm[1][0])

conn = engine.connect()
ids = str(len(df_stats))
with conn:
        result = conn.execute(insert(stats),[{"Id":ids, 
                                              "Date":df['time_data_inserted'].max(),
                                              "Precision":precision,
                                               "Recall":recall}])

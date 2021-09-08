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
import os
# from v_functions import plot_cm, plot_roc, plot_delta_auc, plot_prc, cmFile, rocFile, deltaaucFile, prcFile, colors
import v_functions
from config import token, user, password, host, port, database

if os.path.isfile(v_functions.cmFile):
    os.remove(v_functions.cmFile)
if os.path.isfile(v_functions.rocFile):
    os.remove(v_functions.rocFile)
if os.path.isfile(v_functions.deltaaucFile):
    os.remove(v_functions.deltaaucFile)
if os.path.isfile(v_functions.prcFile):
    os.remove(v_functions.prcFile)

colors = v_functions.colors 
  
rds_connection_string = "postgres:postgres@npl-instance-1.cnrgtjkaikng.us-east-2.rds.amazonaws.com:5432/sentiment_db"
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

v_functions.plot_cm(y_actual,x_predict_test)
cm = confusion_matrix(y_actual,x_predict_test > .5)
precision = cm[1][1]/(cm[1][1] + cm[0][1])
recall = cm[1][1]/(cm[1][1] + cm[1][0])

plt.figure(figsize=(15,7))
v_functions.plot_roc("ROC", y_actual, x_predict_test, color=colors[0])

plt.figure(figsize=(20,10))
v_functions.plot_delta_auc("Delta AUC",y_actual,x_predict_test,color=colors[1])


plt.figure(figsize=(10,10))
v_functions.plot_prc("PRC", y_actual, x_predict_test, color=colors[2])



conn = engine.connect()
ids = str(len(df_stats))
with conn:
        result = conn.execute(insert(stats),[{"Id":ids, 
                                              "Date":df['time_data_inserted'].max(),
                                              "Precision":precision,
                                               "Recall":recall}])

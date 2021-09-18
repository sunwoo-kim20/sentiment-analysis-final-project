from nltk.corpus.reader import senseval
from nltk.corpus.reader.knbc import test
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import nltk
import string
from v_functions import METRICS, make_model, early_stopping
from nltk.corpus import stopwords
import keras.metrics
from sqlalchemy import Table, Column, String, MetaData, Date, create_engine, insert, Float, SmallInteger
from sqlalchemy.orm import Session
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import Normalizer
import numpy as np
from config import rds_connection_string
import v_functions
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import sklearn
from datetime import datetime
import psycopg2
psycopg2.extensions.register_adapter(np.int64, psycopg2._psycopg.AsIs)

METRICS = METRICS

engine = create_engine(f'postgresql://{rds_connection_string}')
conn = engine.connect()

metadata = MetaData(engine)
tweet_data = Table('tweet_data', metadata, autoload=True, autoload_with=engine)

wn = nltk.WordNetLemmatizer()
string.punctuation
stop = stopwords.words('english')

neg = len(pd.read_sql_query('select sentiments from tweet_sentiment WHERE tweet_sentiment.sentiments = 0 LIMIT 2500', con=engine))
pos = len(pd.read_sql_query('select sentiments from tweet_sentiment WHERE tweet_sentiment.sentiments = 1 LIMIT 2500', con=engine))

min_count = 0
if (neg == 2500) and (pos == 2500):
    min = 2500
elif neg > pos:
    min_count = pos
elif pos > neg:
    min_count = neg
    
querry_string_base = 'select joined_lemm, sentiments from tweet_sentiment WHERE tweet_sentiment.sentiments =' 
neg_string = querry_string_base + ' 0 ORDER BY RANDOM() LIMIT ' + f'{min_count}'
pos_string = querry_string_base + ' 1 ORDER BY RANDOM() LIMIT ' + f'{min_count}'
model_df = pd.concat([pd.read_sql_query(neg_string, con=engine), pd.read_sql_query(pos_string, con=engine)])

x_twt = model_df['joined_lemm']
y_twt = model_df['sentiments']
X_train, X_test, y_train, y_test = train_test_split(x_twt, y_twt, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2)
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
vectorized_train = vectorizer.transform(X_train).toarray()
vectorized_test = vectorizer.transform(X_test).toarray()
vectorized_val = vectorizer.transform(X_val).toarray()
norm = Normalizer().fit(vectorized_train)
norm_vectorized_train = norm.transform(vectorized_train)
norm_vectorized_test = norm.transform(vectorized_test)
norm_vectorized_val = norm.transform(vectorized_val)
pickle.dump(vectorizer, open("Resources/vectorizers/tweet_vectorizer.pickle", "wb"))

train_labels_twt = np.array(y_train)
bool_train_labels_twt = train_labels_twt != 0
val_labels_twt = np.array(y_val)
test_labels_twt = np.array(y_test)

train_features_twt = norm_vectorized_train
val_features_twt = norm_vectorized_val
test_features_twt = norm_vectorized_test

train_features_twt = np.clip(train_features_twt, -5, 5)
val_features_twt = np.clip(val_features_twt, -5, 5)
test_features_twt = np.clip(test_features_twt, -5, 5)

EPOCHS = 50
BATCH_SIZE = 2048


model_twt = make_model(train_features_twt)
baseline_history = model_twt.fit(
    train_features_twt,
    train_labels_twt,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_features_twt, test_labels_twt),
    callbacks=[early_stopping])
model_twt.save("Resources/models/deep_sentiment_twitter_model_trained.h5", save_format='tf')
x_predict_twt = model_twt.predict(val_features_twt)
y_actual_twt = val_labels_twt

neg_composite_string = querry_string_base + ' 0 ORDER BY RANDOM() LIMIT 1250'
pos_composite_string = querry_string_base + ' 1 ORDER BY RANDOM() LIMIT 1250'
tweet_composite_df = pd.concat([pd.read_sql_query(neg_composite_string, con=engine), pd.read_sql_query(pos_composite_string, con=engine)])

review_composite_df = pd.read_sql_query('select joined_lemm, sentiments from sentiment_data ORDER BY RANDOM() LIMIT 2500', con=engine)

final_composite_df = pd.concat([tweet_composite_df, review_composite_df])

x_com = final_composite_df['joined_lemm']
y_com = final_composite_df['sentiments']
X_train, X_test, y_train, y_test = train_test_split(x_com, y_com, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2)
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
vectorized_train = vectorizer.transform(X_train).toarray()
vectorized_test = vectorizer.transform(X_test).toarray()
vectorized_val = vectorizer.transform(X_val).toarray()
norm = Normalizer().fit(vectorized_train)
norm_vectorized_train = norm.transform(vectorized_train)
norm_vectorized_test = norm.transform(vectorized_test)
norm_vectorized_val = norm.transform(vectorized_val)
pickle.dump(vectorizer, open("Resources/vectorizers/composite_vectorizer.pickle", "wb"))

train_labels_com = np.array(y_train)
bool_train_labels_com = train_labels_com != 0
val_labels_com = np.array(y_val)
test_labels_com = np.array(y_test)

train_features_com = norm_vectorized_train
val_features_com = norm_vectorized_val
test_features_com = norm_vectorized_test

train_features_com = np.clip(train_features_com, -5, 5)
val_features_com = np.clip(val_features_com, -5, 5)
test_features_com = np.clip(test_features_com, -5, 5)


EPOCHS = 50
BATCH_SIZE = 2048


model_com = make_model(train_features_com)
baseline_history = model_com.fit(
    train_features_com,
    train_labels_com,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_features_com, test_labels_com),
    callbacks=[early_stopping])
model_com.save("Resources/models/deep_sentiment_composite_model_trained.h5", save_format='tf')

x_predict_com = model_com.predict(val_features_com)
y_actual_com = val_labels_com


df = pd.read_sql_query('select joined_lemm, sentiments from sentiment_data ORDER BY RANDOM() LIMIT 1000', con=engine)
neg_composite_string2 = querry_string_base + ' 0 ORDER BY RANDOM() LIMIT 1000'
pos_composite_string2 = querry_string_base + ' 1 ORDER BY RANDOM() LIMIT 1000'
comp_df = pd.concat([pd.read_sql_query(neg_composite_string2, con=engine), pd.read_sql_query(pos_composite_string2, con=engine),df])

filter_df = pd.read_sql_query('select joined_lemm from filter_data ORDER BY RANDOM() LIMIT 3000', con=engine)
stevenz = []
for i in range(len(filter_df)):
    stevenz.append(1)
filter_df['sentiments'] = stevenz
comp_df['sentiments'] = df.sentiments.apply(lambda x: 0 if x in ['0'] else 0)

adjudicator_df = pd.concat([comp_df,filter_df])

x_adj = adjudicator_df['joined_lemm']
y_adj = adjudicator_df['sentiments']
X_train, X_test, y_train, y_test = train_test_split(x_adj, y_adj, test_size=0.2)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.2)
vectorizer = TfidfVectorizer()
vectorizer.fit(X_train)
vectorized_train = vectorizer.transform(X_train).toarray()
vectorized_test = vectorizer.transform(X_test).toarray()
vectorized_val = vectorizer.transform(X_val).toarray()
norm = Normalizer().fit(vectorized_train)
norm_vectorized_train = norm.transform(vectorized_train)
norm_vectorized_test = norm.transform(vectorized_test)
norm_vectorized_val = norm.transform(vectorized_val)
pickle.dump(vectorizer, open("Resources/vectorizers/adjudication_vectorizer.pickle", "wb"))

train_labels_adj = np.array(y_train)
bool_train_labels_adj = train_labels_adj != 0
val_labels_adj = np.array(y_val)
test_labels_adj = np.array(y_test)

train_features_adj = norm_vectorized_train
val_features_adj = norm_vectorized_val
test_features_adj = norm_vectorized_test

train_features_adj = np.clip(train_features_adj, -5, 5)
val_features_adj = np.clip(val_features_adj, -5, 5)
test_features_adj = np.clip(test_features_adj, -5, 5)


EPOCHS = 50
BATCH_SIZE = 2048


model_adj = make_model(train_features_adj)
baseline_history = model_adj.fit(
    train_features_adj,
    train_labels_adj,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(test_features_adj, test_labels_adj),
    callbacks=[early_stopping])
model_adj.save("Resources/models/deep_adjudicator_model_trained.h5", save_format='tf')

x_predict_adj = model_adj.predict(val_features_adj)
y_actual_adj = val_labels_adj


colors = v_functions.colors 
  
# Base = declarative_base()
# Base.metadata.create_all(conn)

                

if os.path.isfile(v_functions.cmFile_twt):
    os.remove(v_functions.cmFile_twt)
if os.path.isfile(v_functions.cmFile_com):
    os.remove(v_functions.cmFile_com)
if os.path.isfile(v_functions.cmFile_adj):
    os.remove(v_functions.cmFile_adj)
if os.path.isfile(v_functions.rocFile):
    os.remove(v_functions.rocFile)
if os.path.isfile(v_functions.deltaaucFile):
    os.remove(v_functions.deltaaucFile)
if os.path.isfile(v_functions.prcFile):
    os.remove(v_functions.prcFile)

v_functions.plot_cm("Twitter Confusion Matrix", v_functions.cmFile_twt,y_actual_twt,x_predict_twt)
v_functions.plot_cm("Composite Confusion Matrix",v_functions.cmFile_com,y_actual_com,x_predict_com)
v_functions.plot_cm("Adjudicator Confusion Matrix", v_functions.cmFile_adj,y_actual_adj,x_predict_adj)





def pre_rec(y_actual,x_predict):
    cm = confusion_matrix(y_actual,x_predict > .5)
    upper_p = cm[1][1]
    lower_p = cm[1][1] + cm[0][1]
    precision = 42
    if lower_p != 0:
        precision = upper_p / lower_p
    upper_r = cm[1][1]
    lower_r = cm[1][1] + cm[1][0]
    recall = 42
    if lower_r != 0:
        recall = upper_r / lower_r
    return (precision, recall)
# plt.figure(figsize=(20,10))
# v_functions.plot_delta_auc("Delta AUC",y_actual_twt,x_predict_twt,color=colors[1])


# plt.figure(figsize=(10,10))
# v_functions.plot_prc("PRC", y_actual_twt, x_predict_twt, color=colors[2])
date = datetime.now()
batch_df = pd.read_sql_query('select batch_max, version from stats_data', con=engine)
steve = len(batch_df)
batch_min = 1
batch_max = pd.read_sql_query('select batch from tweet_sentiment', con=engine)['batch'].max()
if steve != 0:

    batch_min += batch_df['batch_max'].max()
    real_version = batch_df['version'].max()
    string1 = 'select predicted_sentiments_adj from filter_data WHERE batch > ' + str(batch_min) + ' and batch <= ' + str(batch_max)
    df1 = pd.read_sql_query(string1, con=engine)





    string2 = 'select predicted_sentiments_adj,predicted_sentiments_twt,predicted_sentiments_com, sentiments from tweet_sentiment WHERE batch  > ' + str(batch_min) + ' and batch <= ' + str(batch_max)

    df2 = pd.read_sql_query(string2, con=engine)

    df_holder = pd.DataFrame()
    df_holder['predicted_sentiments_adj'] = df2['predicted_sentiments_adj']
    holder1 = []
    holder2 = []
    for i in range(len(df1)):
        holder1.append(1)
    df1['sentiments_adj'] = holder1
    for i in range(len(df_holder)):
        holder2.append(0)
    df_holder['sentiments_adj'] = holder2

    joined_adj_df = pd.concat([df1,df_holder])
    x_predict_adj_real = joined_adj_df['predicted_sentiments_adj']
    y_actual_adj_real = joined_adj_df['sentiments_adj']

    x_predict_twt_real = df2['predicted_sentiments_twt']
    x_predict_com_real = df2['predicted_sentiments_com']
    y_actual_real = df2['sentiments']
    precision_adj_real, recall_adj_real = pre_rec(y_actual_adj_real,x_predict_adj_real)

    precision_twt_real, recall_twt_real = pre_rec(y_actual_real,x_predict_twt_real)

    precision_com_real, recall_com_real = pre_rec(y_actual_real,x_predict_com_real)


    fpr_twt_real, tpr_twt_real, _ = sklearn.metrics.roc_curve(y_actual_real, x_predict_twt_real)
    fpr_com_real, tpr_com_real, _ = sklearn.metrics.roc_curve(y_actual_real, x_predict_com_real)
    fpr_adj_real, tpr_adj_real, _ = sklearn.metrics.roc_curve(y_actual_adj_real, x_predict_adj_real)
    auc_twt_real = sklearn.metrics.auc(fpr_twt_real,tpr_twt_real)
    auc_com_real = sklearn.metrics.auc(fpr_com_real,tpr_com_real)
    auc_adj_real = sklearn.metrics.auc(fpr_adj_real,tpr_adj_real)
    meta = MetaData()
    performance = Table('performance_stats', metadata, autoload=True, autoload_with=engine)

    conn = engine.connect()
    with conn:
        conn.execute(insert(performance),[{
            "version":real_version,
            'precision_adj':precision_adj_real,
            'recall_adj':recall_adj_real,
            'tpr_adj':tpr_adj_real.tolist(),
            'fpr_adj':fpr_adj_real.tolist(),
            'auc_adj':auc_adj_real,       
            'precision_twt':precision_twt_real,
            'recall_twt':recall_twt_real,
            'tpr_twt':tpr_twt_real.tolist(),
            'fpr_twt':fpr_twt_real.tolist(),
            'auc_twt':auc_twt_real,
            'precision_com':precision_com_real,
            'recall_com':recall_com_real,
            'tpr_com':tpr_com_real.tolist(),
            'fpr_com':fpr_com_real.tolist(),
            'auc_com':auc_com_real,
            "date":date,
            }])




precision_adj, recall_adj = pre_rec(y_actual_adj,x_predict_adj)
precision_twt, recall_twt = pre_rec(y_actual_twt,x_predict_twt)
precision_com, recall_com = pre_rec(y_actual_com,x_predict_com)


fpr_twt, tpr_twt, _ = sklearn.metrics.roc_curve(y_actual_twt, x_predict_twt)
fpr_com, tpr_com, _ = sklearn.metrics.roc_curve(y_actual_com, x_predict_com)
fpr_adj, tpr_adj, _ = sklearn.metrics.roc_curve(y_actual_adj, x_predict_adj)
auc_twt = sklearn.metrics.auc(fpr_twt,tpr_twt)
auc_com = sklearn.metrics.auc(fpr_com,tpr_com)
auc_adj = sklearn.metrics.auc(fpr_adj,tpr_adj)

plt.figure(figsize=(15,7))
v_functions.plot_roc("ROC", y_actual_twt, x_predict_twt, y_actual_com, x_predict_com, y_actual_adj, x_predict_adj)

df = pd.read_sql_query('select predicted_sentiments_rd, predicted_sentiments_twt, predicted_sentiments_com, sentiments from tweet_sentiment', con=engine)
version = steve + 1
meta = MetaData()
stats = Table('stats_data', metadata, autoload=True, autoload_with=engine)

      

conn = engine.connect()
with conn:
    conn.execute(insert(stats),[{
        "version":version,
        'batch_min':batch_min,
        'batch_max':batch_max,
        'precision_adj':precision_adj,
        'recall_adj':recall_adj,
        'tpr_adj':tpr_adj.tolist(),
        'fpr_adj':fpr_adj.tolist(),
        'auc_adj':auc_adj,       
        'precision_twt':precision_twt,
        'recall_twt':recall_twt,
        'tpr_twt':tpr_twt.tolist(),
        'fpr_twt':fpr_twt.tolist(),
        'auc_twt':auc_twt,
        'precision_com':precision_com,
        'recall_com':recall_com,
        'tpr_com':tpr_com.tolist(),
        'fpr_com':fpr_com.tolist(),
        'auc_com':auc_com,
        "date":date,
        }])





from nltk.corpus.reader import senseval
from nltk.corpus.reader.knbc import test
from tensorflow.keras.models import load_model
import pickle
import pandas as pd
import nltk
import string
from v_functions import lema_tweet, lema, METRICS, make_model, early_stopping
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
pickle.dump(vectorizer, open("tweet_vectorizer.pickle", "wb"))

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
model_twt.save("deep_sentiment_twitter_model_trained.h5", save_format='tf')
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
pickle.dump(vectorizer, open("composite_vectorizer.pickle", "wb"))

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
model_com.save("deep_sentiment_composite_model_trained.h5", save_format='tf')

x_predict_com = model_com.predict(val_features_com)
y_actual_com = val_labels_com


df = pd.read_sql_query('select joined_lemm, sentiments from sentiment_data ORDER BY RANDOM() LIMIT 1000', con=engine)
neg_composite_string2 = querry_string_base + ' 0 ORDER BY RANDOM() LIMIT 1000'
pos_composite_string2 = querry_string_base + ' 1 ORDER BY RANDOM() LIMIT 1000'
comp_df = pd.concat([pd.read_sql_query(neg_composite_string2, con=engine), pd.read_sql_query(pos_composite_string2, con=engine),df])

filter_df = pd.read_sql_query('select joined_lemm from filter_data ORDER BY RANDOM() LIMIT 3000', con=engine)
for i in range(len(filter_df)):
    filter_df['sentiments'] = 1
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
pickle.dump(vectorizer, open("adjudication_vectorizer.pickle", "wb"))

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
model_adj.save("deep_adjudicator_model_trained.h5", save_format='tf')

x_predict_adj = model_adj.predict(val_features_com)
y_actual_adj = val_labels_adj












if os.path.isfile(v_functions.cmFile):
    os.remove(v_functions.cmFile)
if os.path.isfile(v_functions.rocFile):
    os.remove(v_functions.rocFile)
if os.path.isfile(v_functions.deltaaucFile):
    os.remove(v_functions.deltaaucFile)
if os.path.isfile(v_functions.prcFile):
    os.remove(v_functions.prcFile)

colors = v_functions.colors 
  
# Base = declarative_base()
# Base.metadata.create_all(conn)
meta = MetaData()
stats = Table("stats_data", meta,
        Column('Id', String, primary_key=True), 
        Column('Date', Date), 
        Column('Precision_rd', Float),
        Column('Recall_rd', Float),
        Column('tpr_rd', Float),
        Column('fpr_rd', Float),
        Column('auc_rd', Float),
        Column('Precision_twt', Float),
        Column('Recall_twt', Float),
        Column('tpr_twt', Float),
        Column('fpr_twt', Float),
        Column('auc_twt', Float),
        Column('Precision_com', Float),
        Column('Recall_rd_com', Float),
        Column('tpr_com', Float),
        Column('fpr_com', Float),
        Column('auc_com', Float),
                )



v_functions.plot_cm(y_actual_twt,x_predict_twt)




cm_adj = confusion_matrix(y_actual_adj,x_predict_adj > .5)
precision_adj = cm_adj[1][1]/(cm_adj[1][1] + cm_adj[0][1])
recall_adj = cm_adj[1][1]/(cm_adj[1][1] + cm_adj[1][0])

cm_twt = confusion_matrix(y_actual_com,x_predict_com > .5)
precision_twt = cm_twt[1][1]/(cm_twt[1][1] + cm_twt[0][1])
recall_twt = cm_twt[1][1]/(cm_twt[1][1] + cm_twt[1][0])

cm_com = confusion_matrix(y_actual_com,x_predict_com > .5)
precision_com = cm_com[1][1]/(cm_com[1][1] + cm_com[0][1])
recall_com = cm_com[1][1]/(cm_com[1][1] + cm_com[1][0])

plt.figure(figsize=(15,7))
v_functions.plot_roc("ROC", y_actual_twt, x_predict_twt, color=colors[0])

plt.figure(figsize=(20,10))
v_functions.plot_delta_auc("Delta AUC",y_actual_twt,x_predict_twt,color=colors[1])


plt.figure(figsize=(10,10))
v_functions.plot_prc("PRC", y_actual_twt, x_predict_twt, color=colors[2])

df = pd.read_sql_query('select predicted_sentiments_rd, predicted_sentiments_twt, predicted_sentiments_com, sentiments from tweet_sentiment', con=engine)

x_predict_rd = df['predicted_sentiments_rd']

x_predict_twt = df['predicted_sentiments_twt']

x_predict_com = df['predicted_sentiments_com']

y_actual = df['sentiments']
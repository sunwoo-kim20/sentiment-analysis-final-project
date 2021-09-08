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
from config import token, user, password, host, port, database

rds_connection_string = "postgres:postgres@npl-instance-1.cnrgtjkaikng.us-east-2.rds.amazonaws.com:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')
conn = engine.connect()

metadata = MetaData(engine)
tweet_data = Table('tweet_data', metadata, autoload=True, autoload_with=engine)

wn = nltk.WordNetLemmatizer()
string.punctuation
stop = stopwords.words('english')

neg = len(pd.read_sql_query('select sentiments from tweet_data WHERE tweet_data.sentiments = 0 LIMIT 2500', con=engine))
pos = len(pd.read_sql_query('select sentiments from tweet_data WHERE tweet_data.sentiments = 1 LIMIT 2500', con=engine))

min_count = 0
if (neg == 2500) and (pos == 2500):
    min = 2500
elif neg > pos:
    min_count = pos
elif pos > neg:
    min_count = neg
    
querry_string_base = 'select * from tweet_data WHERE tweet_data.sentiments =' 
neg_string = querry_string_base + ' 0 ORDER BY RANDOM() LIMIT ' + f'{min_count}'
pos_string = querry_string_base + ' 1 ORDER BY RANDOM() LIMIT ' + f'{min_count}'
model_df = pd.concat([pd.read_sql_query(neg_string, con=engine), pd.read_sql_query(pos_string, con=engine)])

neg_composite_string = querry_string_base + ' 0 ORDER BY RANDOM() LIMIT 1250'
pos_composite_string = querry_string_base + ' 1 ORDER BY RANDOM() LIMIT 1250'
tweet_composite_df = pd.concat([pd.read_sql_query(neg_composite_string, con=engine), pd.read_sql_query(pos_composite_string, con=engine)])

final_tweet_composite_df = pd.DataFrame()
final_tweet_composite_df['text'] = tweet_composite_df['tweet']
final_tweet_composite_df['sentiments'] = tweet_composite_df['sentiments']

review_composite_df = pd.read_sql_query('select * from sentiment_data ORDER BY RANDOM() LIMIT 2500', con=engine)

final_composite_df = pd.concat([final_tweet_composite_df, review_composite_df])

clean_tweet_df = lema(model_df,'tweet')
clean_composite_df = lema(final_composite_df, 'text')



x = clean_tweet_df['joined_lemm']
y = clean_tweet_df['sentiments']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
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

train_labels = np.array(y_train)
bool_train_labels = train_labels != 0
val_labels = np.array(y_val)
test_labels = np.array(y_test)

train_features = norm_vectorized_train
val_features = norm_vectorized_val
test_features = norm_vectorized_test

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)

EPOCHS = 50
BATCH_SIZE = 2048


model = make_model(train_features)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels),
    callbacks=[early_stopping])
model.save("deep_sentiment_twitter_model_trained.h5", save_format='tf')


x_com = clean_composite_df['joined_lemm']
y_com = clean_composite_df['sentiments']
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

train_labels = np.array(y_train)
bool_train_labels = train_labels != 0
val_labels = np.array(y_val)
test_labels = np.array(y_test)

train_features = norm_vectorized_train
val_features = norm_vectorized_val
test_features = norm_vectorized_test

train_features = np.clip(train_features, -5, 5)
val_features = np.clip(val_features, -5, 5)
test_features = np.clip(test_features, -5, 5)


EPOCHS = 50
BATCH_SIZE = 2048


model = make_model(train_features)
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels),
    callbacks=[early_stopping])
model.save("deep_sentiment_composite_model_trained.h5", save_format='tf')
import pandas as pd
import sklearn
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import load_model
from tensorflow import keras
from sklearn.preprocessing import Normalizer
from sqlalchemy import create_engine
from sqlalchemy import MetaData, update, Table
from sqlalchemy.orm import Session
from config import rds_connection_string
import pickle
from v_functions import lema, make_model, METRICS, early_stopping
vectorizer = TfidfVectorizer()


engine = create_engine(f'postgresql://{rds_connection_string}')
metadata = MetaData(engine)
sentiment_data = Table('sentiment_data', metadata, autoload=True, autoload_with=engine)
conn = engine.connect()
session = Session(bind=engine)

df = pd.read_sql_query('select sentiments, joined_lemm from sentiment_data ORDER BY RANDOM() LIMIT 5000', con=engine)


x = df['joined_lemm']
y = df['sentiments']
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

pickle.dump(vectorizer, open("Resources/vectorizers/vectorizer.pickle", "wb"))

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

model.save("Resources/models/deep_sentiment_model_trained_zenith.h5", save_format='tf')
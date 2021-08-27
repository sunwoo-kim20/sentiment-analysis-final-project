import csv
import os
import pandas as pd
import sklearn
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow import keras
import string
import re
from sklearn.preprocessing import Normalizer
wn = nltk.WordNetLemmatizer()
string.punctuation
stop = stopwords.words('english')

def remove_punct(text):
    text_nopunct = "".join([char for char in text if char not in string.punctuation])
    return text_nopunct

def tokenize(text):
    tokens = re.split('\W+', text)
    return tokens

def remove_stopwords(tokenized_list):
    text = [word for word in tokenized_list if word not in stop]# To remove all stopwords
    return text

def lemmatizing(tokenized_text):
    text = [wn.lemmatize(word) for word in tokenized_text]
    return text

def lema(df,column):
    df['body_text_clean'] = df[column].apply(lambda x: remove_punct(x))
    df['body_text_tokenized'] = df['body_text_clean'].apply(lambda x: tokenize(x.lower()))
    df['body_text_nostop'] = df['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
    df['body_text_lemmatized'] = df['body_text_nostop'].apply(lambda x: lemmatizing(x))
    it_list = []
    for row in df['body_text_lemmatized']:
        it_list.append(" ".join(row))
    df['joined_lemm'] = it_list
    final_df = pd.DataFrame()
    final_df['sentiment'] = df['sentiment']
    final_df['joined_lemm'] = df['joined_lemm']
    return final_df

def plot_metrics(history):
    metrics = ['loss', 'prc', 'precision', 'recall']
    for n, metric in enumerate(metrics):
        name = metric.replace("_"," ").capitalize()
        plt.subplot(2,2,n+1)
        plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
        plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(name)
    if metric == 'loss':
        plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
        plt.ylim([0.8,1])
    else:
        plt.ylim([0,1])

    plt.legend()

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))
    
def plot_roc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)

    plt.plot(100*fp, 100*tp, label=name, linewidth=2, **kwargs)
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,20])
    plt.ylim([80,100.5])
    plt.grid(True)
    ax = plt.gca()
    ax.set_aspect('equal')
    


read_csv = pd.read_csv('Resources/data/imdb_reviews_data.csv')
df = pd.DataFrame(read_csv)

clean_df = lema(df,'text')
clean_df = clean_df.iloc[np.random.choice(np.arange(len(df)), 5000, False)]

clean_df['sentiments'] = df.sentiment.apply(lambda x: 1 if x in ['positive'] else 0)

x = clean_df['joined_lemm']
y = clean_df['sentiments']
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

label_encoder = LabelEncoder()
label_encoder.fit(y_train)

encoded_y_train = label_encoder.transform(y_train)
encoded_y_test = label_encoder.transform(y_test)

y_train_categorical = to_categorical(encoded_y_train)
y_test_categorical = to_categorical(encoded_y_test) 

export_vector = vectorizer.fit(X_train)

neg, pos = np.bincount(clean_df['sentiments'])
total = neg + pos
print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
    total, pos, 100 * pos / total))

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


print('Training labels shape:', train_labels.shape)
print('Validation labels shape:', val_labels.shape)
print('Test labels shape:', test_labels.shape)

print('Training features shape:', train_features.shape)
print('Validation features shape:', val_features.shape)
print('Test features shape:', test_features.shape)

METRICS = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.BinaryAccuracy(name='accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

EPOCHS = 50
BATCH_SIZE = 2048

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

def make_model(metrics=METRICS, output_bias=None):
    if output_bias is not None:
        output_bias = tf.keras.initializers.Constant(output_bias)
    model = keras.Sequential([
        keras.layers.Dense(
            16, activation='relu',
            input_shape=(train_features.shape[-1],)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dropout(0.5),

        keras.layers.Dense(1, activation='sigmoid',
                         bias_initializer=output_bias),
  ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=metrics)

    return model

model = make_model()
model.summary()

results = model.evaluate(train_features, train_labels, batch_size=BATCH_SIZE, verbose=0)
print("Loss: {:0.4f}".format(results[0]))

model = make_model()
baseline_history = model.fit(
    train_features,
    train_labels,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=(val_features, val_labels),
    callbacks=[early_stopping])

train_predictions_baseline = model.predict(train_features, batch_size=BATCH_SIZE)
test_predictions_baseline = model.predict(test_features, batch_size=BATCH_SIZE)

model.save("deep_sentiment_model_trained.h5")
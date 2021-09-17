import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords 
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow import keras
import string
import re
import pickle
import os

wn = nltk.WordNetLemmatizer()
string.punctuation
stop = stopwords.words('english')

colors = ["red","blue","yellow"]    

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
    final_df[column] = df[column]
    final_df['sentiments'] = df['sentiments']
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
cmFile_twt = "static/images/cm_twt.png"
cmFile_com = "static/images/cm_com.png"
cmFile_adj = "static/images/cm_adj.png"

def plot_cm(title,file_save,labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(title)
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig(file_save, dpi=300)

rocFile = "static/images/roc.png"    
def plot_roc(name, labels_twt, predictions_twt, labels_com, predictions_com, labels_adj, predictions_adj):
    fpr_twt, tpr_twt, _ = sklearn.metrics.roc_curve(labels_twt, predictions_twt)
    fpr_com, tpr_com, _ = sklearn.metrics.roc_curve(labels_com, predictions_com)
    fpr_adj, tpr_adj, _ = sklearn.metrics.roc_curve(labels_adj, predictions_adj)
    # auc1 = sklearn.metrics.auc(fpr1,tpr1)
    # auc1 = sklearn.metrics.auc(fpr1,tpr1)
    # auc1 = sklearn.metrics.auc(fpr1,tpr1)
    plt.plot(100*fpr_twt, 100*tpr_twt, label="model_twt", linewidth=2, color="red")
    plt.plot(100*fpr_com, 100*tpr_com, label="model_com", linewidth=2, color="blue")
    plt.plot(100*fpr_adj, 100*tpr_adj, label="model_adj", linewidth=2, color="yellow")
    plt.title(f'MODEL ROC')
    plt.xlabel('False positives [%]')
    plt.ylabel('True positives [%]')
    plt.xlim([-0.5,101])
    plt.ylim([20,100.5])
    plt.grid(True)
    ax = plt.gca()
    plt.savefig(rocFile, dpi=300)
#     ax.set_aspect('equal')

deltaaucFile = "static/images/deltaauc.png"    
def plot_delta_auc(name, labels, predictions, **kwargs):
    fp, tp, _ = sklearn.metrics.roc_curve(labels, predictions)
    boundry = len(tp) - 1
    x_axis = np.arange(0,1,(1/boundry))
    y_axis = []
    for i in range(boundry):
        y_axis.append(tp[(i + 1)]-tp[i])
    plt.plot(x_axis, y_axis, label=name, linewidth=2, **kwargs)
    plt.xlabel('Partition')
    plt.ylabel('DeltaAUC')
    plt.grid(True)
    ax = plt.gca()
    plt.savefig(deltaaucFile, dpi=300)

prcFile = "static/images/prc.png"  
def plot_prc(name, labels, predictions, **kwargs):
    precision, recall, _ = sklearn.metrics.precision_recall_curve(labels, predictions)

    plt.plot(precision, recall, label=name, linewidth=2, **kwargs)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    ax = plt.gca()
    plt.savefig(prcFile, dpi=300)
#     ax.set_aspect('equal')

def lema_tweet(df,column):
    df['body_text_clean'] = df[column].apply(lambda x: remove_punct(x))
    df['body_text_tokenized'] = df['body_text_clean'].apply(lambda x: tokenize(x.lower()))
    df['body_text_nostop'] = df['body_text_tokenized'].apply(lambda x: remove_stopwords(x))
    df['body_text_lemmatized'] = df['body_text_nostop'].apply(lambda x: lemmatizing(x))
    it_list = []
    for row in df['body_text_lemmatized']:
        it_list.append(" ".join(row))
    df['joined_lemm'] = it_list
    final_df = pd.DataFrame()

    final_df['joined_lemm'] = df['joined_lemm']
    return final_df

def lema_tweetz(df,column):
    df['body_text_clean'] = [df[column]].apply(lambda x: remove_punct(x))
    df['body_text_tokenized'] = [df['body_text_clean']].apply(lambda x: tokenize(x.lower()))
    df['body_text_nostop'] = [df['body_text_tokenized']].apply(lambda x: remove_stopwords(x))
    df['body_text_lemmatized'] = [df['body_text_nostop']].apply(lambda x: lemmatizing(x))
    it_list = []
    for row in df['body_text_lemmatized']:
        it_list.append(" ".join(row))
    df['joined_lemm'] = it_list
    final_df = pd.DataFrame()

    final_df['joined_lemm'] = df['joined_lemm']
    return final_df

def predictModel(df):
    with open('Resources/vectorizers/vectorizer.pickle', 'rb') as inp:
        vectorize = pickle.load(inp)
        model = load_model('Resources/models/deep_sentiment_model_trained_zenith.h5')
        predict_me = vectorize.transform(df['joined_lemm']).toarray()
        return float(model.predict(predict_me)[0][0])

def predictTwtModel(df):
    with open('Resources/vectorizers/tweet_vectorizer.pickle', 'rb') as inp:
        tweet_vectorizer = pickle.load(inp)
        model = load_model('Resources/models/deep_sentiment_twitter_model_trained.h5')
        predict_me = tweet_vectorizer.transform(df['joined_lemm']).toarray()
        return float(model.predict(predict_me)[0][0])

def predictComModel(df):
    with open('Resources/vectorizers/composite_vectorizer.pickle', 'rb') as inp:
        composite_vectorizer = pickle.load(inp) 
        model = load_model('Resources/models/deep_sentiment_composite_model_trained.h5')
        predict_me = composite_vectorizer.transform(df['joined_lemm']).toarray()
        return float(model.predict(predict_me)[0][0])
def predictAdjModel(df):
    with open('Resources/vectorizers/adjudication_vectorizer.pickle', 'rb') as inp:
        composite_vectorizer = pickle.load(inp) 
        model = load_model('Resources/models/deep_adjudicator_model_trained.h5')
        predict_me = composite_vectorizer.transform([df['joined_lemm']]).toarray()
        return float(model.predict(predict_me)[0][0])

# if os.path.isfile("batch_strings.pickle"):
#     with open('batch_strings.pickle', 'rb') as inp:
#         batch_strings = pickle.load(inp)
# else:
#     steve = np.arange(0,999999,1)
#     batch_strings = {}
#     for i in steve:
#         batch_strings[f'{i}'] = f'{i + 1}'
#     pickle.dump(batch_strings, open("batch_strings.pickle", "wb"))

# if os.path.isfile("batch_ints.pickle"):
#     with open('batch_ints.pickle', 'rb') as inp:
#         batch_ints = pickle.load(inp)
# else:
#     holder = np.arange(0,999999,1)
#     batch_ints = {}
#     for i in holder:
#         batch_ints[f'{i}'] = i
#     pickle.dump(batch_ints, open("batch_ints.pickle", "wb"))




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


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_prc', 
    verbose=1,
    patience=10,
    mode='max',
    restore_best_weights=True)

    
def make_model(train_features, metrics=METRICS, output_bias=None):
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
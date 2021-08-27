from numpy.lib.function_base import vectorize
from tensorflow.keras.models import load_model
import pickle

def predictModel(theTweet):
    list_for_vectorize = []
    list_for_vectorize.append(theTweet)
    with open('vectorizer.pkl', 'rb') as inp:
        vectorize = pickle.load(inp)
        model = load_model('deep_sentiment_model_trained.h5')
        predict_me = vectorize.transform(list_for_vectorize).toarray()
        return float(model.predict(predict_me)[0][0])


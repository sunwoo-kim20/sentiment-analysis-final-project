from numpy.lib.function_base import vectorize
from tensorflow.keras.models import load_model
import pickle

def predictModel(theTweet):
    with open('vectorizer.pkl', 'rb') as inp:
        vectorize = pickle.load(inp)
        model = load_model('deep_sentiment_model_trained.h5')
        predict_me = vectorize.transform(theTweet).toarray()
        print(model.predict(predict_me))
    return ""



test_string = ["this is great i love it"]
predictModel(test_string)
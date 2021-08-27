from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
import tweet
from flask import send_from_directory
import os
import json
from bson import json_util
from modelPredict import predictModel

# initialize flask
app = Flask(__name__)

# initalize mongo connection with pymongo
mongo = PyMongo(app, uri='mongodb://localhost:27017/tweets')

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/apicall")
def apicalled():
	tweet_data = tweet.api_call()[0]
	#mongo.db.collection.update({}, tweet_data, upsert=True)	
	#api_data = mongo.db.collection.find_one()
	tweet_data['sentiment'] = predictModel(tweet_data['tweet'])
	return json.loads(json_util.dumps(tweet_data))

@app.route("/data")
def tweet_data():
	tweet_show = mongo.db.collection.find_one()
	return json.loads(json_util.dumps(tweet_show))

if __name__ == "__main__":
	app.run(debug=True)	
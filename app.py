from flask import Flask, render_template, redirect
from flask_pymongo import PyMongo
import tweet
from flask import send_from_directory
import os
import json
from bson import json_util

# initialize flask
app = Flask(__name__)

# initalize mongo connection with pymongo
mongo = PyMongo(app, uri='mongodb://localhost:27017/tweets')

@app.route("/")
def home():
	tweet_data = tweet.api_call()
	mongo.db.collection.update({}, tweet_data, upsert=True)	
	api_data = mongo.db.collection.find_one()
	return render_template("index.html", api_data = api_data)



@app.route("/data")
def tweet_data():
    tweet = mongo.db.collection.find_one()
    return json.loads(json_util.dumps(tweet))

if __name__ == "__main__":
	app.run(debug=True)	
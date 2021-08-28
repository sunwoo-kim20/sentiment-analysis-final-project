# Dependencies
# ----------------------------------
from flask import Flask, render_template, redirect
import requests
import tweet
from flask import send_from_directory
import os
import json
from bson import json_util
from modelPredict import predictModel
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float


# initialize flask
app = Flask(__name__)
tweet_data= {}

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/apicall")
def apicalled():
	tweet_data = tweet.api_call()[0]
	tweet_data['sentiment'] = predictModel(tweet_data['tweet'])
	return json.loads(json_util.dumps(tweet_data))


@app.route("/positive_update")
def positive_update():
	tweet_data['time_data_inserted'] =
	tweet_data['predicted_sentiments'] =

@app.route("/negative_update")
def negative_update():
	tweet_data['time_data_inserted'] =
	tweet_data['predicted_sentiments'] =

if __name__ == "__main__":
	app.run(debug=True)

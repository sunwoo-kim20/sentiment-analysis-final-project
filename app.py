from flask import Flask, render_template, redirect
import requests
import tweet
from flask import send_from_directory
import os
import json
from bson import json_util
from modelPredict import predictModel

# initialize flask
app = Flask(__name__)


@app.route("/")
def home():
	return render_template("index.html")

@app.route("/apicall")
def apicalled():
	tweet_data = tweet.api_call()[0]
	tweet_data['sentiment'] = predictModel(tweet_data['tweet'])
	return json.loads(json_util.dumps(tweet_data))

@app.route("_sentiment_given")
def update_db():
	print(requests.args.get('selected',0)

if __name__ == "__main__":
	app.run(debug=True)	
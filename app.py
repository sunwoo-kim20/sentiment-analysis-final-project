# Dependencies
# ----------------------------------
from flask import Flask, render_template, redirect, jsonify
import requests
import tweet
from flask import send_from_directory
import os
import json
from modelPredict import predictModel
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, BigInteger, Date
from sqlalchemy.orm import Session
from datetime import datetime
from data import return_data


# initialize flask
app = Flask(__name__)


# Initialize global variable
tweet_data= {}

# Create database connection String
rds_connection_string = "postgres:postgres@localhost:5432/sentiment_db"

Base = declarative_base()
# Create class for updating data
class Tweet(Base):
	__tablename__ = 'tweet'
	id = Column(BigInteger, primary_key=True)
	tweet = Column(String())
	sentiments = Column(Integer)
	predicted_sentiments = Column(Integer)
	time_data_inserted = Column(Date)

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/voting")
def voting():
	return render_template("voting.html")

@app.route("/stats")
def statistics():
	return render_template("statistics.html")

@app.route("/apicall")
def apicalled():
	tweet_data = tweet.api_call()[0]
	tweet_data['predicted_sentiments'] = predictModel(tweet_data['tweet'])
	return jsonify(tweet_data)

@app.route("/positive_update")
def positive_update():
	tweet_data['time_data_inserted'] = datetime.now()
	tweet_data['sentiment'] = 1

	# Create connection to SQL database
	engine = create_engine(f'postgresql://{rds_connection_string}')
	conn = engine.connect()
	session = Session(bind = engine)

	# Create tweet instance
	tweet_upload = Tweet(
		id = tweet_data.id,
		tweet = tweet_data.tweet,
		sentiments = tweet_data.sentiment,
		predicted_sentiments = tweet_data.predicted_sentiments,
		time_data_inserted = tweet_data.time_data_inserted)

	# Add new instance to database
	session.add(tweet_upload)
	session.commit()

@app.route("/negative_update")
def negative_update():
	tweet_data['time_data_inserted'] = datetime.now()
	tweet_data['sentiment'] = 0

	# Create connection to SQL database
	engine = create_engine(f'postgresql://{rds_connection_string}')
	conn = engine.connect()
	session = Session(bind = engine)

	# Create tweet instance
	tweet_upload = Tweet(
		id = tweet_data.id,
		tweet = tweet_data.tweet,
		sentiments = tweet_data.sentiment,
		predicted_sentiments = tweet_data.predicted_sentiments,
		time_data_inserted = tweet_data.time_data_inserted)

	# Add new instance to database
	session.add(tweet_upload)
	session.commit()

@app.route("/data")
def datacalled():
	sentiments,predicted_sentiments,date = return_data()
	data_list = []
	holder = len(sentiments)
	for i in range(holder):
		data_list.append({
            'date': date[i],
            'sentiments':sentiments[i],
            'predicted_sentiments':predicted_sentiments[i]
		})
		
	return jsonify(data_list)

if __name__ == "__main__":
	app.run(debug=True)

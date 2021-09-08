# Dependencies
# ----------------------------------
from flask import Flask, render_template, redirect, jsonify, request
from flask import send_from_directory
import pandas as pd
from sqlalchemy import create_engine, insert
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData, update, Table
from sqlalchemy.orm import Session
from datetime import datetime
from v_functions import lema_tweetz, predictModel, lema_tweet, lema, predictTwtModel, predictComModel
# import database
import tweet
from multiprocessing import Value
import os
from config import token, user, password, host, port, database

counter = Value('i', 0)


# initialize flask
app = Flask(__name__)

# Create database connection String
rds_connection_string = "postgres:postgres@npl-instance-1.cnrgtjkaikng.us-east-2.rds.amazonaws.com:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

metadata = MetaData(engine)
tweet_data = Table('tweet_data', metadata, autoload=True, autoload_with=engine)
tweet_sentiment = Table('tweet_sentiment', metadata, autoload=True, autoload_with=engine)
filter_data = Table('filter_data', metadata, autoload=True, autoload_with=engine)
@app.route("/")
def home():
	return render_template("index.html")

@app.route("/voting")
def voting():
	return render_template("voting.html")

@app.route("/stats")
def statistics():
	return render_template("statistics.html")

@app.route("/load_tweet")
def load_tweet():
	conn = engine.connect()
	session = Session(bind=engine)
	available_tweets = session.query(tweet_data).filter(tweet_data.c.holder == 1).count()
	session.close()
	# print(available_tweets)
	if available_tweets == 0:
		tweet.api_call()
	df = pd.read_sql_query('select * from tweet_data WHERE tweet_data.holder = 1', con=conn)
	df = df.iloc[0]
	tweet_dict = {
		"id":df['id'],
		"tweet":df['tweet'],
		"joined_lemm":df['joined_lemm']
		}

	return jsonify(tweet_dict)


@app.route("/positive_update", methods = ['POST'])
def positive_update():
	# Try to grab values, will catch if someone clicks on face before a tweet loads
	if os.path.isfile("deep_sentiment_twitter_model_trained.h5"):
		df = pd.DataFrame()
		df['tweet'] = [request.form['tweet']]
		df['joined_lemm'] = [request.form['joined_lemm']]

		predicted_sentiments_rd = predictModel(df)
		predicted_sentiments_twt = predictTwtModel(df)
		predicted_sentiments_com = predictComModel(df)

		tweet_dict = {
			"id": request.form['id'],
			"tweet": request.form['tweet'],
			"joined_lemm": request.form['joined_lemm'],
			"sentiments": 1,
			"predicted_sentiments_rd":predicted_sentiments_rd,
			"predicted_sentiments_twt":predicted_sentiments_twt,
			"predicted_sentiments_com":predicted_sentiments_com,
			"Date": datetime.now()
		}

		conn = engine.connect()
		tweet_update = (
			update(tweet_data).
			where(tweet_data.c.id == tweet_dict['id']).
			values(holder=0)
			)
		conn.execute(tweet_update)


		# Create connection to SQL database
		conn = engine.connect()
		with conn:
			conn.execute(insert(tweet_sentiment),[{
				"Id":request.form['id'],
				"tweet":tweet_dict['tweet'],
				"joined_lemm":tweet_dict['joined_lemm'],
				"sentiments": tweet_dict['sentiments'],
				"predicted_sentiments_rd":tweet_dict['predicted_sentiments_rd'],
				"Date":tweet_dict['Date'],
				"predicted_sentiments_twt":tweet_dict['predicted_sentiments_twt'],
				"predicted_sentiments_com":tweet_dict['predicted_sentiments_com'],

				}])
	else:
		df = pd.DataFrame()
		df['tweet'] = [request.form['tweet']]
		df['joined_lemm'] = [request.form['joined_lemm']]

		predicted_sentiments_rd = predictModel(df)


		tweet_dict = {
			"id": request.form['id'],
			"tweet": request.form['tweet'],
			"joined_lemm": request.form['joined_lemm'],
			"sentiments": 1,
			"predicted_sentiments_rd":predicted_sentiments_rd,
			"Date": datetime.now()
		}

		conn = engine.connect()
		tweet_update = (
			update(tweet_data).
			where(tweet_data.c.id == tweet_dict['id']).
			values(holder=0)
			)
		conn.execute(tweet_update)


		# Create connection to SQL database
		conn = engine.connect()
		with conn:
			conn.execute(insert(tweet_sentiment),[{
				"Id":request.form['id'],
				"tweet":tweet_dict['tweet'],
				"joined_lemm":tweet_dict['joined_lemm'],
				"sentiments": tweet_dict['sentiments'],
				"predicted_sentiments_rd":tweet_dict['predicted_sentiments_rd'],
				"Date":tweet_dict['Date'],
				"predicted_sentiments_twt":42,
				"predicted_sentiments_com":42,

				}])		
	return {}

@app.route("/negative_update", methods = ['POST'])
def negative_update():
	if os.path.isfile("deep_sentiment_twitter_model_trained.h5"):
		df = pd.DataFrame()
		df['tweet'] = [request.form['tweet']]
		df['joined_lemm'] = [request.form['joined_lemm']]

		predicted_sentiments_rd = predictModel(df)
		predicted_sentiments_twt = predictTwtModel(df)
		predicted_sentiments_com = predictComModel(df)

		tweet_dict = {
			"id": request.form['id'],
			"tweet": request.form['tweet'],
			"joined_lemm": request.form['joined_lemm'],
			"sentiments": 0,
			"predicted_sentiments_rd":predicted_sentiments_rd,
			"predicted_sentiments_twt":predicted_sentiments_twt,
			"predicted_sentiments_com":predicted_sentiments_com,
			"Date": datetime.now()
		}

		conn = engine.connect()
		tweet_update = (
			update(tweet_data).
			where(tweet_data.c.id == tweet_dict['id']).
			values(holder=0)
			)
		conn.execute(tweet_update)


		# Create connection to SQL database
		conn = engine.connect()
		with conn:
			conn.execute(insert(tweet_sentiment),[{
				"Id":request.form['id'],
				"tweet":tweet_dict['tweet'],
				"joined_lemm":tweet_dict['joined_lemm'],
				"sentiments": tweet_dict['sentiments'],
				"predicted_sentiments_rd":tweet_dict['predicted_sentiments_rd'],
				"Date":tweet_dict['Date'],
				"predicted_sentiments_twt":tweet_dict['predicted_sentiments_twt'],
				"predicted_sentiments_com":tweet_dict['predicted_sentiments_com'],

				}])
	else:
		df = pd.DataFrame()
		df['tweet'] = [request.form['tweet']]
		df['joined_lemm'] = [request.form['joined_lemm']]

		predicted_sentiments_rd = predictModel(df)


		tweet_dict = {
			"id": request.form['id'],
			"tweet": request.form['tweet'],
			"joined_lemm": request.form['joined_lemm'],
			"sentiments": 0,
			"predicted_sentiments_rd":predicted_sentiments_rd,
			"Date": datetime.now()
		}

		conn = engine.connect()
		tweet_update = (
			update(tweet_data).
			where(tweet_data.c.id == tweet_dict['id']).
			values(holder=0)
			)
		conn.execute(tweet_update)

		# Create connection to SQL database
		conn = engine.connect()
		with conn:
			conn.execute(insert(tweet_sentiment),[{
				"Id":request.form['id'],
				"tweet":tweet_dict['tweet'],
				"joined_lemm":tweet_dict['joined_lemm'],
				"sentiments": tweet_dict['sentiments'],
				"predicted_sentiments_rd":tweet_dict['predicted_sentiments_rd'],
				"Date":tweet_dict['Date'],
				"predicted_sentiments_twt":42,
				"predicted_sentiments_com":42,

				}])
	return {}
@app.route("/neutral_update", methods = ['POST'])
def neutralupdate():
	tweet_dict = {
	"id": request.form['id'],
	"tweet": request.form['tweet'],
	"joined_lemm": request.form['joined_lemm'],
	"Date": datetime.now()
		}

	conn = engine.connect()
	tweet_update = (
		update(tweet_data).
		where(tweet_data.c.id == tweet_dict['id']).
		values(holder=0)
		)
	conn.execute(tweet_update)

	conn = engine.connect()
	with conn:
		conn.execute(insert(filter_data),[{
			"id":request.form['id'],
			"tweet":tweet_dict['tweet'],
			"joined_lemm":tweet_dict['joined_lemm'],
			"Date":tweet_dict['Date'],
			}])
	return {}
@app.route("/data")
def datacalled():
	# import stats
	# session = Session(bind=engine)
	# vals = session.query(tweet_data).filter(tweet_data.c.sentiments != 9).all()
	# data_list = []
	# holder = len(vals)
	# for i in range(holder):
	# 	data_list.append({
    #         'id': vals[i][0],
	# 		'tweet': vals[i][1],
	# 		'sentiments': vals[i][2],
	# 		'predicted_sentiments_rd': vals[i][3],
	# 		'date': vals[i][4],
	# 	})

	# return jsonify(data_list)
	return {}

if __name__ == "__main__":
	app.run(debug=True)

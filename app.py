# Dependencies
# ----------------------------------
from flask import Flask, render_template, redirect, jsonify, request
from flask import send_from_directory
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import MetaData, update, Table
from sqlalchemy.orm import Session
from datetime import datetime
from data import return_data
from modelPredict import predictModel
# import tweet


# initialize flask
app = Flask(__name__)


# Initialize global variable
tweet_dict= {}

# Create database connection String
rds_connection_string = "postgres:postgres@localhost:5432/sentiment_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

metadata = MetaData(engine)
tweet_data = Table('tweet_data', metadata, autoload=True, autoload_with=engine)


# Base = declarative_base()
# # Create class for updating data
# class Tweet(Base):
# 	__tablename__ = 'tweet_data'
# 	id = Column(String, primary_key=True)
# 	tweet = Column(String())
# 	sentiments = Column(Integer)
# 	predicted_sentiments = Column(Integer)
# 	time_data_inserted = Column(Date)

@app.route("/")
def home():
	return render_template("index.html")

@app.route("/voting")
def voting():
	return render_template("voting.html")

@app.route("/stats")
def statistics():
	return render_template("statistics.html")

# @app.route("/apicall")
# def apicalled():
# 	tweet.api_call()

@app.route("/load_tweet")
def load_tweet():
	conn = engine.connect()
	df = pd.read_sql_query('select * from tweet_data WHERE tweet_data.sentiments = 9', con=conn)
	df = df.iloc[0]
	tweet_dict = {
		"id":df['id'],
		"tweet":df['tweet'],
		"sentiments":int(df['sentiments']),
		"predicted_sentiments":df["predicted_sentiments"],
		"time_data_inserted":df['time_data_inserted']
	}

	tweet_dict['predicted_sentiments'] = predictModel(tweet_dict['tweet'])
	print(tweet_dict)

	update_db = (
		update(tweet_data).
		where(tweet_data.c.id == tweet_dict['id']).
		values(predicted_sentiments=tweet_dict['predicted_sentiments'])
	)
	conn.execute(update_db)

	return jsonify(tweet_dict)

@app.route("/positive_update", methods=['POST'])
def positive_update():
	tweetID = request.form['tweetid']
	tweet_dict['id'] = tweetID
	tweet_dict['time_data_inserted'] = datetime.now()
	tweet_dict['sentiment'] = 1
	print(tweet_dict)

	# Create connection to SQL database
	conn = engine.connect()

	# Create object update
	tweet_update = (
		update(tweet_data).
		where(tweet_data.c.id == tweet_dict.id).
		values(sentiments=tweet_dict.sentiments, time_data_inserted=tweet_dict.time_data_inserted)
	)
	conn.execute(tweet_update)

@app.route("/negative_update", methods=['POST'])
def negative_update():
	tweetID = request.form['tweetid']
	tweet_dict['id'] = tweetID
	tweet_dict['time_data_inserted'] = datetime.now()
	tweet_dict['sentiment'] = 0
	print(tweet_dict)

	# Create connection to SQL database
	conn = engine.connect()

	# Create object update
	tweet_update = (
		update(tweet_data).
		where(tweet_data.c.id == tweet_dict.id).
		values(sentiments=tweet_dict.sentiments, time_data_inserted=tweet_dict.time_data_inserted)
	)
	conn.execute(tweet_update)

	# tweet_data['time_data_inserted'] = datetime.now()
	# tweet_data['sentiment'] = 0

	# # Create connection to SQL database
	# engine = create_engine(f'postgresql://{rds_connection_string}')
	# conn = engine.connect()
	# session = Session(bind = engine)

	# # Create tweet instance
	# tweet_upload = Tweet(
	# 	id = tweet_data.id,
	# 	tweet = tweet_data.tweet,
	# 	sentiments = tweet_data.sentiment,
	# 	predicted_sentiments = tweet_data.predicted_sentiments,
	# 	time_data_inserted = tweet_data.time_data_inserted)

	# # Add new instance to database
	# session.add(tweet_upload)
	# session.commit()

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

from pymongo import MongoClient
import pandas as pd
import json

def get_json_df(count):
	f = open("/content/drive/My Drive/Colab Notebooks/data/namuwiki_20190312.json")
	json_data = json.load(f.read())
	for post in posts[:count]:
		string = post['title'] + " " + post["text"]
		if len(string) == 0: continue
		temp = pd.DataFrame({"text":[string]})
		df = df.append(temp, ignore_index = True)
	f.close()
	return df

def get_coll():
	client = MongoClient('localhost', 27017)
	db = client["namuwiki"]
	return db.namu

def get_posts_df(coll, start, count):
	posts = coll.find().skip(start).limit(count)
	df = pd.DataFrame(columns = ["text"])
	for post in posts:
		string = post['title'] + " " + post["text"]
		if len(string) == 0: continue
		temp = pd.DataFrame({"text":[string]})
		df = df.append(temp, ignore_index = True)
		print("title:",post['title'])
		#print(2,post['text'])
		#print(3,df)
	return df
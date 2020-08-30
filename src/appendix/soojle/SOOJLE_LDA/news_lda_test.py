import time
import pandas as pd
from nltk.corpus import stopwords
import numpy as np
import gensim
from gensim.test.utils import datapath
from gensim import corpora
from gensim.models.ldamulticore import LdaMulticore
import pyLDAvis.gensim
import sys
import os
import warnings
import tknizer
from collections import Iterable
warnings.filterwarnings('ignore')
NUM_TOPICS = 5
temp_file = datapath(os.getcwd() + "\\output\\soojle_lda_model")
# 형태소 분석 API 선택
tkn_func = tknizer.konlpy_make_tokens
#tkn_func = tknizer.ETRI_make_tokens
start = time.time()
def main():
	corpus = []
	dictionary = corpora.Dictionary()

	print("start")
	print("docs loading...")
	df = pd.read_csv("news_data.csv")
	idx = 0
	last = len(df)
	while(True):
		if idx > last: break
		print("##",idx,"docs")
		news_df = df.loc[idx:idx,:]
		print("docs tokenizing...")
		tokenized_doc = news_df['text'].apply(lambda x: tkn_func(x, idx))
		print("make Dict...")
		dictionary.add_documents(tokenized_doc)
		print("Token to Corpus...")
		corpus += [dictionary.doc2bow(text) for text in tokenized_doc]
		idx += 1
		get_time()
		print()
	## 싱글 코어
	# ldamodel = gensim.models.ldamodel.LdaModel(
	# 			corpus, 
	# 			num_topics = NUM_TOPICS, 
	# 			id2word = dictionary, 
	# 			passes=20) # passes 알고리즘 반복 횟수
	## 멀티코어
	get_time()
	print("Model Learning...")
	ldamodel = LdaMulticore(
					corpus, 
					num_topics = NUM_TOPICS, 
					id2word = dictionary, 
					passes= 20,
					workers = 4)
	topics = ldamodel.print_topics(
		num_words = 5) # 토픽 단어 제한
	#토픽 및 토픽에 대한 단어의 기여도
	for topic in topics:
		print(topic)
	for i, topic_list in enumerate(ldamodel[corpus]):
	    if i==5:
	        break
	    print(i,'번째 문서의 topic 비율:',topic_list)
	get_time()
	print("model saving...")
	save_model(ldamodel, dictionary)
	visual(ldamodel, corpus, dictionary)	
	print("end")
	#get_topics(ldamodel, dictionary, "IML 사테라")
	
	
def get_topics(ldamodel, dictionary, doc):
	df = pd.DataFrame({'text':[doc]})
	tokenized_doc = df['text'].apply(lambda x: tkn_func(x))
	corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
	for topic_list in ldamodel[corpus]:
		temp = topic_list
		temp = sorted(topic_list, key = lambda x: (x[1]), reverse=True)
		print('해당 문서의 topic 비율:',temp)

def save_model(ldamodel, dictionary):
	ldamodel.save(temp_file)
	dictionary.save(os.getcwd() + "\\output\\soojle_lda_dict")
	print("model saved")

def load_model():
	dictionary = corpora.Dictionary.load(os.getcwd() + "\\output\\soojle_lda_dict")
	lda = LdaModel.load(temp_file)
	print("loaded")
	return lda, dictionary

def get_time():
	print("WorkingTime: {} sec".format(round(time.time()-start,3)))

def visual(ldamodel, corpus, dictionary):
	####시각화 관련
	#pyLDAvis.enable_notebook()
	print("graphing...")
	vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
	print("downloading...")
	pyLDAvis.save_html(vis,"gensim_output.html")
	print("displaying...")
	pyLDAvis.show(vis)
	
if __name__ == '__main__':
	main()
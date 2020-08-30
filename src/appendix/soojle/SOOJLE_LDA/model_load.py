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
from mongo_namu import *

warnings.filterwarnings('ignore')
#주제의 총 토픽
temp_file = datapath(os.getcwd() + "\\output\\soojle_lda_model")
# 형태소 분석 API 선택
tkn_func = tknizer.konlpy_make_tokens
#tkn_func = tknizer.ETRI_make_tokens

def main():
	print("start")
	print("Model Loading")
	ldamodel, dictionary = load_model()
	topics = ldamodel.print_topics(
		num_words = 5) # 토픽 단어 제한
	#토픽 및 토픽에 대한 단어의 기여도
	print("모델 로드 테스트")
	for topic in topics:
		print(topic)
	print("딕셔너리 테스트:",dictionary[0])
	print("end")
	get_topics(ldamodel, dictionary, "IML 사테라")
	
def get_topics(ldamodel, dictionary, doc):
	df = pd.DataFrame({'text':[doc]})
	tokenized_doc = df['text'].apply(lambda x: tkn_func(x))
	corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
	for topic_list in ldamodel[corpus]:
		temp = topic_list
		temp = sorted(topic_list, key = lambda x: (x[1]), reverse=True)
		print('해당 문서의 topic 비율:',temp)

def save_model():
	ldamodel.save(temp_file)
	dictionary.save(os.getcwd() + "\\output\\soojle_lda_dict")
	print("saved")

def show_topics(ldamodel):
	topics = ldamodel.print_topics(
		num_words = 5) # 토픽 단어 제한
	#토픽 및 토픽에 대한 단어의 기여도
	print("모델 로드 테스트")
	for topic in topics:
		print(topic)


def load_model():
	dictionary = corpora.Dictionary.load(os.getcwd() + "\\output\\soojle_lda_dict")
	lda = gensim.models.ldamodel.LdaModel.load(temp_file)
	print("loaded")
	return lda, dictionary

####시각화 관련
# pyLDAvis.enable_notebook()
# print("graphing...")
# vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
# print("displaying...")
# pyLDAvis.show(vis)
# print("downloading...")
# pyLDAvis.save_html(vis, "./gensim_output.html")
if __name__ == '__main__':
	main()
# LDA : 단어가 특정 토픽에 존재할 확률과 문서에 특정 토픽이 존재할 확률을 결합확률로 추정하여 토픽을 추출한다.
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import numpy as np
import gensim
from gensim import corpora
import pyLDAvis.gensim
import sys
import warnings
warnings.filterwarnings('ignore')

print("start")
dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
#####list, 테스트용 뉴스데이터 갯수 결정
documents = dataset.data
# 실제 토픽(답안)
#print(len(dataset.target_names), dataset.target_names)
for idx, data in enumerate(dataset.target_names): print(idx, data)
news_df = pd.DataFrame({'document':documents}).iloc[:,:]

#### 전처리 및 토큰화
# 알파벳을 제외하고 모두 제거
def preprocess(news_df):
	news_df['clean_doc'] = news_df['document'].str.replace("[^a-zA-Z#]", " ")
	# 길이가 3이하인 단어는 제거 (길이가 짧은 단어 제거)
	news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: ' '.join([w for w in x.split() if len(w)>3]))
	# 전체 단어에 대한 소문자 변환
	news_df['clean_doc'] = news_df['clean_doc'].apply(lambda x: x.lower())
	# NLTK로부터 불용어 사전 로드
	stop_words = stopwords.words('english')
	# 토큰화 및 불용어 제거
	tokenized_doc = news_df['clean_doc'].apply(lambda x: x.split())
	tokenized_doc = tokenized_doc.apply(lambda x: [item for item in x if item not in stop_words])
	return tokenized_doc

tokenized_doc = preprocess(news_df)

##### 단어 학습
#####단어 정수 인코딩/인덱스 생성 및 빈도수 기록
# 인덱스 사전 > len으로 길이 가능
dictionary = corpora.Dictionary(tokenized_doc)
# 각 문서 DTM
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
# (단어 인덱스, 빈도수)
# print(corpus[n][m])
# 단어 인덱스로 검색하기
# print(dictionary[66])


#### LDA 모델 훈련시키기
NUM_TOPICS = 20
ldamodel = gensim.models.ldamodel.LdaModel(
			corpus, 
			num_topics = NUM_TOPICS, 
			id2word = dictionary, 
			passes=20) # passes 알고리즘 반복 횟수
topics = ldamodel.print_topics(
	num_words = 7) # 토픽 단어 제한


#### 결과 값 확인
#토픽 및 토픽에 대한 단어의 기여도
for topic in topics:
	print(topic)
#문서별 토픽 분포
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율:',topic_list)

def get_topics(doc):
	df = pd.DataFrame({'document':[doc]})
	tokenized_doc = preprocess(df)
	corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
	for topic_list in ldamodel[corpus]:
		temp = topic_list
		temp = sorted(topic_list, key = lambda x: (x[1]), reverse=True)
		print('해당 문서의 topic 비율:',temp)

#### 시각화
## 주피터 노트북 only
# pyLDAvis.enable_notebook()
# print("graphing...")
# vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
# #pyLDAvis.display(vis)
# print("displaying...")
# #pyLDAvis.show(vis)
# print("donwloading...")
# pyLDAvis.save_html(vis, "./gensim_output.html")
# print("end")
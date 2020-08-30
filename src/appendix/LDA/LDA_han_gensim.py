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
import tknizer
warnings.filterwarnings('ignore')
print("start")
print("Topic answers(6): 정치, 경제, 사회, 생활, 세계, IT/과학")

news_df = pd.read_csv("data/NNST_data2.csv").loc[:100,:]
tokenized_doc = news_df['text'].apply(lambda x: tknizer.make_tokens(x))
dictionary = corpora.Dictionary(tokenized_doc)
corpus = [dictionary.doc2bow(text) for text in tokenized_doc]
# (단어 인덱스, 빈도수)
# print(corpus[n][m])
# 단어 인덱스로 검색하기
# print(dictionary[66])
NUM_TOPICS = 6
ldamodel = gensim.models.ldamodel.LdaModel(
			corpus, 
			num_topics = NUM_TOPICS, 
			id2word = dictionary, 
			passes=20) # passes 알고리즘 반복 횟수
topics = ldamodel.print_topics(
	num_words = 5) # 토픽 단어 제한
#토픽 및 토픽에 대한 단어의 기여도
for topic in topics:
	print(topic)
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==5:
        break
    print(i,'번째 문서의 topic 비율:',topic_list)

#### 시각화
## 주피터 노트북 only
#pyLDAvis.enable_notebook()
print("graphing...")
vis = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
#pyLDAvis.display(vis)
print("displaying...")
#pyLDAvis.show(vis)
print("donwloading...")
pyLDAvis.save_html(vis, "./output/gensim_output.html")
print("end")
# LSA : DTM을 차원 축소 하여 축소 차원에서 근접 단어들을 토픽으로 묶는다.
import pandas as pd
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
import numpy as np

dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
#####list, 테스트용 뉴스데이터
documents = dataset.data
# 실제 토픽(답안)
#print(len(dataset.target_names), dataset.target_names)


#### 전처리
news_df = pd.DataFrame({'document':documents})
# 알파벳을 제외하고 모두 제거`
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


#### 역토큰화
detokenized_doc = []
for i in range(len(news_df)):
	t = " ".join(tokenized_doc[i])
	detokenized_doc.append(t)
news_df['clean_doc'] = detokenized_doc


#### TF-IDF 행렬 만들기
vectorizer = TfidfVectorizer(stop_words = "english",
							#max_features = 10000, # 최대 단어 제한
							max_df = 0.5,
							smooth_idf = True)
X = vectorizer.fit_transform(news_df['clean_doc'])


#### 토픽모델링
svd_model = TruncatedSVD(n_components = 20, 
						algorithm = 'randomized',
						n_iter = 100,
						random_state = 122)
svd_model.fit(X)
#(토픽의 수, 해당 토픽과 관련된 단어)
print(np.shape(svd_model.components_))

#결과출력
terms = vectorizer.get_feature_names()
def get_topics(components, feature_names, n = 5):
	for idx, topic in enumerate(components):
		print("Topic %d:" % (idx+1), 
			[(feature_names[i], topic[i]) for i in topic.argsort()[:-n - 1:-1]])
get_topics(svd_model.components_, terms)
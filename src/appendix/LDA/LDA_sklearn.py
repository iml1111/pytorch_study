import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

print("start")

##### 데이터 가져오기
data = pd.read_csv('abcnews-date-text.csv', error_bad_lines=False)
print("총 뉴스 갯수: ",len(data))


#### 전처리
print("tokenizing...")
text = data.loc[:,['headline_text']]
text['headline_text'] = text.apply(lambda row: 
	nltk.word_tokenize(row['headline_text']), axis=1)

print("filtering...")
stop = stopwords.words('english')
text['headline_text'] = text['headline_text'].apply(
	lambda x: [word for word in x if word not in (stop)])
tokenized_doc = text['headline_text'].apply(lambda x: [word for word in x if len(word) > 3])


#### 문서를 벡터화시키기
print("Detokenizing...")
detokenized_doc = []
for i in range(len(text)):
    t = ' '.join(tokenized_doc[i])
    detokenized_doc.append(t)
text['headline_text'] = detokenized_doc

print("make TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', 
max_features= 1000) # 상위 1000개의 단어만 보존(테스트)
X = vectorizer.fit_transform(text['headline_text'])
#X.shape # TF-IDF 행렬의 크기 확인


#### 토픽 모델링 - LDA
lda_model = LDA(n_components = 10, learning_method = "online", random_state = 111, max_iter = 1)
lda_top = lda_model.fit_transform(X)

### 테스트 출력
terms = vectorizer.get_feature_names() # 단어 집합. 1,000개의 단어가 저장됨.
def get_topics(components, feature_names, n=5):
    for idx, topic in enumerate(components):
        print("Topic %d:" % (idx+1), [(feature_names[i], topic[i].round(2)) for i in topic.argsort()[:-n - 1:-1]])

get_topics(lda_model.components_, terms)
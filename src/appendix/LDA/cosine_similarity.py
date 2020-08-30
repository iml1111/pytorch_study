import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

data = pd.read_csv("movies_metadata.csv", low_memory=False)
data = data.head(20000)#데이터 줄이기
tfidf = TfidfVectorizer(stop_words = 'english')
data['overview'] = data['overview'].fillna("")
# overview에 대해서 tf-idf 수행
tfidf_matrix = tfidf.fit_transform(data['overview'])
#print("(N개의 데이터, 총 M개의 단어)")
#print(tfidf_matrix.shape)
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
indices = pd.Series(data.index, index = data['title']).drop_duplicates()
#print("영화별 인덱싱")
#print(indices[:4])

def get_recommendations(title, cosine_sim = cosine_sim):
	# 선택한 영화의 타이틀로부터 해당되는 인덱스를 받아옵니다. 이제 선택한 영화를 가지고 연산할 수 있습니다.
    idx = indices[title]
    # 모든 영화에 대해서 해당 영화와의 유사도를 구합니다.
    sim_scores = list(enumerate(cosine_sim[idx]))
    # 유사도에 따라 영화들을 정렬합니다.
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # 가장 유사한 10개의 영화를 받아옵니다.
    sim_scores = sim_scores[1:11]
    # 가장 유사한 10개의 영화의 인덱스를 받아옵니다.
    movie_indices = [i[0] for i in sim_scores]
    # 가장 유사한 10개의 영화의 제목을 리턴합니다.
    return data['title'].iloc[movie_indices]
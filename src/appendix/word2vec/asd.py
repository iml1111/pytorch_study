import gensim
model = gensim.models.Word2Vec.load('ko.bin')

def search(string):
	a=model.wv.most_similar(string)
	print(a)
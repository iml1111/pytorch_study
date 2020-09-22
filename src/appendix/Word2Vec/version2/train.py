'''
https://github.com/jojonki/word2vec-pytorch
# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
'''
import torch.nn as nn
from model import SkipGram

embed_size = 100 # 임베딩할 차원수 
learning_rate = 0.001
n_epoch = 30


def train_skipgram():
	losses = []
	loss_fn = nn.MSELoss()
	model = SkipGram(vocab_size, embed_size)
'''
Word2Vec 신경망 모듈
'''
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class SkipGramModel(nn.Module):
    '''Word2Vec Skip-Gram model 신경망 클래스'''
    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
        self.init_emb()

    def init_emb(self):
        '''각 layer 가중치 초기화'''
        initrange = 0.5 / self.emb_dimension
        self.u_embeddings.weight.data.uniform_(-initrange, initrange)
        self.v_embeddings.weight.data.uniform_(-0, 0)

    def forward(self, pos_u, pos_v, neg_v):
        """
        forward process.
        the pos_u and pos_v shall be the same size.
        the neg_v shall be {negative_sampling_count} * size_of_pos_u
        eg:
        5 sample per batch with 200d word embedding and 6 times neg sampling.
        pos_u 5 * 200
        pos_v 5 * 200
        neg_v 5 * 6 * 200
        :param pos_u:  positive pairs u, list
        :param pos_v:  positive pairs v, list
        :param neg_v:  negative pairs v, list
        :return:

        pos_u: batch_size * 중심단어
        pos_v: batch_size * 주변단어
        neg_v: 5 * batch_size * 거짓된 주변단어
        """
        emb_u = self.u_embeddings(pos_u)
        emb_v = self.v_embeddings(pos_v)

        score = torch.mul(emb_u, emb_v).squeeze()
        score = torch.sum(score, dim=1)
        score = F.logsigmoid(score)

        neg_emb_v = self.v_embeddings(neg_v)
        neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
        neg_score = F.logsigmoid(-1 * neg_score)

        return -1 * (torch.sum(score) + torch.sum(neg_score))

    def save_embedding(self, id2word: dict, file_name: str='word_vectors.txt', use_cuda: bool=False):
        """
        Save all embeddings to file.
        As this class only record word id, so the map from id to word has to be transfered from outside.
        :param id2word: map from word id to word.
        :param file_name: file name.
        :param use_cuda:
        :return:
        """
        if use_cuda:
            embedding = self.u_embeddings.weight.cpu().data.numpy()
        else:
            embedding = self.u_embeddings.weight.data.numpy()
        fout = open(file_name, 'w', encoding='utf-8')
        fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
        for wid, w in id2word.items():
            e = embedding[wid]
            e = ' '.join(map(lambda x: str(x), e))
            fout.write('%s %s\n' % (w, e))

def test():
    vocab_size, embed_size = 100, 100
    model = SkipGramModel(vocab_size, embed_size)
    id2word = dict()
    for i in range(vocab_size):
        id2word[i] = str(i)
    model.save_embedding(id2word)


if __name__ == '__main__':
    test()
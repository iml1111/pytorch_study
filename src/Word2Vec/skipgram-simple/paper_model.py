import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class SkipGramModel(nn.Module):
   def __init__(self, emb_size, emb_dimension):
      super(SkipGramModel, self).__init__()
      self.emb_size = emb_size
      self.emb_dimension = emb_dimension
      self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)
      self.v_embeddings = nn.Embedding(emb_size, emb_dimension, sparse = True)
      self.init_emb()
   def init_emb(self):
      initrange = 0.5 / self.emb_dimension
      self.u_embeddings.weight.data.uniform_(-initrange, initrange)
      self.v_embeddings.weight.data.uniform_(-0, 0)
   def forward(self, pos_u, pos_v, neg_v):
      emb_u = self.u_embeddings(pos_u)
      emb_v = self.v_embeddings(pos_v)
      score = torch.mul(emb_u, emb_v).squeeze()
      score = torch.sum(score, dim = 1)
      score = F.logsigmoid(score)
      neg_emb_v = self.v_embeddings(neg_v)
      neg_score = torch.bmm(neg_emb_v, emb_u.unsqueeze(2)).squeeze()
      neg_score = F.logsigmoid(-1 * neg_score)
      return -1 * (torch.sum(score)+torch.sum(neg_score))
   def save_embedding(self, id2word, file_name, use_cuda):
      if use_cuda:
         embedding = self.u_embeddings.weight.cpu().data.numpy()
      else:
         embedding = self.u_embeddings.weight.data.numpy()
      fout = open(file_name, 'w')
      fout.write('%d %d\n' % (len(id2word), self.emb_dimension))
      for wid, w in id2word.items():
         e = embedding[wid]
         e = ' '.join(map(lambda x: str(x), e))
         fout.write('%s %s\n' % (w, e))
def test():
   model = SkipGramModel(100, 100)
   id2word = dict()
   for i in range(100):
      id2word[i] = str(i)
   model.save_embedding(id2word)


class SkipGramNeg(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super(SkipGramNeg, self).__init__()
        self.input_emb = nn.Embedding(vocab_size, emb_dim)
        self.output_emb = nn.Embedding(vocab_size, emb_dim)
        self.log_sigmoid = nn.LogSigmoid()

        initrange = (2.0 / (vocab_size + emb_dim)) ** 0.5  # Xavier init
        self.input_emb.weight.data.uniform_(-initrange, initrange)
        self.output_emb.weight.data.uniform_(-0, 0)


    def forward(self, target_input, context, neg):
        """
        :param target_input: [batch_size]
        :param context: [batch_size]
        :param neg: [batch_size, neg_size]
        :return:
        """
        # u,v: [batch_size, emb_dim]
        v = self.input_emb(target_input)
        u = self.output_emb(context)
        # positive_val: [batch_size]
        positive_val = self.log_sigmoid(torch.sum(u * v, dim=1)).squeeze()

        # u_hat: [batch_size, neg_size, emb_dim]
        u_hat = self.output_emb(neg)
        # [batch_size, neg_size, emb_dim] x [batch_size, emb_dim, 1] = [batch_size, neg_size, 1]
        # neg_vals: [batch_size, neg_size]
        neg_vals = torch.bmm(u_hat, v.unsqueeze(2)).squeeze(2)
        # neg_val: [batch_size]
        neg_val = self.log_sigmoid(-torch.sum(neg_vals, dim=1)).squeeze()

        loss = positive_val + neg_val
        return -loss.mean()

    def predict(self, inputs):
        return self.input_emb(inputs)
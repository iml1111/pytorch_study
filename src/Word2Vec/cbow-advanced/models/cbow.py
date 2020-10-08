import torch.nn as nn


class CBOW(nn.Module):

    def __init__(self, vocab_size, embd_size, window_size, hidden_size):
        super(CBOW, self).__init__()
        self.emb = nn.Embedding(vocab_size, embd_size)
        self.layers = nn.Sequential(
            nn.Linear(2 * window_size * embd_size, hidden_size),
            nn.ReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, vocab_size),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        emb_x = self.emb(x)
        flatten_emb_x = emb_x.view((emb_x.size(0), -1))
        y = self.layers(flatten_emb_x)
        return y
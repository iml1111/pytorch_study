import torch.nn as nn


class SequenceClassifier(nn.Module):
    '''RNN 시퀸스 분류기'''
    def __init__(
        self,
        input_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        super().__init__()

        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True, # (b, h, w) 순으로 하기위해
            dropout=dropout_p,
            bidirectional=True,
        )
        self.layers = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        '''
        # input_size가 28이지만 시퀸셜이므로 한번에 28 * 28이 들어감
        x = (batch_size, h, w)
        z = (batch_size, h, hidden_size * 2)
        # z의 맨 마지막 스텝만 가져옴
        z_last = (batch_size, hidden_size * 2)
        y = (batch_size, output_size)
        '''
        z, _ = self.rnn(x)
        z = z[:, -1]
        y = self.layers(z)
        return y
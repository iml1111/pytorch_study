import torch.nn as nn


class RNNClassifier(nn.Module):
    ''' 
    # RNN 텍스트 분류기 신경망 
    input_size = one hot vector의 길이 (== 총 단어 수)
    word_vec_size = 임베딩을 거쳐 나올 벡터 사이즈
    hidden_size = 아웃풋으로 나올 히든 사이즈
    n_classes = 최종 결과 클래스 
    n_layers = RNN 계층 수
    dropout_p = 계층별 드롭아웃 비율
    '''
    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        n_classes,
        n_layers=4, # 레이어가 4이상 깊어지면 배니싱 문제 생김
        dropout_p=.3):

        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)
        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True)
        
        # bidirection RNN 이므로 -> hidden_size * 2
        self.generator = nn.Linear(hidden_size * 2, n_classes)
        
        # 두가지 경우의 수가 존재함
        # 1. act = LogSoftmax, loss = NLLLoss
        # 2. act = Softmax, loss = CrossEntropy
        # 뭐가 더 좋은지는 모르겠음
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        '''
        # 모든 문장(X)는 신경망 진입 전에 길이가 맞춰진 상태로 시작됨
        # x = (batch_size, length) > one hot vector의 인덱스
        '''
        
        # embed_x = (batch_size, length, word_vec_size)
        embed_x = self.emb(x)

        # z = (batch_size, length, hidden_size * 2) > bi-RNN이므로 2배
        z, _ = self.rnn(embed_x)
        
        # z[:, -1] = (batch_size, hidden_size * 2) > 마지막 time-step만 보존
        # sliced_z = (batch_size, n_classes) 
        sliced_z = self.generator(z[:, -1])
        
        # y = (batch_size, n_classes)
        y = self.activation(sliced_z)
        return y
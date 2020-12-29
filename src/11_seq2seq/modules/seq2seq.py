import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.nn.utils.rnn import pad_packed_sequence as unpack
import modules.data_loader as data_loader


class Encoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Encoder, self).__init__()

        self.rnn = nn.LSTM(
            input_size=word_vec_size,
            # Bi-direction인 Encoder와
            # Uni-direction인 Decoder와의 괴리를 맞추기 위함
            hidden_size=int(hidden_size / 2),
            num_layers=n_layers,
            dropout=dropout_p,
            batch_first=True,
            bidirectional=True
        )

    def forward(self, emb):
        '''
        원문을 인코딩하는 영역
        - time-step을 신경쓸 필요 없음, 한꺼번에 넘겨서 한꺼번에 받음
        - PAD를 가진 시퀸스를 효율적으로 병렬연산 하기 위해, pack, unpack으로 처리
        '''
        if isinstance(emb, tuple):
            # x = (batch_size, length, word_vec_size)
            # lengths = (batch_size) - 각 문장마다의 길이 (PAD 제외)
            x, lengths = emb
            x = pack(x, lengths.tolist(), batch_first=True)
        else:
            x = emb

        # y = (batch_size, length, hidden_size)
        # h[0] = (num_layers * 2, batch_size, hidden_size / 2)
        y, h = self.rnn(x)
        if isinstance(emb, tuple):
            y, _ = unpack(y, batch_first=True)

        return y, h


class Decoder(nn.Module):

    def __init__(self, word_vec_size, hidden_size, n_layers=4, dropout_p=.2):
        super(Decoder, self).__init__()

        self.rnn = nn.LSTM(
            # input feeding으로 이전 스텝의 결과값을 이어붙임
            input_size=word_vec_size + hidden_size,
            hidden_size=hidden_size,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=False,
            batch_first=True
        )

    def forward(self, emb_t, h_t_1_tilde, h_t_1):
        '''
        인코더에서 넘어온 원문의 hidden_state를 
        이용해 번역문으로 바꾸는 영역
        
        - time-step을 고려하기 위해 한번 호출마다 한 time-step만 수행
        - 따라서, emb_t, h_t_1_tilde는 모두 각 배치의 한개씩만 가져옴
        
        emb_t: 현재 time-step의 단어
        h_t_1_tilde: 이전 타임스텝의 예측 벡터값 (input feeding)
        h_t_1: 이전 타임스텝에서 넘어오면서 받은 Decoder 히든 사이즈
        (cell, hidden state로 나뉘며 신경쓰지말고 그대로 넘겨주면 됨)
        '''
        # emb_t = (batch_size, 1, word_vec_size)
        # h_t_1_tilde = (batch_size, 1, hidden_size)
        # h_t_1[0] = (n_layers, batch_size, hidden_size) - 히든 스테이트
        batch_size = emb_t.size(0)
        hidden_size = h_t_1[0].size(-1)

        if h_t_1_tilde is None:
            # 첫 번째 time-step의 경우 0으로 임의값 할당
            h_t_1_tilde = emb_t.new(batch_size, 1, hidden_size).zero_()

        # Input Feeding
        # x = (batch_size, 1, word_vec_size + hidden_size)
        x = torch.cat([emb_t, h_t_1_tilde], dim=-1)
        # y = (batch_size, 1, hidden_size)
        # h[0] = (n_layers, batch_size, hidden_size)
        y, h = self.rnn(x, h_t_1)
        return y, h


class Attention(nn.Module):

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.linear = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, h_src, h_t_tgt, mask=None):
        '''
        Encoder 단의 원문 벡터값과 현재 값의 유사도 산출
        h_src: Encoder 전체 time-step의 히든 스테이트 출력값
        h_t_tgt: Decoder의 현재 time-step 히든 스테이트 출력값
        mask: 각 문장 각 토큰에 대한 PAD 여부 (True or False)
        '''
        # h_src = (batch_size, length, hidden_size)
        # h_t_tgt = (batch_size, 1, hidden_size)
        # mask = (batch_size, length)
        
        # query = (batch_size, 1, hidden_size)
        query = self.linear(h_t_tgt)
        # weight = (batch_size, 1, length)
        weight = torch.bmm(query, h_src.transpose(1, 2))

        if mask:
            # PAD token 자리의 가중치를 모두 -inf로 치환(학습 미반영)
            # 마스크를 씌우기 위해 mask가 해당 weight의 shape과 같아야함
            # mask.unsqueeze(1) = (batch_size, 1, length)
            weight.masked_fill_(mask.unsqueeze(1), -float('inf'))

        weight = self.softmax(weight)
        # context_vector = (batch_size, 1, hidden_size)
        context_vector = torch.bmm(weight, h_src)
        return context_vector


class Generator(nn.Module):

    def __init__(self, hidden_size, output_size):
        super(Generator, self).__init__()

        self.output = nn.linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=-1) # 마지막 차원으로 수행

    def forward(self, x):
        # x = (batch_size, length, hidden_size)

        # Generator의 경우, 모든 time-step의 값을 
        # Decoder에서 한번에 받아서 수행
        # y = (batch_size, length, output_size)
        y = self.softmax(self.output(x))
        return y


class Seq2Seq(nn.Module):

    def __init__(
        self,
        input_size,
        word_vec_size,
        hidden_size,
        output_size,
        n_layers=4,
        dropout_p=.2
    ):
        super(Seq2Seq, self).__init__()
        # 원문 vocab size와 같음
        self.input_size = input_size
        self.word_vec_size = word_vec_size
        self.hidden_size = hidden_size
        # 번역문 vocab size와 같음
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p

        self.emb_src = nn.Embedding(input_size, word_vec_size)
        self.emb_dec = nn.Embedding(output_size, word_vec_size)

        self.encoder = Encoder(
            word_vec_size=word_vec_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_p=dropout_p,
        )
        self.decoder = Decoder(
            word_vec_size=word_vec_size,
            hidden_size=hidden_size,
            n_layers=n_layers,
            dropout_p=dropout_p,
        )
        self.attn = Attention(hidden_size)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.tanh = nn.Tanh()
        self.gneerator = Generator(hidden_size, output_size)
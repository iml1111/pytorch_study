import torch.nn as nn
import torch.nn.functional as F


class CBOW(nn.Module):
    def __init__(self, vocab_size, embd_size, context_size, hidden_size):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
        self.linear1 = nn.Linear(2*context_size*embd_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, inputs):
        # inputs? 임베딩 레이어는 복수 인풋도 가능하다?
        # 단 룩업 테이블이므로, 그냥 inputs 길이만큼 반환됨
        embedded = self.embeddings(inputs).view((1, -1))
        hid = F.relu(self.linear1(embedded))

        # 결과적으로 중심 단어의 아이디 자체를 예측함
        out = self.linear2(hid)
        log_probs = F.log_softmax(out)
        return log_probs
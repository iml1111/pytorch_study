import torch
import torch.nn as nn
import torch.nn.functional as F


class SkipGram(nn.Module):
    def __init__(self, vocab_size, embd_size):
        super(SkipGram, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embd_size)
    
    def forward(self, focus, context):
        '''
        focus : 중심단어의 아이디
        context : 주변 단어의 아이디 (네거티브 샘플일수도 있음)
        '''
    	# 중심단어 임베딩하기
        embed_focus = self.embeddings(focus).view((1, -1))
        # 주변단어 임베딩하기
        embed_ctx = self.embeddings(context).view((1, -1))
        # 두 단어간의 내적값이 score 
        # 같을수록 1에 가까워지고
        # 다를수록 0에 가까워짐

        # 결과적으로 두 단어 임베딩간의 유사도를 예측함
        score = torch.mm(embed_focus, torch.t(embed_ctx))
        log_probs = F.logsigmoid(score)
    
        return log_probs
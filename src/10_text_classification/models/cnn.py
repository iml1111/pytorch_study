import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    ''' 
    # CNN 텍스트 분류기 신경망 
    input_size = one hot vector의 길이 (== 총 단어 수)
    word_vec_size = 임베딩을 거쳐 나올 벡터 사이즈
    n_classes = 최종 결과 클래스 
    use_batch_norm = 배치놈 사용여부 (논문에선 사용하지 않음)
    dropout_p = 계층별 드롭아웃 비율
    windows_size = 각 필터의 windows 단위 > 얼마나 많은 단어들을 묶을 것인가
    n_filters = 각 필터의 커널 갯수 > 얼마나 많은 패턴을 잡을 것인가
    '''
    def __init__(
        self,
        input_size,
        word_vec_size,
        n_classes,
        use_batch_norm=False, 
        dropout_p=.5,
        window_sizes=[3, 4, 5],
        n_filters=[100, 100, 100]):

        self.input_size = input_size # 
        self.word_vec_size = word_vec_size
        self.n_classes = n_classes
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        self.window_sizes = window_sizes
        self.n_filters = n_filters
        super().__init__()

        self.emb = nn.Embedding(input_size, word_vec_size)

        # 3개의 부분 신경망을 동시에 병렬적으로 진행
        self.feature_extractors = nn.ModuleList()
        for window_size, n_filter in zip(window_sizes, n_filters):
            self.feature_extractors.append(
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=1, # 임베딩 벡터는 하나의 채널만 가짐 (Mnist와 동일)
                        out_channels=n_filter, # output으로 100개의 패턴을 원함
                        kernel_size=(window_size, word_vec_size),
                    ),
                    nn.ReLU(),
                    nn.BatchNorm2d(n_filter) if use_batch_norm else nn.Dropout(dropout_p),
                )
            )

        # 위의 3개의 신경망에서 나온걸 그대로 이어붙여서 넣음
        self.generator = nn.Linear(sum(n_filters), n_classes)

        # 두가지 경우의 수가 존재함
        # 1. act = LogSoftmax, loss = NLLLoss
        # 2. act = Softmax, loss = CrossEntropy
        # 뭐가 더 좋은지는 모르겠음
        self.activation = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # first, x = (batch_size, length)
        
        # x = (batch_size, length, word_vec_size)
        x = self.emb(x)

        min_length = max(self.window_sizes)
        if min_length > x.size(1):
            # 해당 문장이 단어 갯수가 가장 긴 윈도우보다 작을 경우
            # pad를 만들어 문장을 그만큼 채워주어야 함
            # pad = (batch_size, min_length - length, word_vec_size)
            pad = x.new(x.size(0),
                        min_length - x.size(1),
                        self.word_vec_size).zero_()
            # x = (batch_size, min_length, word_vec_size)
            x = torch.cat([x, pad], dim=1)

        # CNN 신경망에 넣어주기 위해 채널의 형태로 만들어줘야 함
        # x = (batch_size, 1, length, word_vec_size)
        x = x.unsqueeze(1)

        cnn_outs = []
        for block in self.feature_extractors:
            # cnn_out = (batch_size, n_filter, length - windows + 1, 1)
            cnn_out = block(x)

            # 문장의 길이는 계속 가변적으로 주어짐
            # 때문에 max_pooling 계층을 사전에 정의 불가능
            # 계속 동적으로 문장에 따라 값을 바꿔주어야 함
            # 해당 max_pooling으로 인해, 문장의 단어 갯수가 몇개라도 
            # 똑같이 n_filter만큼의 텐서로 바뀜.
            # cnn_out =(batch_size, n_filter)
            cnn_out = nn.functional.max_pool1d(
                input=cnn_out.squeeze(-1),
                kernel_size=cnn_out.size(-2)
            ).squeeze(-1)
            cnn_outs += [cnn_out]

        # 3개의 신경망에서 나온 값을 그대로 이어붙임
        # cnn_outs = (batch_size, sum(n_filters))
        cnn_outs = torch.cat(cnn_outs, dim=-1)
        
        # z = (batch_size, n_classes)
        z = self.generator(cnn_outs)
        
        # y = (batch_size, n_classes)
        y = self.activation(z)
        return y


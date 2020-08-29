from torchtext import data


class DataLoader:
    '''
    Data loader class to load text file using torchtext library.
    '''
    def __init__(
        self,
        train_fn,
        batch_size=64,
        valid_ratio=.2,
        device=-1,
        max_vocab=999999,
        min_freq=1,
        use_eos=False,
        shuffle=True):
        '''
        DataLoader initialization.
        :param train_fn: 학습 셋 파일 이름
        :param batch_size: 배치 사이즈
        :param device: Device-id to load data (-1 for CPU)
        :param max_vocab: 최대 단어 사전 한도
        :param min_freq: 단어의 최소 빈도 -> 이 빈도를 넘어야 단어 사전에 들어감
        :param use_eos: If it is True, put <EOS> after every end of sentence.
        :param shuffle: If it is True, random shuffle the input data.
        '''
        super().__init__()
        # text, label에 대한 data field 정의
        self.label = data.Field(
            # 클래스이므로 연속 데이터가 아님 (애초에 1개임)
            sequential=False,
            # 단어 사전은 필요없지만 클래스를 세주므로 쓰자 (2개일 거임)
            use_vocab=True,
            # 자연어 생성이 아니므로 모르는 단어를 신경쓰지 않음
            unk_token=None)
        self.text = data.Field(
            # 단어 사전 활성화
            use_vocab=True,
            # batch를 앞의 dim으로 당겨줌, 필수
            batch_first=True,
            # 아래 2개는 자연어 생성이 아니면 신경 ㄴㄴ
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None)

        # train set 파일을 불러와 사전에 정의한 field대로 
        # 태뷸라 데이터 셋으로 만들어 스플릿
        train, valid = data.TabularDataset(
            path=train_fn,
            # 컬럼을 나누는 기준으로 tab으로
            format="tsv",
            fields=[
                ('label', self.label),
                ('text', self.text),
            ],).split(split_ratio=(1 - valid_ratio))

        # 위에서 만든 데이터셋으로 로더로 만듬(iter)
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            # 비슷한 길이끼리 미니 배치를 만들도록 정렬
            sort_key=lambda x: len(x.text),
            # 미니 배치 내에서 sort를 할 것인가?
            # 하면 긴 녀석이 먼저 나오고 짧은게 나중에 나옴
            sort_within_batch=True)
            
        # 실제로는 긍정/부정 밖에 없는 단어 사전
        self.label.build_vocab(train)
        # 실제 단어 사전 생성
        self.text.build_vocab(train,
                              max_size=max_vocab,
                              min_freq=min_freq)
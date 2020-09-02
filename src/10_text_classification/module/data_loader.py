import torch
from torchtext import data
from torch.utils.data import Dataset


class DataLoader:
    '''
    텍스트 파일을 torchtext로 불러오기 위한 데이터 로더 객체
    train_fn: 트레인 셋 파일 이름
    batch_size: 배치 사이즈
    device: 연산할 디바이스 (-1 for CPU)
    max_vocab: 최대 단어 사전 크기 (단어 갯수)
    min_freq: 단어 사전에 등록될 최소 빈도수
    use_eos: If it is True, put <EOS> after every end of sentence.
    shuffle: If it is True, random shuffle the input data.
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

        super().__init__()
        
        # 가져올 데이터의 필드 정의
        self.label = data.Field(
            sequential=False, # 정답은 연속적 데이터가 아님
            use_vocab=True, # 빈도수를 세어서 class 개수로 활용
            unk_token=None)
        self.text = data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False,
            eos_token='<EOS>' if use_eos else None)

        # 데이터 파일을 불러와 태뷸라 데이터 객체로 만듬
        # 이때 포맷은 tsv, TAB으로 컬럼을 나눔
        train_dataset = data.TabularDataset(
            path=train_fn,
            format='tsv',
            fields=[('label', self.label),
                    ('text', self.text)])
        train, valid = train_dataset.split(split_ratio=(1 - valid_ratio))

        # 학습에 직접 사용될 데이터 로더 객체로 변환
        self.train_loader, self.valid_loader = data.BucketIterator.splits(
            (train, valid),
            batch_size=batch_size,
            device='cuda:%d' % device if device >= 0 else 'cpu',
            shuffle=shuffle,
            # 비슷한 길이끼리 미니 배치를 만들도록 정렬
            # 미니 배치 내에서 sort를 할 것인가?
            # 하면 긴 녀석이 먼저 나오고 짧은게 나중에 나옴
            sort_key=lambda x: len(x.text),
            sort_within_batch=True)

        # 각 데이터 필드에 대한 단어 사전 생성
        self.label.build_vocab(train)
        self.text.build_vocab(train, 
                              max_size=max_vocab,
                              min_freq=min_freq)


'''
# 여기부터 BERT 전용 코드
'''
class TokenizerWrapper():

    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def collate(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=self.max_length
        )

        return {
            'text': texts,
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long),
        }


class BertDataset(Dataset):

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        return {
            'text': text,
            'label': label,
        }
from collections import Counter
from collections import defaultdict
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
        

class CbowDataHandler:
    '''Cbow Total Dataset'''

    def __init__(
        self,
        file_name,
        train_ratio,
        batch_size,
        window_size):

        with open(file_name, encoding='utf-8') as f:
            corpora = [sentence.split() for sentence in f.readlines()[:]] #check

        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.window_size = window_size
        self.vocab_size, self.w2i, self.i2w = self.build_vocab(corpora)    

        data, label = self.create_dataset(corpora)
        self.train_loader, self.valid_loader = self.create_dataloader(data, label)

    def build_vocab(self, corpora):
        '''Build Vocabulary Dictionary'''
        tokens = sum(corpora, [])
        vocab = Counter(tokens).most_common()
        
        w2i = defaultdict(lambda: -1)
        i2w = defaultdict(lambda: '<__UNK__>')
        for idx, (word, freq) in enumerate(vocab):
            w2i[word] = idx
            i2w[idx] = word

        return len(w2i), w2i, i2w

    def create_dataset(self, corpora):
        '''Create Word2Vec Windows size Dataset'''
        data = []
        label = []
        for corpus in corpora:
            for i in range(self.window_size, len(corpus) - self.window_size):
                context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i: 
                        context.append(self.w2i[corpus[j]])
                data.append(context)
                label.append(self.w2i[corpus[i]])
        return data, label

    def create_dataloader(self, data, label):
        '''Create Pytorch DataLoader'''
        class CbowDataset(Dataset):
            '''Pytorch Custom Dataset'''
            def __init__(self, data, label):
                self.data = data
                self.label = label
                super().__init__()

            def __len__(self):
                return self.data.size(0)

            def __getitem__(self, idx):
                x = self.data[idx]
                y = self.label[idx]
                return x, y

        data = torch.tensor(data)
        label = torch.tensor(label)
        train_cnt = int(data.size(0) * self.train_ratio)
        valid_cnt = data.size(0) - train_cnt
        indices = torch.randperm(data.size(0))
        
        train_x, valid_x = torch.index_select(
            data,
            dim=0,
            index=indices
        ).split([train_cnt, valid_cnt], dim=0)
        train_y, valid_y = torch.index_select(
            label,
            dim=0,
            index=indices
        ).split([train_cnt, valid_cnt], dim=0)

        train_loader = DataLoader(
            dataset=CbowDataset(train_x, train_y),
            batch_size=self.batch_size,
            shuffle=True
        )
        valid_loader = DataLoader(
            dataset=CbowDataset(valid_x, valid_y),
            batch_size=self.batch_size,
            shuffle=True
        )

        return train_loader, valid_loader


if __name__ == '__main__':
    data_handler = CbowDataHandler(
        file_name='../../data/w2v_train.tsv',
        window_size=2,
        train_ratio=.9,
        batch_size=128,
    )
    print('|train| =', len(data_handler.train_loader.dataset),
          '|valid| =', len(data_handler.valid_loader.dataset))
    print('|vocab_size| =', data_handler.vocab_size)

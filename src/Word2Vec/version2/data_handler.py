import numpy as np
from collections import defaultdict
from collections import deque

np.random.seed(1)


class DataHanlder:
    '''
    input data handler
    for storing/sampling data, etc.
    '''
    def __init__(self,
                 log_filename,
                 min_count=5,
                 sub_sampling_t=1e-5,
                 neg_sampling_t=0.75,
                 batch_size=64,
                 neg_sample_count=5,
                 half_window_size=2,
                 read_data_method='memory'):
        """
        init function.
        :param log_filename: word2vec format -> item1 item2 ... itemk, each line a user log
        :param min_count: the same min_count as in word2vec
        :param sub_sampling_t: sub sampling rate, higher than that would be sub sampled using
                                the word2vec paper using:    p(w_i) = 1 - sqrt(sub_sampling / freq)
                                the word2vec code using:     p(w_i) = 1 - (sqrt(sub_sampling / freq) + sub_sampling / freq)
                                we use word2vec code subsampling method here.
        :param neg_sampling_t: the negative sampling t as in the word2vec. seems not shown in the paper, but implemented in
                                the code:
                                    p(w_i) = f(w_i) ** neg_sampling / sum(f(w_i) ** neg_sampling for w_i in vocab)
        :param read_data_method: method to read data:
                                    'memory': load all the sentence to memory, fast but cost memory.
                                    'file': load data from file, slower but save memory
        """
        assert read_data_method in ('memory', 'file')
        self.log_filename = log_filename
        self.min_count = min_count
        self.sub_sampling_t = sub_sampling_t
        self.neg_sampling_t = neg_sampling_t
        self.sentences = deque()
        self.read_data_method = read_data_method
        
        print('read dataset...')
        self.vocab, self.word2id, self, id2word, self.total_word_count, self.sentence_len = self.gen_vocab()
        print(f'got vocab {len(self.vocab)}, total_word_count {self.total_word_count}')
        
        print('gen sub sample table...')
        self.sub_sampling_table = self.get_subsample_table()
        
        print('gen negative sample table...')
        self.neg_sampling_table = self.gen_negative_sample_table()


        # for generating batch
        self.sentences_cursor = 0 
        
        self.batch_size = batch_size
        self.neg_sample_count = neg_sample_count
        self.half_window_size = half_window_size

    def gen_vocab(self):
        """
        from log file generate vocabulary
        :return: {item_id: freq}
        """
        assert self.log_filename != ''
        vocab_freq_dict = defaultdict(int)
        total_word_count = 0
        total_sent_count = 0

        with open(self.log_filename, encoding='utf-8') as f:
            for line in f:
                total_sent_count += 1
                item_ids = line.strip().split()
                if self.read_data_method == 'memory':
                    self.sentences.append(item_ids)
                for item_id in item_ids:
                    vocab_freq_dict[item_id] += 1
                    total_word_count += 1

        vocab, word2id, id2word = {}, {}, {}
        index = 0

        for item_id, freq in vocab_freq_dict.items():
            if freq < self.min_count:
                vocab[item_id] = freq
                word2id[item_id] = index
                id2word[index] = item_id
                index += 1

        return vocab, word2id, id2word, total_word_count, total_sent_count

    def gen_subsample_table(self):
        """
        sub sampling rate, higher than that would be sub sampled using
            the word2vec paper using:    p(w_i) = 1 - sqrt(sub_sampling / freq)
            the word2vec code using:     p(w_i) = 1 - (sqrt(sub_sampling / freq) + sub_sampling / freq)
        we use word2vec code sub sampling method here.
        :return: {word_id: sample_score}
        """
        def sub_sampling(_freq):
            return (self.sub_sampling_t / 1.0 / _freq) ** 0.5 + self.sub_sampling_t / 1.0 / _freq

        '''
        word freq count to word freq ratio
        각 단어의 빈도수를 빈도 비율로 변환
        단, 그 중에서 sub_sampling_t보다 이미 낮은 것은 스킵 (너무 작은건 학습시켜야 하기 때문)
        '''
        sub_sample_tbl = {item: freq / 1.0 / self.total_word_count
                          for item, freq in self.vocab.items()
                          if freq / 1.0 / self.total_word_count > self.sub_sampling_t}

        # freq to score
        sub_sample_tbl = {item: sub_sampling(_freq) for item, _freq in sub_sample_tbl.items()}

        # word to id
        sub_sample_tbl = {self.word2id[i]: j for i, j in sub_sample_tbl.items() if j < 1}

    def gen_negative_sample_table(self):
        """
        implemented same as word2vec c code.
        The way this selection is implemented in the C code is interesting. They have a large array with 100M elements
        (which they refer to as the unigram table). They fill this table with the index of each word in the vocabulary
        multiple times, and the number of times a word’s index appears in the table is given by P(wi) * table_size.
        
            p(w_i) = f(w_i) ** neg_sampling / sum(f(w_i) ** neg_sampling for w_i in vocab)
        
        :return:
        """
        sample_tbl_size = 1e8
        sample_tbl = []
        pow_freq = np.array(list(self.vocab.values())) ** self.neg_sampling_t
        pow_total_freq = sum(pow_freq)
        r = pow_freq / pow_total_freq
        count = np.round(r * sample_tbl_size)
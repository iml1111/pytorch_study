'''
https://github.com/jojonki/word2vec-pytorch
# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import SkipGram
import torch.optim as optim
import matplotlib.pyplot as plt
from data_handler import text, w2i, vocab_size 
from data_handler import create_skipgram_dataset

embed_size = 100 # 임베딩할 차원수 
learning_rate = 0.001
n_epoch = 30


def train_skipgram():
    losses = []
    loss_fn = nn.MSELoss()
    model = SkipGram(vocab_size, embed_size)
    print(model)
    print('vocab_size:', vocab_size)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    skipgram_train_data = create_skipgram_dataset(text)

    model.train() 
    for epoch in range(n_epoch):
        total_loss = .0

        for in_w, out_w, target in skipgram_train_data:
            in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
            out_w_var = Variable(torch.LongTensor([w2i[out_w]]))

            model.zero_grad()
            log_probs = model(in_w_var, out_w_var)
            loss = loss_fn(log_probs[0], Variable(torch.Tensor([target])))
            loss.backward()
            optimizer.step()
            total_loss += float(loss)
        losses.append(total_loss)
    return model, losses

def showPlot(points, title):
    plt.figure()
    fig, ax = plt.subplots()
    plt.plot(points)
    plt.show()


def test_skipgram(test_data, model):
    print('====Test SkipGram====')
    correct_cnt = 0
    model.eval()
    for in_w, out_w, target in test_data:
        in_w_var = Variable(torch.LongTensor([w2i[in_w]]))
        out_w_var = Variable(torch.LongTensor([w2i[out_w]]))

        model.zero_grad()
        log_probs = model(in_w_var, out_w_var)
        _, predicted = torch.max(log_probs.data, 1)
        predicted = predicted[0]
        if predicted == target:
            correct_cnt += 1

    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_cnt/len(test_data)*100, correct_cnt, len(test_data)))        


if __name__ == '__main__':
    sg_model, sg_losses = train_skipgram()
    test_data = create_skipgram_dataset(text)
    
    test_skipgram(test_data, sg_model)
    showPlot(sg_losses, 'SkipGram Losses')

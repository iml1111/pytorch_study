'''
https://github.com/jojonki/word2vec-pytorch
# see http://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html
'''
import torch
import torch.nn as nn
from torch.autograd import Variable
from model import CBOW
import torch.optim as optim
import matplotlib.pyplot as plt
from data_handler import text, w2i, i2w, vocab_size, CONTEXT_SIZE 
from data_handler import create_cbow_dataset

embed_size = 100 # 임베딩할 차원수 
learning_rate = 0.001
n_epoch = 30
hidden_size = 64


def train_cbow():    
    losses = []
    loss_fn = nn.NLLLoss()
    model = CBOW(vocab_size, embed_size, CONTEXT_SIZE, hidden_size)
    print(model)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    cbow_train = create_cbow_dataset(text)
    for epoch in range(n_epoch):
        total_loss = .0
        for context, target in cbow_train:
            ctx_idxs = [w2i[w] for w in context]
            ctx_var = Variable(torch.LongTensor(ctx_idxs))

            model.zero_grad()
            log_probs = model(ctx_var)

            loss = loss_fn(log_probs, Variable(torch.LongTensor([w2i[target]])))

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


# test
# You have to use other dataset for test, but in this case I use training data because this dataset is too small
def test_cbow(test_data, model):
    print('====Test CBOW===')
    correct_ct = 0
    for ctx, target in test_data:
        ctx_idxs = [w2i[w] for w in ctx]
        ctx_var = Variable(torch.LongTensor(ctx_idxs))

        model.zero_grad()
        log_probs = model(ctx_var)
        _, predicted = torch.max(log_probs.data, 1)
        predicted_word = i2w[int(predicted[0])]
        print('predicted:', predicted_word)
        print('label    :', target)
        if predicted_word == target:
            correct_ct += 1
            
    print('Accuracy: {:.1f}% ({:d}/{:d})'.format(correct_ct/len(test_data)*100, correct_ct, len(test_data)))


if __name__ == '__main__':
    cb_model, cb_losses = train_cbow()
    
    #test_data = create_cbow_dataset(text)
    #test_cbow(test_data, cb_model)
    #showPlot(cb_losses, 'CBOW Losses')
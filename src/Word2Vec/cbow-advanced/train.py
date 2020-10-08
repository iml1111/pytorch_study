import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from modules.trainer import Trainer
from modules.data_handler import CbowDataHandler
from models.cbow import CBOW

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', default="model.pth")
    p.add_argument('--train_fn', default='../../data/w2v_train.tsv')
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.9)
    p.add_argument('--batch_size', type=int, default=512)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--learning_rate', type=int, default=0.001)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--window_size', type=int, default=2)
    p.add_argument('--embd_size', type=int, default=100)
    p.add_argument('--hidden_size', type=int, default=64)
    return p.parse_args()


def main(config):
    if config.gpu_id < 0:
        print("Device: CPU")
    else:
        print("Device:", torch.cuda.get_device_name(config.gpu_id))

    print("Building Vocab...")
    data_handler = CbowDataHandler(
        file_name=config.train_fn,
        window_size=config.window_size,
        train_ratio=config.train_ratio,
        batch_size=config.batch_size,
    )
    print('|train| =', len(data_handler.train_loader.dataset),
          '|valid| =', len(data_handler.valid_loader.dataset))
    print('|vocab_size| =', data_handler.vocab_size)

    model = CBOW(
        vocab_size=data_handler.vocab_size,
        embd_size=config.embd_size,
        window_size=config.window_size,
        hidden_size=config.hidden_size,
    )
    # crit = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    crit =nn.NLLLoss()
    print(model)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, 
        data_handler.train_loader, data_handler.valid_loader)

    # Test
    test_data = ['맞교환', '백색', '합판', '이메일']
    ctx_idxs = [data_handler.w2i[w] for w in test_data]
    ctx_var = Variable(torch.LongTensor([ctx_idxs])).to(config.gpu_id)

    model.zero_grad()
    y = model(ctx_var)
    _, predicted = torch.max(y.data, 1)
    predicted_word = data_handler.i2w[int(predicted[0])]

    print('input:', test_data)
    print('predicted:', predicted_word)

if __name__ == '__main__':
    config = define_argparser()
    print(config)
    main(config)

'''
Namespace(batch_size=512, embd_size=100, gpu_id=0, hidden_size=64, learning_rate=0.001, model_fn='model.pth', n_epochs=20, train_fn='../../data/w2v_train.tsv', train_ratio=0.9, verbose=2, window_size=2)
Device: GeForce RTX 2080 Ti
|train| = 2118 |valid| = 236
|vocab_size| = 953
CBOW(
  (emb): Embedding(953, 100)
  (layers): Sequential(
    (0): Linear(in_features=400, out_features=64, bias=True)
    (1): ReLU()
    (2): Linear(in_features=64, out_features=953, bias=True)
    (3): LogSoftmax(dim=-1)
  )
)           
Validation - loss=6.1999e+00 accuracy=0.0932 best_loss=6.1549e+00
'''
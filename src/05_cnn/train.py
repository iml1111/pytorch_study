import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from mnist_classification.data_loader import get_loaders
from mnist_classification.trainer import Trainer
from mnist_classification.models.fc_model import FullyConnectedClassifier
from mnist_classification.models.cnn_model import ConvolutionalClassifier


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', default='model.pth')
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--model', type=str, default='cnn')
    return p.parse_args()


def get_model(config):
    if config.model == 'fc':
        return FullyConnectedClassifier(28 * 28, 10)
    return ConvolutionalClassifier(10)


def main(config):
    if config.gpu_id < 0:
        device = torch.device('cpu')
        print("Device: CPU")
    else:
        device = torch.device('cuda:%d' % config.gpu_id)
        print("Device:", torch.cuda.get_device_name(0))

    train_loader, valid_loader, _ = get_loaders(config)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))

    model = get_model(config).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    if config.verbose >= 2:
        print("Model:", model)
        print("Optimizer:", optimizer)
        print("Loss Func:", crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
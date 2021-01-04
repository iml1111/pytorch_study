import argparse
import pprint
import torch
from torch import optim
import torch.nn as nn
import torch_optimizer as custom_optim

from modules.data_loader import DataLoader
import modules.data_loader as data_loader
from modules.seq2seq import Seq2Seq
from modules.trainer import Trainer
from modules.trainer import IgniteEngine

def define_argparser(is_continue=False):
    p = argparse.ArgumentParser()

    if is_continue:
        p.add_argument(
            '--load_fn',
            required=True,
            help='Model file name to continue.'
        )

    p.add_argument(
        '--model_fn',
        default='./models/model.pth',
        help='Model file name to save. Additional information would be annotated to the file name.'
    )
    p.add_argument(
        '--train',
        default='./data/corpus.shuf.train.tok.bpe',
        help='Training set file name except the extention. (ex: train.en --> train)'
    )
    p.add_argument(
        '--valid',
        default='./data/corpus.shuf.valid.tok.bpe',
        help='Validation set file name except the extention. (ex: valid.en --> valid)'
    )
    p.add_argument(
        '--lang',
        default='enko',
        help='Set of extention represents language pair. (ex: en + ko --> enko)'
    )
    p.add_argument(
        '--gpu_id',
        type=int,
        default=0,
        help='GPU ID to train. Currently, GPU parallel is not supported. -1 for CPU. Default=%(default)s'
    )
    p.add_argument(
        '--off_autocast',
        action='store_true',
        help='Turn-off Automatic Mixed Precision (AMP), which speed-up training.',
    )

    p.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Mini batch size for gradient descent. Default=%(default)s'
    )
    p.add_argument(
        '--n_epochs',
        type=int,
        default=30,
        help='Number of epochs to train. Default=%(default)s'
    )
    p.add_argument(
        '--verbose',
        type=int,
        default=2,
        help='VERBOSE_SILENT, VERBOSE_EPOCH_WISE, VERBOSE_BATCH_WISE = 0, 1, 2. Default=%(default)s'
    )
    p.add_argument(
        '--init_epoch',
        required=is_continue,
        type=int,
        default=1,
        help='Set initial epoch number, which can be useful in continue training. Default=%(default)s'
    )

    p.add_argument(
        '--max_length',
        type=int,
        default=64,
        help='Maximum length of the training sequence. Default=%(default)s'
    )
    p.add_argument(
        '--dropout',
        type=float,
        default=.2,
        help='Dropout rate. Default=%(default)s'
    )
    p.add_argument(
        '--word_vec_size',
        type=int,
        default=512,
        help='Word embedding vector dimension. Default=%(default)s'
    )
    p.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='Hidden size of LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--n_layers',
        type=int,
        default=4,
        help='Number of layers in LSTM. Default=%(default)s'
    )
    p.add_argument(
        '--max_grad_norm',
        type=float,
        default=1e+8,
        help='Threshold for gradient clipping. Default=%(default)s'
    )
    p.add_argument(
        '--iteration_per_update',
        type=int,
        default=8,
        help='Number of feed-forward iterations for one parameter update. Default=%(default)s'
    )
    p.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Initial learning rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_step',
        type=int,
        default=0,
        help='Number of epochs for each learning rate decay. Default=%(default)s',
    )
    p.add_argument(
        '--lr_gamma',
        type=float,
        default=.5,
        help='Learning rate decay rate. Default=%(default)s',
    )
    p.add_argument(
        '--lr_decay_start',
        type=int,
        default=10,
        help='Learning rate decay start at. Default=%(default)s',
    )

    config = p.parse_args()

    return config


def get_model(input_size, output_size, config):
    model = Seq2Seq(
        input_size,
        config.word_vec_size,
        config.hidden_size,
        output_size,
        n_layers=config.n_layers,
        dropout_p=config.dropout
    )
    return model


def get_crit(output_size, pad_index):
    # PAD 토큰에 대하여 가중치를 주지 않도록 설정
    loss_weight = torch.ones(output_size)
    loss_weight[pad_index] = 0.
    crit = nn.NLLLoss(
        weight=loss_weight,
        reduction='sum'
    )
    return crit


def get_optimizer(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config.lr) 
    return optimizer


def get_scheduler(optimizer, config):
    '''
    # learing_rate 스케쥴러
    학습도중, learning_rate를 조정하기 위한 역할
    ex) epoch 9까지 lr:1로 하다가, 10부터 0.5, 0.25, 0.125 식으로 낮춰라 등
    아래 코드에서는 lr_decay_start번째부터 config.lr_step를 곱하며 낮춰감
    # 그러나 이부분은 사용하지 않음 X
    '''
    if config.lr_step > 0:
        lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=[i for i in range(
                max(0, config.lr_decay_start - 1),
                (config.init_epoch - 1) + config.n_epochs,
                config.lr_step
            )],
            gamma=config.lr_gamma,
            last_epoch=config.init_epoch - 1 if config.init_epoch > 1 else -1,
        )
    else:
        lr_scheduler = None

    return lr_scheduler


def main(config, model_weight=None, opt_weight=None):
    def print_config(config):
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(vars(config))
    print_config(config)

    loader = DataLoader(
        config.train,
        config.valid,
        (config.lang[:2], config.lang[-2:]),
        batch_size=config.batch_size,
        device=-1,
        max_length=config.max_length
    )

    input_size, output_size = len(loader.src.vocab), len(loader.tgt.vocab)
    model = get_model(input_size, output_size, config)
    crit = get_crit(output_size, data_loader.PAD)

    if model_weight:
        model.load_state_dict(model_weight)

    if config.gpu_id >= 0:
        model.cuda(config.gpu_id)
        crit.cuda(config.gpu_id)

    optimizer = get_optimizer(model, config)

    if opt_weight and (config.use_adam or config.use_radam):
        optimizer.load_state_dict(opt_weight)

    lr_scheduler = get_scheduler(optimizer, config)

    if config.verbose >= 2:
        print(model)
        print(crit)
        print(optimizer)

    trainer = Trainer(IgniteEngine, config)
    trainer.train(
        model,
        crit,
        optimizer,
        train_loader=loader.train_iter,
        valid_loader=loader.valid_iter,
        src_vocab=loader.src.vocab,
        tgt_vocab=loader.tgt.vocab,
        n_epochs=config.n_epochs,
        lr_scheduler=lr_scheduler
    )


if __name__ == '__main__':
    config = define_argparser()
    main(config)

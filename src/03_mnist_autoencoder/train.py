import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import Autoencoder
from trainer import Trainer
from utils import load_mnist

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_fn", default="model.pth")
    p.add_argument('--gpu_id', 
                   type=int,
                   default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--train_ratio', type=float, default=.8)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)
    p.add_argument('--btl_size', type=int, default=2)

    return p.parse_args()

def main(config):
    # 연산 디바이스 설정
    if config.gpu_id < 0:
        print("Device: CPU")
        device = torch.device('cpu')
    else:
        print("Device:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:%d' % config.gpu_id)

    #학습데이터 호출
    train_x, train_y = load_mnist(is_train=True, flatten=True)

    # train_ratio에 따라 학습, 검증 데이터 분할
    train_cnt = int(train_x.size(0) * config.train_ratio)
    valid_cnt = train_x.size(0) - train_cnt
    indices = torch.randperm(train_x.size(0))
    train_x, valid_x = torch.index_select(
        train_x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)
    train_y, valid_y = torch.index_select(
        train_y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    print("Train:", train_x.shape, train_y.shape)
    print("Valid:", valid_x.shape, valid_y.shape)

    # 모델 객체 호출 및 학습
    model = Autoencoder(btl_size=config.btl_size).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.MSELoss()

    trainer = Trainer(model, optimizer, crit)
    trainer.train((train_x, train_x), 
                  (valid_x, valid_x), 
                  config)

    #모델 저장
    torch.save(
        {
            'model': trainer.model.state_dict(),
            'config':config
        },
        config.model_fn)


if __name__ == '__main__':
    config = define_argparser()
    main(config)
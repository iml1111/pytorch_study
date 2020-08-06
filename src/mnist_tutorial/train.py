import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import ImageClassifier
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

    return p.parse_args()


def main(config):
    # 연산 디바이스 설정
    if config.gpu_id < 0:
        print("Device: CPU")
        device = torch.device('cpu')
    else:
        print("Device:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:%d' % config.gpu_id)

    # 입력값에 따라 학습, 검증 데이터 분할
    x, y = load_mnist(is_train=True, flatten=True)
    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    # 각 데이터셋 셔플링
    indices = torch.randperm(x.size(0))
    x = torch.index_select(
        x,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)
    y = torch.index_select(
        y,
        dim=0,
        index=indices
    ).to(device).split([train_cnt, valid_cnt], dim=0)

    print("Train:", x[0].shape, y[0].shape)
    print("Valid:", x[1].shape, y[1].shape)

    # 모델 선언 및 구조 결정
    model = ImageClassifier(28*28, 10).to(device)
    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    # 학습 수행
    trainer = Trainer(model, optimizer, crit)
    trainer.train(train_data = (x[0], y[0]),
                  valid_data = (x[1], y[1]),
                  config = config)
    
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

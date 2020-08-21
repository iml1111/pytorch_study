import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from classification.data_loader import get_loader
from classification.trainer import Trainer
from classification.model_loader import get_model


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', default="model.pth")
    p.add_argument('--gpu_id', type=int, default=0 if torch.cuda.is_available() else -1)

    p.add_argument('--train_ratio', type=float, default=.6)
    p.add_argument('--valid_ratio', type=float, default=.2)
    p.add_argument('--test_ratio', type=float, default=.2)

    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=20)
    p.add_argument('--verbose', type=int, default=2)

    p.add_argument('--model_name', type=str, default='resnet')
    p.add_argument('--dataset_name', type=str, default='catdog')
    p.add_argument('--n_classes', type=int, default=2) # 클래시피케이션 수
    p.add_argument('--freeze', action='store_true') 
    p.add_argument('--use_pretrained', action='store_true')
    '''
    rand_init: 아키텍쳐만 같고 완전히 새로 학습 : 아무것도 안쓸떄
    pretrained: 모델 파라미터를 가져오되 모두 재학습: 프리트레인만 true
    freezed: 모델 파라미터를 가져오고 마지막 레이어를 제외한 나머지 학습 X: 둘다 true

    freezed pretrain의 경우, 굉장히 빠르게 어큐러시가 수렴하게 됨
    '''
    return p.parse_args()


def main(config):
    if config.gpu_id < 0:
        device = torch.device('cpu')
        print("Device: CPU")
    else:
        device = torch.device('cuda:%d' % config.gpu_id)
        print("Device:", torch.cuda.get_device_name(0))

    model, input_size = get_model(config)
    model = model.to(device)

    train_loader, valid_loader, test_loader = get_loader(config, input_size)

    print("Train:", len(train_loader.dataset))
    print("Valid:", len(valid_loader.dataset))
    print("Test:", len(test_loader.dataset))

    optimizer = optim.Adam(model.parameters())
    crit = nn.CrossEntropyLoss()

    if config.verbose >= 2:
        print(model)
        print(optimizer)
        print(crit)

    trainer = Trainer(config)
    trainer.train(model, crit, optimizer, train_loader, valid_loader)

    '''
    # use_pretrained / freeze
    Epoch 20 - |param|=8.80e+01 |g_param|=4.59e-01 loss=9.8717e-02 accuracy=0.9552
    Validation - loss=9.3166e-02 accuracy=0.9645 best_loss=8.3037e-02
    
    # only pretrained
    Epoch 20 - |param|=1.26e+02 |g_param|=5.50e-01 loss=1.1496e-01 accuracy=0.9497
    Validation - loss=1.5151e-01 accuracy=0.9366 best_loss=1.5098e-01

    # rand init
    Epoch 20 - |param|=1.66e+02 |g_param|=1.01e+00 loss=2.8314e-01 accuracy=0.8696
    Validation - loss=2.9703e-01 accuracy=0.8676 best_loss=2.8204e-01
    '''


if __name__ == '__main__':
    config = define_argparser()
    main(config)
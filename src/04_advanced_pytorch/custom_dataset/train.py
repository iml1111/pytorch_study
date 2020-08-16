import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.datasets import load_breast_cancer
from custom_dataset import CustomDataset
from trainer import Trainer
from model import CancerClassifier
import matplotlib.pyplot as plt
import torch.nn.functional as F

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument("--model_fn", default="model.pth")
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--n_epochs', type=int, default=1000)
    # p.add_argument('--print_interval', type=int, default=500)
    p.add_argument('--early_stop', type=int, default=100)
    p.add_argument('--gpu_id', 
                   type=int,
                   default=0 if torch.cuda.is_available() else -1)
    return p.parse_args()


def main(config):
    # 연산 디바이스 설정
    if config.gpu_id < 0:
        print("Device: CPU")
        device = torch.device('cpu')
    else:
        print("Device:", torch.cuda.get_device_name(0))
        device = torch.device('cuda:%d' % config.gpu_id)

    # 유방암 데이터 가져오기
    cancer_data = load_breast_cancer()
    df = pd.DataFrame(cancer_data.data, columns=cancer_data.feature_names)
    df['class'] = cancer_data.target
    data = torch.from_numpy(df.values).float()
    x = data[:, :30]
    y = data[:, -1:]

    # 학습, 검증, 테스트 데이터 나누고 섞기
    ratios = [.6, .2, .2]
    train_cnt = int(x.size(0) * ratios[0])
    valid_cnt = int(x.size(0) * ratios[1])
    test_cnt = x.size(0) - train_cnt - valid_cnt
    cnts = [train_cnt, valid_cnt, test_cnt]
    indices = torch.randperm(x.size(0))
    x = torch.index_select(x, dim=0, index=indices).to(device)
    y = torch.index_select(y, dim=0, index=indices).to(device)
    x = x.split(cnts, dim=0)
    y = y.split(cnts, dim=0)

    # 토치 데이터셋, 로더를 이용하여 데이터 객체화
    train_loader = DataLoader(
        dataset=CustomDataset(x[0], y[0]),
        batch_size=config.batch_size,
        shuffle=True)
    valid_loader = DataLoader(
        dataset=CustomDataset(x[1], y[1]),
        batch_size=config.batch_size,
        shuffle=False)
    test_loader = DataLoader(
        dataset=CustomDataset(x[2], y[2]),
        batch_size=config.batch_size,
        shuffle=False)
    print("Train %d / Valid %d / Test %d samples." % (
        len(train_loader.dataset),
        len(valid_loader.dataset),
        len(test_loader.dataset),
    ))

    # 모델 선언 및 구조 결정
    model = CancerClassifier(x[0].size(-1), y[0].size(-1)).to(device)
    optimizer = optim.Adam(model.parameters())

    # 학습 수행
    trainer = Trainer(model, optimizer, train_loader, valid_loader)
    trainer.train(config)

    # Loss history
    plot_from = 2
    plt.figure(figsize=(20, 10))
    plt.grid(True)
    plt.title("Train / Valid Loss History")
    plt.plot(
        range(plot_from, len(trainer.train_history)), 
        trainer.train_history[plot_from:],
        range(plot_from, len(trainer.valid_history)), 
        trainer.valid_history[plot_from:],
    )
    plt.yscale('log')
    plt.show()

    # Evaluate
    test_loss = 0
    y_hat = []
    model.eval()
    with torch.no_grad():
        for x_i, y_i in test_loader:
            y_hat_i = model(x_i)
            loss = F.binary_cross_entropy(y_hat_i, y_i)
            test_loss += float(loss) # Gradient is already detached.
            y_hat += [y_hat_i]
    test_loss = test_loss / len(test_loader)
    y_hat = torch.cat(y_hat, dim=0)
    print("Test loss: %.4e" % test_loss)
    correct_cnt = (y[2] == (y_hat > .5)).sum()
    total_cnt = float(y[2].size(0))
    print('Test Accuracy: %.4f' % (correct_cnt / total_cnt))


if __name__ == '__main__':
    config = define_argparser()
    main(config)
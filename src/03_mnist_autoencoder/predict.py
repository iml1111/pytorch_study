import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist
import torch
from model import Autoencoder

# 연산 디바이스 결정
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device("cpu")

# 테스트 데이터 호출
test_x, test_y = load_mnist(is_train=False, flatten=True)
test_x, test_y = test_x.to(device), test_y.to(device)

# 저장된 모델 호출
def load(fn, device):
    d = torch.load(fn, map_location=device)
    return d['model']

model_fn = "./model.pth"
model = Autoencoder(btl_size=2).to(device)
model.load_state_dict(load(model_fn, device))


def show_image(x):
    if x.dim() == 1:
        x = x.view(int(x.size(0) ** .5), -1)
    plt.imshow(x, cmap='gray')
    plt.show()


# 오토인코더를 돌려 원본 이미지와 비교해보기
def get_random_num():
    model.eval()
    with torch.no_grad():
        index = int(random.random() * test_x.size(0))
        pred_y = model(test_x[index].view(1, -1))
        pred_y = model(test_x[index].view(1, -1)).squeeze()
        show_image(test_x[index])
        show_image(pred_y)


# 2차원 잠재 공간에서 분포 그래프 출력하기
def get_latent_space(btl_size=2):
    if btl_size == 2:
        color_map = [
            'brown', 'red', 'orange', 'yellow', 'green',
            'blue', 'navy', 'purple', 'gray', 'black',
        ]

    plt.figure(figsize=(20, 10))
    with torch.no_grad():
        latents = model.encoder(test_x[:1000])

        for i in range(10):
            target_latents = latents[test_y[:1000] == i]
            # target_y = test_y[:1000][test_y[:1000] == i]
            plt.scatter(target_latents[:, 0],
                        target_latents[:, 1],
                        marker='o',
                        color=color_map[i],
                        label=i)
        plt.legend()
        plt.grid(axis='both')
        plt.show()


# 2차원 잠재 공간 내에 각 이미지 시각화
def get_latent_view(btl_size=2):
    if btl_size == 2:
        min_range, max_range = -2., 2.
        n = 20
        step = (max_range - min_range) / float(n)
        
        with torch.no_grad():
            lines = []

            for v1 in np.arange(min_range, max_range, step):
                z = torch.stack([
                    torch.FloatTensor([v1] * n),
                    torch.FloatTensor([v2 for v2 in np.arange(min_range,
                                                              max_range, step)]),
                ], dim=-1)
                
                line = torch.clamp(model.decoder(z).view(n, 28, 28), 0, 1)
                line = torch.cat([line[i] for i in range(n - 1, 0, -1)], dim=0)
                lines += [line]
                
            lines = torch.cat(lines, dim=-1)
            plt.figure(figsize=(20, 20))
            show_image(lines)

import random
import numpy as np
import matplotlib.pyplot as plt
from utils import load_mnist

# 연산 디바이스 결정
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 테스트 데이터 호출
test_x, test_y = load_mnist(is_train=False, flatten=True)
test_x, test_y = x.to(device), y.to(device)

# 저장된 모델 호출
model_fn = "./model.pth"
model = Autoencoder(btl_size=5).to(device)
model.load_state_dict(load(model_fn, device))


def show_image(x):
    if x.dim() == 1:
        x = x.view(int(x.size(0) ** .5), -1)
    plt.imshow(x, cmap='gray')
    plt.show()


def get_random_num():
	index = int(random.random() *)
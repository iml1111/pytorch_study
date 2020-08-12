import torch.nn as nn


class Autoencoder(nn.Module):
    ''' 오토인코더 신경망 구조'''
    def __init__(self, btl_size=2):
        self.btl_size = btl_size
        super().__init__()

        # 인코더, 디코더의 각 끝은 정규화 함수나 활성화 함수를 사용하지 않음
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),

            nn.Linear(300, 150),
            nn.ReLU(),
            nn.BatchNorm1d(150),

            nn.Linear(150, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),

            nn.Linear(100, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, 25),
            nn.ReLU(),
            nn.BatchNorm1d(25),

            nn.Linear(25, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),

            nn.Linear(10, btl_size),
        )

        self.decoder = nn.Sequential(
            nn.Linear(btl_size, 10),
            nn.ReLU(),
            nn.BatchNorm1d(10),

            nn.Linear(10, 25),
            nn.ReLU(),
            nn.BatchNorm1d(25),

            nn.Linear(25, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, 100),
            nn.ReLU(),
            nn.BatchNorm1d(100),

            nn.Linear(100, 150),
            nn.ReLU(),
            nn.BatchNorm1d(150),

            nn.Linear(150, 300),
            nn.ReLU(),
            nn.BatchNorm1d(300),

            nn.Linear(300, 500),
            nn.ReLU(),
            nn.BatchNorm1d(500),

            nn.Linear(500, 28 * 28),
        )

    def forward(self, x):
        z = self.encoder(x)
        y = self.decoder(z)
        return y
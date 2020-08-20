import torch.nn as nn


class ConvolutionBlock(nn.Module):
    '''Convolution Block 클래스'''
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, (3, 3), padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),

            nn.Conv2d(out_channels, out_channels, (3, 3), stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        '''
        x = (batch_size, in_channels, h, w)
        y = (batch_size, out_channels, h', w') 
        대충 h, w는 공식상 절반정도로 줄어들 것임
        '''
        return self.layers(x)


class ConvolutionalClassifier(nn.Module):
    '''CNN 신경망 클래스'''
    def __init__(self, output_size):
        '''
        input_size가 정해져 있지 않음.
        단, input_x의 shape는 지켜야 함
        '''
        self.output_size = output_size
        super().__init__()

        '''
        h, w는 대충 2배로 줄어드는거같지만 실제로는 공식대로 적용됨
        채널 또한 대충 2배로 늘어나는거 처럼 보이지만 반드시 해야하는건 아님 (관례가 그럼)
        '''
        self.blocks = nn.Sequential( # |x| = (n, 1, 28, 28)
            ConvolutionBlock(1, 32), # (n, 32, 14, 14)
            ConvolutionBlock(32, 64), # (n, 64, 7, 7) 
            ConvolutionBlock(64, 128), # (n, 128, 4, 4) 
            ConvolutionBlock(128, 256), # (n, 256, 2, 2)
            ConvolutionBlock(256, 512), # (n, 512, 1, 1)
        )
        self.layers = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.BatchNorm1d(50),
            nn.Linear(50, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        '''
        순전파에 앞서, x의 차원수를 체크해야 함.
        1. x는 2차원 이상이여야함 (n, 784) => X
        2. x가 3차원일 경우, 채널을 가진 4차원 형태로 변환 
        (n, 28, 28) => (n, 1, 28, 28)
        '''
        assert x.dim() > 2
        if x.dim() == 3:
            x = x.view(-1, 1, x.size(-2), x.size(-1))

        '''
        x = (batch_size, 1, h, w)
        z = (batch_size, 512, 1, 1) -> (batch_size, 512)
        y = (batch_size, output_size)
        '''
        z = self.blocks(x)
        y = self.layers(z.squeeze())
        return y
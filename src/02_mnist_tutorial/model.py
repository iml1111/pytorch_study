import torch.nn as nn

class ImageClassifier(nn.Module):
    '''이미지 분류기 신경망 클래스'''

    def __init__(self,
                 input_size,
                 output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 500), # Dense Layer
            nn.LeakyReLU(),             # Activation Func
            nn.BatchNorm1d(500),        # Normalization
            
            nn.Linear(500, 400),
            nn.LeakyReLU(),
            nn.BatchNorm1d(400),

            nn.Linear(400, 300),
            nn.LeakyReLU(),
            nn.BatchNorm1d(300),

            nn.Linear(300, 200),
            nn.LeakyReLU(),
            nn.BatchNorm1d(200),

            nn.Linear(200, 100),
            nn.LeakyReLU(),
            nn.BatchNorm1d(100),

            nn.Linear(100, 50),
            nn.LeakyReLU(),
            nn.BatchNorm1d(50),

            nn.Linear(50, output_size),
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        '''
        input(x) : (batch_size, input_size)
        output(y) : (batch_size, output_size)
        '''
        return self.layers(x)
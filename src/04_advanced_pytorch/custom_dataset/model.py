import torch.nn as nn

class CancerClassifier(nn.Module):
    '''유방암 분류기 신경망 클래스'''

    def __init__(self,
                 input_size,
                 output_size):
        self.input_size = input_size
        self.output_size = output_size
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_size, 22),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(22),

            nn.Linear(22, 15),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(15),
            
            nn.Linear(15, 10),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(10),

            nn.Linear(10, 5),
            nn.LeakyReLU(),
           #nn.BatchNorm1d(5),
            
            nn.Linear(5, 4),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(4),
            
            nn.Linear(4, 3),
            nn.LeakyReLU(),
            #nn.BatchNorm1d(3),
            
            nn.Linear(3, output_size),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.layers(x)

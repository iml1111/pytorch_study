from copy import deepcopy
import numpy as np
import torch
import torch.nn.functional as F

class Trainer():
    '''모델 학습기 클래스'''

    def __init__(self, model, optim, train_loader, valid_loader):
        self.model = model
        self.optim = optim
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.train_history = []
        self.valid_history = []
        super().__init__()

    def _train(self, config):
        '''
        한 에포크당 학습 과정
        '''

        # 학습 모드 On
        # 이걸 해야 학습이 모델에 반영됨
        self.model.train()

        total_loss = 0

        for i, (x_i, y_i) in enumerate(self.train_loader):
            pred_y_i = self.model(x_i)
            loss = F.binary_cross_entropy(pred_y_i, y_i)

            # 그래디언트 초기화 및 역전파 진행
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            # 효율적인 메모리 연산을 위해 float로 캐스팅
            total_loss += float(loss)

        self.train_history += [total_loss / len(self.train_loader)]
        return total_loss / len(self.train_loader)

    def _validate(self, config):
        '''
        한 에포크당 검증 과정
        '''

        # 검증 모드 On
        # 이걸 해야 검증 결과가 모델에 반영이 안됨
        self.model.eval()

        # 해당 구문 내에서의 모든 grad 변화를 반영하지 않음
        with torch.no_grad():
            total_loss = 0
            for i, (x_i, y_i) in enumerate(self.valid_loader):
                pred_y_i = self.model(x_i)
                loss = F.binary_cross_entropy(pred_y_i, y_i)
                total_loss += float(loss)

            self.valid_history += [total_loss / len(self.valid_loader)]
            return total_loss / len(self.valid_loader)

    def train(self, config):
        '''
        모델 학습 함수
        '''
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(config)
            valid_loss = self._validate(config)

            # 검증 loss 값이 현재의 최소 loss 값보다 낮을 경우, 갱신
            if valid_loss <= lowest_loss:
                lowest_loss = valid_loss
                best_model = deepcopy(self.model.state_dict())

            print("Epoch(%d/%d): train_loss=%.4e  valid_loss=%.4e  lowest_loss=%.4e" % (
                epoch_index + 1,
                config.n_epochs,
                train_loss,
                valid_loss,
                lowest_loss,
            ))

        # 베스트 모델 객체 저장
        self.model.load_state_dict(best_model)
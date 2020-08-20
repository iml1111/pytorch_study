from copy import deepcopy
import numpy as np
import torch

class Trainer():
    '''모델 학습기 클래스'''

    def __init__(self, model, optim, crit):
        self.model = model
        self.optim = optim
        self.crit = crit
        super().__init__()

    def _train(self, x, y, config):
        '''
        한 에포크당 학습 과정
        '''

        # 학습 모드 On
        # 이걸 해야 학습이 모델에 반영됨
        self.model.train()

        # 학습 데이터 셔플
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)

        # 배치 사이즈에 맞춰서 자르기
        x = x.split(config.batch_size, dim=0)
        y = y.split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            pred_y_i = self.model(x_i)
            # squeeze : (N, 1) -> (N,)로 변환
            # print("-----")
            # print(pred_y_i.size())
            # print(y_i.squeeze().size())
            # print("-----")
            loss = self.crit(pred_y_i, y_i.squeeze())

            # 그래디언트 초기화 및 역전파 진행
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (
                    i + 1,
                    len(x),
                    float(loss)
                ))
            # 효율적인 메모리 연산을 위해 float로 캐스팅
            total_loss += float(loss)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        '''
        한 에포크당 검증 과정
        '''

        # 검증 모드 On
        # 이걸 해야 검증 결과가 모델에 반영이 안됨
        self.model.eval()

        # 해당 구문 내에서의 모든 grad 변화를 반영하지 않음
        with torch.no_grad():
            indices = torch.randperm(x.size(0), device=x.device)
            x = torch.index_select(x, dim=0, index=indices)
            y = torch.index_select(y, dim=0, index=indices)
            x = x.split(config.batch_size, dim=0)
            y = y.split(config.batch_size, dim=0)

            total_loss = 0

            for i, (x_i, y_i) in enumerate(zip(x, y)):
                pred_y_i = self.model(x_i)
                loss = self.crit(pred_y_i, y_i.squeeze())

                if config.verbose >= 2:
                    print("Train Iteration(%d/%d): loss=%.4e" % (
                        i + 1,
                        len(x),
                        float(loss)
                    ))

                total_loss += float(loss)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        '''
        모델 학습 함수
        '''
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

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
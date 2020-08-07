from copy import deepcopy
import numpy as np
import torch

class Trainer():
    '''학습기 클래스'''
    def __init__(self, model, optim, crit):
        self.model = model
        self.optim = optim
        self.crit = crit
        super().__init__()

    def _train(self, x, y, config):
        # Turn on Train mode
        self.model.train()

        #Dataset Shuffle
        indices = torch.randperm(x.size(0), device=x.device)
        x = torch.index_select(x, dim=0, index=indices)
        y = torch.index_select(y, dim=0, index=indices)
        x = x.split(config.batch_size, dim=0)
        y = y.split(config.batch_size, dim=0)

        total_loss = 0

        for i, (x_i, y_i) in enumerate(zip(x, y)):
            pred_y_i = self.model(x_i)
            loss = self.crit(pred_y_i, y_i.squeeze())

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()

            if config.verbose >= 2:
                print("Train Iteration(%d/%d): loss=%.4e" % (
                                    i + 1, len(x), float(loss)))

            total_loss += float(loss)

        return total_loss / len(x)

    def _validate(self, x, y, config):
        # Turn on Valid mode
        self.model.eval()

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
                                        i + 1, len(x), float(loss)))

                total_loss += float(loss)

            return total_loss / len(x)

    def train(self, train_data, valid_data, config):
        lowest_loss = np.inf
        best_model = None

        for epoch_index in range(config.n_epochs):
            train_loss = self._train(train_data[0], train_data[1], config)
            valid_loss = self._validate(valid_data[0], valid_data[1], config)

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

        self.model.load_state_dict(best_model)

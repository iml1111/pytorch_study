from copy import deepcopy
import numpy as np
import torch
from ignite.engine import Engine
from ignite.engine import Events
from ignite.metrics import RunningAverage
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from utils import get_grad_norm, get_parameter_norm

VERBOSE_SILENT = 0
VERBOSE_EPOCH_WISE = 1
VERBOSE_BATCH_WISE = 2


class IgniteEngine(Engine):

    def __init__(self, func, model, crit, optimizer, config):
        self.model = model
        self.crit = crit
        self.optimizer = optimizer
        self.config = config
        self.best_loss = np.inf
        self.best_model = None
        self.device = next(model.parameters()).device
        # 인자로 받은 함수 실행
        super().__init__(func) 

    @staticmethod
    def train(engine, mini_batch):
        engine.model.train()
        engine.optimizer.zero_grad()
        x, y = mini_batch
        # 학습하는 중간에 데이터를 GPU로 전송해줌
        x, y = x.to(engine.device), y.to(engine.device)

        pred_y = engine.model(x)

        loss = engine.crit(pred_y, y)
        loss.backward()

        # 회귀 예측의 경우, 어큐러시 측정이 불가능하므로 스킵해야 함
        # Y가 정수로 나온다면 Classification일 것이라 가정
        if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
            accuracy = (torch.argmax(pred_y, dim=-1) == y).sum() / float(y.size(0))
        else:
            accuracy = 0

        # 파라미터 놈: 학습이 진행될수록 점진적으로 커짐
        p_norm = float(get_parameter_norm(engine.model.parameters()))
        # 그래드 놈: 값이 크면 클수록 많이 배우고 있다는 뜻
        # 처음 시작할 때는 많이 배우기 때문에 값이 클 것임 (=기울기가 가파름)
        # 진행됨에 따라서 서서히 줄어드는 게 보편적 (물론 아닐 수도 있음)
        # 적었다 커졌다 날뛰거나, Nan으로 Loss 자체가 날라가서 학습이 실패할수도 있음
        # 즉, 학습의 안정성을 보장함
        g_norm = float(get_grad_norm(engine.model.parameters()))

        engine.optimizer.step()

        return {
            'loss': float(loss),
            'accuracy': float(accuracy),
            '|param|': p_norm,
            '|g_param|': g_norm,
        }

    @staticmethod
    def validate(engine, mini_batch):
        engine.model.eval()

        with torch.no_grad():
            x, y = mini_batch
            # 학습하는 중간에 데이터를 GPU로 전송해줌
            x, y = x.to(engine.device), y.to(engine.device)

            pred_y = engine.model(x)

            loss = engine.crit(pred_y, y)

            if isinstance(y, torch.LongTensor) or isinstance(y, torch.cuda.LongTensor):
                accuracy = (torch.argmax(pred_y, dim=-1) == y).sum() / float(y.size(0))
            else:
                accuracy = 0

            return {
                'loss': float(loss),
                'accuracy': float(accuracy)
            }

    @staticmethod
    def attach(train_engine, validation_engine, verbose=VERBOSE_BATCH_WISE):
        '''현재 상황 보고 및 출력 함수'''
        def attach_running_average(engine, metric_name):
            RunningAverage(output_transform=lambda x: x[metric_name]).attach(
                engine,
                metric_name
            )
        '''
        Train Attach Process
        '''
        training_metric_names = ['loss', 'accuracy', '|param|', '|g_param|']

        for metric_name in training_metric_names:
            attach_running_average(train_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(train_engine, training_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @train_engine.on(Events.EPOCH_COMPLETED)
            def print_train_tag(engine):
                print('Epoch {} - |param|={:.2e} |g_param|={:.2e} loss={:.4e} accuracy={:.4f}'.format(
                    engine.state.epoch,
                    engine.state.metrics['|param|'],
                    engine.state.metrics['|g_param|'],
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                ))
        '''
        Validate Attach Process
        '''
        validation_metric_names = ['loss', 'accuracy']

        for metric_name in validation_metric_names:
            attach_running_average(validation_engine, metric_name)

        if verbose >= VERBOSE_BATCH_WISE:
            pbar = ProgressBar(bar_format=None, ncols=120)
            pbar.attach(validation_engine, validation_metric_names)

        if verbose >= VERBOSE_EPOCH_WISE:
            @validation_engine.on(Events.EPOCH_COMPLETED)
            def print_valid_logs(engine):
                print('Validation - loss={:.4e} accuracy={:.4f} best_loss={:.4e}'.format(
                    engine.state.metrics['loss'],
                    engine.state.metrics['accuracy'],
                    engine.best_loss,
                ))

    @staticmethod
    def check_best(engine):
        loss = float(engine.state.metrics['loss'])
        if loss <= engine.best_loss:
            engine.best_loss = loss
            engine.best_model = deepcopy(engine.model.state_dict())

    @staticmethod
    def save_model(engine, train_engine, config, **kwargs):
        torch.save(
            {
                'model':engine.best_model,
                'config': config,
                **kwargs
            }, config.model_fn
        )


class Trainer:

    def __init__(self, config):
        self.config = config

    def train(self, model, crit, optimizer,
              train_loader, valid_loader):
        train_engine = IgniteEngine(
            IgniteEngine.train,
            model, crit, optimizer, self.config)
        validation_engine = IgniteEngine(
            IgniteEngine.validate,
            model, crit, optimizer, self.config)

        IgniteEngine.attach(
            train_engine,
            validation_engine,
            verbose=self.config.verbose
        )

        def run_validation(engine, validation_engine, valid_loader):
            validation_engine.run(valid_loader, max_epochs=1)

        train_engine.add_event_handler(
            Events.EPOCH_COMPLETED, # event
            run_validation, # function
            validation_engine, # arguments
            valid_loader,
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            IgniteEngine.check_best,
        )
        validation_engine.add_event_handler(
            Events.EPOCH_COMPLETED,
            IgniteEngine.save_model,
            train_engine, self.config,
        )

        train_engine.run(
            train_loader,
            max_epochs=self.config.n_epochs,
        )

        model.load_state_dict(validation_engine.best_model)

        return model
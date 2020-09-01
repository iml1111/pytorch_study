import sys
import argparse
import torch
from torchtext import data
from models.rnn import RNNClassifier
from models.cnn import CNNClassifier


def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', default='model.pth')
    p.add_argument('--gpu_id', type=int, default=-1)
    p.add_argument('--batch_size', type=int, default=256)
    # 정답일 확률이 높은  TOP N을 출력
    p.add_argument('--top_k', type=int, default=1)
    p.add_argument('--max_length', type=int, default=256)
    
    p.add_argument('--drop_rnn', action='store_true')
    p.add_argument('--drop_cnn', action='store_true')
    return p.parse_args()


def read_text(max_length=256):
    lines = []
    for line in sys.stdin:
        if line.strip() != "":
            lines += [line.strip().split(' ')[:max_length]]
    return lines


def define_field():
    '''
    굳이 하나를 예측하기 위해 DataLoader를 호출할 필요는 없음
    그러므로 더미 필드를 만들어 신경망에 똑같은 포맷을 넣어줌
    '''
    return (
        data.Field(
            use_vocab=True,
            batch_first=True,
            include_lengths=False),
        data.Field(
            sequential=False,
            use_vocab=True,
            unk_token=None)
    )


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id)

    train_config = saved_data['config']
    rnn_best = saved_data['rnn']
    cnn_best = saved_data['cnn']
    vocab = saved_data['vocab']
    classes = saved_data['classes']

    vocab_size = len(vocab)
    n_classes = len(classes)

    # 더미 필드 선언 및 기존 단어 사전를 덮어씌움
    text_field, label_field = define_field()
    text_field.vocab = vocab
    label_field.vocab = classes

    lines = read_text(max_length=config.max_length)

    with torch.no_grad():
        # 테스트 데이터 텐서화
        x = text_field.numericalize(
            text_field.pad(lines),
            device='cuda:%d' % config.gpu_id if config.gpu_id >= 0 else 'cpu'
        )
        ensemble = []
        if rnn_best is not None and not config.drop_rnn:
            model = RNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                hidden_size=train_config.hidden_size,
                n_classes=n_classes,
                n_layers=train_config.n_layers,
                dropout_p=train_config.dropout)
            model.load_state_dict(rnn_best)
            ensemble += [model]
        if cnn_best is not None and not config.drop_cnn:
            model = CNNClassifier(
                input_size=vocab_size,
                word_vec_size=train_config.word_vec_size,
                n_classes=n_classes,
                use_batch_norm=train_config.use_batch_norm,
                dropout_p=train_config.dropout,
                window_sizes=train_config.window_sizes,
                n_filters=train_config.n_filters,
            )
            model.load_state_dict(cnn_best)
            ensemble += [model]

        pred_y_list = []
        
        for model in ensemble:
            if config.gpu_id >= 0:
                model.cuda(config.gpu_id)
            model.eval()

            pred_y = []
            for idx in range(0, len(lines), config.batch_size):
                pred_y += [model(x[idx:idx + config.batch_size]).cpu()]

            # y_hat = (len(lines), n_classes)
            # 리스트로 합쳐놓은 걸 그대로 이어서 실제로 텐서화
            pred_y = torch.cat(pred_y, dim=0)

            # 모델마다 나온 결과값 저장
            pred_y_list += [pred_y]
            model.cpu()

        # |y_hats| = (len(ensemble), len(lines), n_classes)
        # 마찬가지로 리스트를 하나의 차원으로 취급해서 텐서화
        # exp()을 취해서 로그 확률을 그냥 확률로 변환
        pred_y_list = torch.stack(pred_y_list).exp()
        
        # |y_hats| = (len(lines), n_classes)
        # sum(dim=0) 두 모델에 대한 값을 평균냄
        # == pred_y_list.mean(dim=0)
        pred_y_list = pred_y_list.sum(dim=0) / len(ensemble)

        probs, indice = pred_y_list.topk(config.top_k)

        for i in range(len(lines)):
            print('%s\t%s\n' % (
                # 예측한 라벨
                ' '.join([classes.itos[indice[i][j]] for j in range(config.top_k)]), 
                # 원문
                ' '.join(lines[i])
            ))


if __name__ == '__main__':
    config = define_argparser()
    main(config)


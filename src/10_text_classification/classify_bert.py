import sys
import argparse

import torch
import torch.nn.functional as F

from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

def define_argparser():
    p = argparse.ArgumentParser()
    p.add_argument('--model_fn', default='bert_model.pth')
    p.add_argument('--gpu_id', 
                   type=int,
                   default=0 if torch.cuda.is_available() else -1)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--top_k', type=int, default=1)
    return p.parse_args()


def read_text():
    lines = []
    for line in sys.stdin:
        if line.strip() != '':
            lines += [line.strip()]
    return lines


def main(config):
    saved_data = torch.load(
        config.model_fn,
        map_location='cpu' if config.gpu_id < 0 else 'cuda:%d' % config.gpu_id
    )

    train_config = saved_data['config']
    bert_best = saved_data['bert']
    index_to_label = saved_data['classes']

    lines = read_text()

    with torch.no_grad():
        # Declare model and load pre-trained weights.
        tokenizer = AutoTokenizer.from_pretrained(train_config.pretrained_model_name)
        model = BertForSequenceClassification.from_pretrained(
            train_config.pretrained_model_name,
            num_labels=len(index_to_label)
        )
        # 학습시켰던 다운스트림 태스크 모델을 삽입.
        model.load_state_dict(bert_best)

        if config.gpu_id >= 0:
            model.cuda(config.gpu_id)
        device = next(model.parameters()).device

        model.eval()

        y_hats = []
        for idx in range(0, len(lines), config.batch_size):
            mini_batch = tokenizer(
                lines[idx:idx + config.batch_size],
                padding=True,
                truncation=True,
                return_tensors='pt'
            )

            x = mini_batch['input_ids']
            x = x.to(device)
            mask = mini_batch['attention_mask']
            mask = mask.to(device)

            y_hat = F.softmax(model(x, attention_mask=mask)[0], dim=-1)
            y_hats += [y_hat]

        # Concatenate the mini-batch wise result
        y_hats = torch.cat(y_hats, dim=0)
        # |y_hats| = (len(lines), n_classes)

        probs, indice = y_hats.cpu().topk(config.top_k)
        # |indice| = (len(lines), top_k)

        for i in range(len(lines)):
            sys.stdout.write('%s\t%s\n' % (
                ' '.join([index_to_label[int(indice[i][j])] for j in range(config.top_k)]), 
                lines[i]
            ))


if __name__ == '__main__':
    config = define_argparser()
    main(config)

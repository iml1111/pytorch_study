import sys
import argparse
import torch

sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')

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
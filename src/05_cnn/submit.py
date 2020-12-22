import torch
import torch.nn
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import csv

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Device:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Device: CPU")


def load(fn, device):
    d = torch.load(fn, map_location=device)
    return d['config'], d['model']

def load_mnist():
    csv_data = pd.read_csv('test.csv')
    x = torch.tensor(csv_data.values, dtype=torch.double)[:,1:]
    return x

def test(model, x): 
    model.eval()
    x = x.view(-1, 28, 28)
    print(x.size())
    print(x)
    with torch.no_grad():
        pred_y = model(x.float())
        y_list = torch.argmax(pred_y, dim=-1)
        return y_list


def export_csv(y_list):
    f = open("submit.csv", 'w', newline='')
    wr = csv.writer(f)
    wr.writerow(["index","label"])
    for idx, y in enumerate(y_list):
        wr.writerow([idx, int(y)])
    f.close()


def main():
    from train import get_model
    model_fn = './model.pth'
    train_config, state_dict = load(model_fn, device)
    model = get_model(train_config).to(device)
    model.load_state_dict(state_dict)

    x = load_mnist()
    x = x.to(device)

    y_list = test(model, x)
    export_csv(y_list)


if __name__ == '__main__':
    main()
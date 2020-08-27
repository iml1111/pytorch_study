import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from classification.data_loader import get_loader
from classification.model_loader import get_model
from argparse import Namespace

model_fn = './model.pth'
config = {
    "model_name":"resnet",
    "dataset_name":"catdog",
    "train_ratio":.6,
    "valid_ratio":.2,
    "test_ratio":.2,
    'batch_size':10
}
config = Namespace(**config)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Device:", torch.cuda.get_device_name(0))
else:
    device = torch.device('cpu')
    print("Device: CPU")


def load(fn, device):
    d = torch.load(fn, map_location=device)
    return d['config'], d['model']


def plot(x, pred_y):
    for i in range(x.size(0)):
        img = (np.array(x[i].detach().cpu(), dtype='float')).transpose((1,2,0))
        plt.imshow(img, cmap='gray')
        print(pred_y[i])
        if float(torch.argmax(pred_y[i], dim=-1)):
            print("Predict: Dog!")
        else:
            print("Predict: Cat!")
        plt.show()


def test(model, x, y, to_be_shown=True):
    model.eval()
    with torch.no_grad():
        pred_y = model(x)
        correct_cnt = (y.squeeze() == torch.argmax(pred_y, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print("accuracy: %.4f" % accuracy)
        if to_be_shown:
            plot(x, pred_y)


def main():
    train_config, state_dict = load(model_fn, device)
    model, input_size = get_model(train_config)
    model.load_state_dict(state_dict)

    _, _, test_loader = get_loader(config, input_size)
    for x, y in test_loader:
        test(model, x, y, to_be_shown=True)
        break


if __name__ == '__main__':
    main()

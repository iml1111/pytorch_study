import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from mnist_classification.data_loader import load_mnist

model_fn = './model.pth'

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
        img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28, 28)
        plt.imshow(img, cmap='gray')
        print("Predict:", float(torch.argmax(pred_y[i], dim=-1)))
        plt.show()
 

def test(model, x, y, to_be_shown=True):
    model.eval()

    print(x.size())
    print(x)

    with torch.no_grad():
        pred_y = model(x)
        correct_cnt = (y.squeeze() == torch.argmax(pred_y, dim=-1)).sum()
        total_cnt = float(x.size(0))

        accuracy = correct_cnt / total_cnt
        print("accuracy: %.4f" % accuracy)
        if to_be_shown:
            plot(x, pred_y)


def main():
    from train import get_model
    train_config, state_dict = load(model_fn, device)

    model = get_model(train_config).to(device)
    model.load_state_dict(state_dict)
    print(train_config)

    x, y = load_mnist(is_train=False,
                      flatten=True if train_config.model == 'fc' else False)
    x, y = x.to(device), y.to(device)
    test(model, x[:20], y[:20], to_be_shown=True)


if __name__ == '__main__':
    main()
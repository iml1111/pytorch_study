import torch
import torch.nn
import sys
import numpy as np
import matplotlib.pyplot as plt
from model import ImageClassifier
from data_loader import load_mnist


def load(fn, device):
	d = torch.load(fn, map_location=device)
	return d['model']


def plot(x, pred_y):
	for i in range(x.size(0)):
		img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28, 28)
		plt.imshow(img, cmap='gray')
		print("Predict:", float(torch.argmax(pred_y[i], dim=-1)))
		plt.show()


def test(model, x, y, to_be_shown=True):
	model.eval()

	with torch.no_grad():
		pred_y = model(x)
		correct_cnt = (y.squeeze() == torch.argmax(pred_y, dim=-1)).sum()
		total_cnt = float(x.size(0))

		accuracy = correct_cnt / total_cnt
		print("Accuracy: %.4f" % accuracy)

		if to_be_shown:
			plot(x, pred_y)


def main():
	model_fn = "./model.pth"
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	x, y = load_mnist(is_train=True, flatten=True)
	x, y = x.to(device), y.to(device)

	model = ImageClassifier(28 * 28, 10).to(device)
	model.load_state_dict(load(model_fn, device))

	test(model, x[:20], y[:20], to_be_shown=True)


if __name__ == '__main__':
	main()
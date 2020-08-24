import torch
import torch.nn
import numpy as np
import matplotlib.pyplot as plt
from mnist_classification.data_loader import load_mnist
from train import get_model


model_fn = './model.pth'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def load(fn, device):
	d = torch.load(fn, map_location=device)
	return d['config'], d['model']


def plot(x, y_hat):
	for i in range(x.size(0)):
		img = (np.array(x[i].detach().cpu(), dtype='float')).reshape(28, 28)
		plt.imshow(img, cmap='gray')
		print('Predict:', float(torch.argmax(y_hat[i], dim=-1)))
		plt.show()


def test(model, x, y, to_be_shown=True):
	model.eval()
	with torch.no_grad():
		
		y_hat = model(x)
		print("x_Size:",x.size())
		print("y_hat_Size:",y_hat.size())
		print("y_size:", y.size())

		print(y.squeeze())
		print(y_hat[0])
		print(torch.argmax(y_hat, dim=-1))

		correct_cnt = (y.squeeze() == torch.argmax(y_hat, dim=-1)).sum()
		total_cnt = float(x.size(0))

		accuracy = correct_cnt / total_cnt
		print("accuracy: %.4f" % accuracy)

		if to_be_shown:
			plot(x, y_hat)


def main():
	train_config, state_dict = load(model_fn, device)
	model = get_model(train_config).to(device)
	model.load_state_dict(state_dict)
	print(model)

	# Load MNIST test set.
	x, y = load_mnist(is_train=False,
                  	  flatten=True if train_config.model == 'fc' else False)

	x, y = x.to(device), y.to(device)

	test(model, x[:20], y[:20], to_be_shown=True)

if __name__ == '__main__':
	main()
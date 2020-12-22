import torch
from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset):
	'''파이토치 데이터 셋 클래스'''
	def __init__(self, data, labels, flatten=True):
		self.data = data
		self.labels = labels
		self.flatten = flatten
		super().__init__()

	def __len__(self):
		return self.data.size(0)

	def __getitem__(self, idx):
		x = self.data[idx]
		y = self.labels[idx]
		if self.flatten:
			x = x.view(-1)
		return x, y


def load_mnist(is_train=True, flatten=True):
	'''MNIST 데이터 호출 함수'''
	from torchvision import datasets, transforms

	dataset = datasets.MNIST(
		'../data', 
		train=is_train, 
		download=True,
		transform=transforms.Compose([
			transforms.ToTensor(),
		]),
	)

	x = dataset.data.float() / 255.
	y = dataset.targets
	if flatten:
		x = x.view(x.size(0), -1)

	return x, y


def get_loaders(config):
	x, y = load_mnist(is_train=True, flatten=False)
	train_cnt = int(x.size(0) * config.train_ratio)
	valid_cnt = x.size(0) - train_cnt

	flatten = True if config.model == 'fc' else False

	# 트레인, 밸리드 데이터 셔플 
	indices = torch.randperm(x.size(0))
	train_x, valid_x = torch.index_select(
		x,
		dim=0,
		index=indices
	).split([train_cnt, valid_cnt], dim=0)
	train_y, valid_y = torch.index_select(
		y,
		dim=0,
		index=indices
	).split([train_cnt, valid_cnt], dim=0)

	# 테스트 데이터 불러와서 트레인에 합치기 (어차피 안씀)
	test_x, test_y = load_mnist(is_train=False, flatten=False)
	train_x = torch.cat((train_x, test_x), 0)
	train_y = torch.cat((train_y, test_y), 0)

	# 데이터 로더 생성(셔플: 트레인은 필수, 나머지는 자유)
	train_loader = DataLoader(
		dataset=MnistDataset(train_x, train_y, flatten=flatten),
		batch_size=config.batch_size,
		shuffle=True,
	)
	valid_loader = DataLoader(
		dataset=MnistDataset(valid_x, valid_y, flatten=flatten),
		batch_size=config.batch_size,
		shuffle=True,
	)
	test_loader = DataLoader(
		dataset=MnistDataset(test_x, test_y, flatten=flatten),
		batch_size=config.batch_size,
		shuffle=False,
	)

	return train_loader, valid_loader, test_loader

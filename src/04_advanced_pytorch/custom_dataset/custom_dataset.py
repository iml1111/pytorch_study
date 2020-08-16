from torch.utils.data import Dataset

class CustomDataset(Dataset):

	def __init__(self, data, labels):
		self.data = data
		self.labels = labels
		super().__init__()

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx], self.labels[idx]
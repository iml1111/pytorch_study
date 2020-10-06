import torch
from torchtext import data
from torch.utils.data import Dataset


class DataLoader:
	'''Data Loader'''
	def __init__(
		self,
		train_fn,
		batch_size=64,
		valid_ratio=.2,
		device=-1,
		max_vocab=999999,
		min_freq=1,
		use_eos=False,
		shuffle=True):

		self.label = data.Field(
			sequential=False,
			use_vocab=True,
			unk_token=None)
		self.text = data.Field(
			use_vocab=True,
			batch_first=True,
			include_lengths=False,
			eos_token='<EOS>' if use_eos else None)

		train_dataset
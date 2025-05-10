import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from transformers import EsmConfig


class OneHotEncoding(nn.Module):

	def __init__(self,
		vocab_size: int):
		super().__init__()
		self.embed_size = vocab_size

	def forward(self, x):
		return F.one_hot(x.to(torch.int64), num_classes=self.embed_size)


class SimpleMLP(nn.Module):

	def __init__(self,
		in_dim: int,
		hid_dim: int,
		out_dim: int,
		dropout: float = 0.1):
		super().__init__()
		self.mlp = nn.Sequential(
			weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
			nn.ReLU(),
			nn.Dropout(dropout),
			weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
		)

	def forward(self, x):
		return self.mlp(x)


class Cnn(nn.Module):

	def __init__(self,
		in_dim: int,
		out_dim: int,
		cnn_width: int = 7,
		dropout: float = 0.1):
		super().__init__()
		self.cnn = weight_norm(nn.Conv1d(
				in_dim, 
				in_dim, 
				cnn_width, padding=cnn_width//2), 
			dim=None
		)
		self.dense = weight_norm(nn.Linear(
				in_dim,
				out_dim),
			dim=None
		)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		tmp = x.permute(0, 2, 1)
		output = self.cnn(tmp).permute(0, 2, 1)
		output = self.dropout(output)

		logits = self.dense(output)
		return logits


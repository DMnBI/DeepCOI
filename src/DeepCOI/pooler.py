import torch
from torch import nn


class ConvolutionalAttention(nn.Module):

	def __init__(self,
		embed_dim: int = 1280,
		width: int = 9):
		super().__init__()

		self._la_w1 = nn.Conv1d(embed_dim, embed_dim, width, padding=width//2)
		self._la_w2 = nn.Conv1d(embed_dim, embed_dim, width, padding=width//2)
		self._la_mlp = nn.Linear(embed_dim*2, embed_dim)

	def forward(self, embeddings, masks=None):
		_temp = embeddings.permute(0, 2, 1)
		a = self._la_w1(_temp).softmax(dim=-1)
		v = self._la_w2(_temp)
		v_max = v.max(dim=-1).values
		v_sum = (a * v).sum(dim=-1)

		return self._la_mlp(torch.cat([v_max, v_sum], dim=1)), a


class MeanPooling(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, embeddings, masks=None):
		if masks is None:
			extended_masks = torch.ones_like(embeddings)
		else:
			extended_masks = masks.unsqueeze(-1).expand_as(embeddings)
		embeddings = embeddings * extended_masks

		return torch.sum(embeddings, dim=1) / extended_masks.sum(dim=1)


class MaxPooling(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, embeddings, masks=None):
		if masks is None:
			extended_masks = torch.ones_like(embeddings)
		else:
			extended_masks = masks.unsqueeze(-1).expand_as(embeddings)
		_tmp = embeddings + (-10000 * (1 - extended_masks))

		return _tmp.max(dim=1).values


class CLRPooling(nn.Module):

	def __init__(self, embed_dim: int):
		super().__init__()
		self.dense = nn.Linear(embed_dim, embed_dim)
		self.activation = nn.Tanh()

	def forward(self, embeddings, masks=None):
		clr = embeddings[:, 0, :]
		pooled = self.dense(clr)
		
		return self.activation(pooled)

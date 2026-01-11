import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class DropPath(nn.Module):

	def __init__(self, drop_prob: float = 0.0):
		super().__init__()
		self.drop_prob = drop_prob

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		if self.drop_prob == 0.0 or not self.training:
			return x
		keep_prob = 1 - self.drop_prob
		shape = (x.shape[0],) + (1,) * (x.ndim - 1)
		random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
		random_tensor.floor_()
		return x.div(keep_prob) * random_tensor

class SE(nn.Module):

	def __init__(self, in_channels: int, reduct: int = 4):
		super().__init__()
		self.se = nn.Sequential(
			nn.Linear(in_channels, in_channels // reduct),
			nn.ReLU(),
			nn.Linear(in_channels // reduct, in_channels),
			nn.Sigmoid()
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, C, H, W = x.shape
		squeezed = x.mean(dim=(2, 3))
		out = self.se(squeezed)
		out = out.view(B, C, 1, 1)
		return x * out


class InvertedResidual(nn.Module):

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		stride: int = 1,
		expand_ratio: int = 4,
		drop_path: float = 0.0,
		se_reduct: int = 4
	):
		super().__init__()
		hidden_dim = in_channels * expand_ratio
		self.use_residual = stride == 1 and in_channels == out_channels
		self.expand_ratio = expand_ratio

		if expand_ratio != 1:
			self.expand = nn.Sequential(
				nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
				nn.BatchNorm2d(hidden_dim),
				nn.GELU(),
			)

		self.depthwise = nn.Sequential(
			nn.Conv2d(hidden_dim, hidden_dim, 3, stride=stride, padding=1, groups=hidden_dim, bias=False),
			nn.BatchNorm2d(hidden_dim),
			nn.GELU(),
		)

		self.se = SE(hidden_dim, se_reduct)

		self.project = nn.Sequential(
			nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
			nn.BatchNorm2d(out_channels),
		)

		self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.expand(x) if self.expand_ratio != 1 else x
		out = self.depthwise(out)
		out = self.se(out)
		out = self.project(out)
		if self.use_residual:
			return x + self.drop_path(out)
		return out


class Encoder(nn.Module):

	def __init__(
		self,
		n_blocks: int,
		channels: List[int],
		strides: List[int],
		expand_ratio: int = 4,
		drop_path_rate: float = 0.1,
		n_classes: int = 100,
	):
		super().__init__()

		self.stem = nn.Sequential(
			nn.Conv2d(3, channels[0], kernel_size=3, stride=1, padding=1, bias=False),
			nn.BatchNorm2d(channels[0]),
			nn.GELU()
		)

		drop_path_rates = [drop_path_rate * i / (n_blocks - 1) for i in range(n_blocks)]

		blocks = []
		for i in range(n_blocks):
			blocks.append(
				InvertedResidual(
					in_channels=channels[i],
					out_channels=channels[i + 1],
					stride=strides[i],
					expand_ratio=expand_ratio,
					drop_path=drop_path_rates[i]
				)
			)

		self.blocks = nn.Sequential(*blocks)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.fc = nn.Linear(channels[-1], n_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.stem(x)
		out = self.blocks(out)
		out = self.pool(out)
		out = out.flatten(1)
		out = self.fc(out)
		return out

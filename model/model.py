import torch
from torch import nn
from torchvision import transforms
from torch.nn import functional as F
from typing import List

class DepthWiseSeparable(nn.Module):
	"""
		Implementation of DepthWiseSeparable Convolutions.
		Shoutout MobileNet.
	"""

	def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int):
		super().__init__()
		self.depth = nn.Conv2d(
			in_channels,
			in_channels,
			kernel_size = kernel_size,
			padding = padding,
			groups = in_channels,
			bias = False  # BatchNorm has its own bias
		)
		self.bn1 = nn.BatchNorm2d(in_channels)
		self.point = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size = 1,
			bias = False  # BatchNorm has its own bias
		)
		self.bn2 = nn.BatchNorm2d(out_channels)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = F.gelu(self.bn1(self.depth(x)))
		out = F.gelu(self.bn2(self.point(out)))
		return out

class ChannelAttention(nn.Module): 
	"""
		Weights channels when combining. 
	"""
	
	def __init__(self, in_channels: int, ratio: int):
		super().__init__()
		self.ca = nn.Sequential(
			nn.Linear(in_channels, in_channels // ratio),
			nn.ReLU(inplace=True),
			nn.Linear(in_channels // ratio, in_channels),
			nn.Sigmoid()
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		B, C, H, W = x.shape
		pooled = x.mean(dim = (2, 3))
		out = self.ca(pooled).view(B, C, 1, 1)
		return x * out

class Blocky(nn.Module):

	def __init__(
		self,
		in_channels: int,
		out_channels: int,
		kernel_size: int,
		padding: int,
		ratio: int,
		dropout: float = 0.1
	):
		super().__init__()
		self.block = nn.Sequential(
			DepthWiseSeparable(in_channels, out_channels, kernel_size, padding),
			ChannelAttention(out_channels, ratio),
			nn.Dropout2d(p=dropout)
		)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		return self.block(x)

class Encoder(nn.Module):
	"""
		Block for generating dense token representation of a patch.
	"""

	def __init__(
		self,
		n_blocks: int,
		kernel_sizes: List[int],
		channels: List[int],
		paddings: List[int],
		ratios: List[int],
		n_classes: int,
		dropout: float = 0.1
	):
		super().__init__()

		blocks = []
		for i in range(n_blocks):
			blocks.append(
				Blocky(
					in_channels=channels[i],
					out_channels=channels[i+1],
					kernel_size=kernel_sizes[i],
					padding=paddings[i],
					ratio=ratios[i],
					dropout=dropout
				)
			)

		self.blocks = nn.Sequential(*blocks)
		self.pool = nn.AdaptiveAvgPool2d(1)
		self.dropout = nn.Dropout(p=dropout * 3)  # Higher dropout before classifier
		self.fc = nn.Linear(channels[-1], n_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = self.blocks(x)
		out = self.pool(out)
		out = out.flatten(1)
		out = self.dropout(out)
		out = self.fc(out)
		return out




		






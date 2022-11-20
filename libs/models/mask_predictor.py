import torch
from torch import nn


class MaskPredictor(nn.Module):
	def __init__(self, input_dim, output_dim, inner_dim=1024):
		super(MaskPredictor, self).__init__()

		
		self.masknet = nn.Sequential(nn.Linear(input_dim, inner_dim, bias=True),
									nn.ReLU(),
									nn.Linear(inner_dim, output_dim, bias=True),
									)
		
		self.initilization()

	def initilization(self):
		torch.nn.init.normal_(self.masknet[0].weight, mean=0.0, std=0.01)
		torch.nn.init.normal_(self.masknet[2].weight, mean=0.0, std=0.01)

	def forward(self, input):

		out = self.masknet(input)
		out = torch.nn.Sigmoid()(out)
			
		return out


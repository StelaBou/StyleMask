"""
This file defines the core research contribution
"""
import math
import matplotlib
matplotlib.use('Agg')
import torch
from torch import nn
import torchvision.transforms as transforms
import os

from libs.models.inversion import psp_encoders



def get_keys(d, name):
	if 'state_dict' in d:
		d = d['state_dict']
	d_filt = {k[len(name) + 1:]: v for k, v in d.items() if k[:len(name)] == name}
	return d_filt


class pSp(nn.Module):

	def __init__(self, opts):
		super(pSp, self).__init__()

		self.opts = opts
		# compute number of style inputs based on the output resolution
		self.opts.n_styles = int(math.log(self.opts.output_size, 2)) * 2 - 2
		self.n_styles = self.opts.n_styles
		# Define architecture
		self.encoder = psp_encoders.Encoder4Editing(50, 'ir_se', self.opts)
		# Load weights if needed
		ckpt = torch.load(self.opts.checkpoint_path, map_location='cpu')
		self.encoder.load_state_dict(get_keys(ckpt, 'encoder'), strict=True)

		self.__load_latent_avg(ckpt)

	def forward(self, real_image, randomize_noise=False, inject_latent=None, return_latents=False, alpha=None, average_code=False, input_is_full=False):
		
		codes = self.encoder(real_image)	
		if self.latent_avg is not None:
			if codes.ndim == 2:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)[:, 0, :]
			else:
				codes = codes + self.latent_avg.repeat(codes.shape[0], 1, 1)
		return codes	
	
	def __load_latent_avg(self, ckpt, repeat=None):
		if 'latent_avg' in ckpt:
			self.latent_avg = ckpt['latent_avg'].to(self.opts.device)
			if repeat is not None:
				self.latent_avg = self.latent_avg.repeat(repeat, 1)
		else:
			self.latent_avg = None
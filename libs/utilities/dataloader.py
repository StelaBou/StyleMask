"""
"""
import torch
import os
import glob
import cv2
import numpy as np
from torchvision import transforms, utils
from PIL import Image
from torch.utils.data import Dataset

from libs.utilities.utils import make_noise

np.random.seed(0)

class CustomDataset_validation(Dataset):

	def __init__(self, synthetic_dataset_path = None, validation_pairs = None, shuffle = True):
		"""
		Args:
			synthetic_dataset_path:				path to synthetic latent codes. If None generate random 
			num_samples:						how many samples for validation
			
		"""
		self.shuffle = shuffle
		self.validation_pairs = validation_pairs
		self.synthetic_dataset_path = synthetic_dataset_path
	
		if self.synthetic_dataset_path is not None:
			z_codes = np.load(self.synthetic_dataset_path)
			z_codes = torch.from_numpy(z_codes)
			if self.validation_pairs is not None:
				self.num_samples = 2 * self.validation_pairs
				if z_codes.shape[0] > self.num_samples:
					z_codes = z_codes[:self.num_samples]
				else:
					self.num_samples = z_codes.shape[0]
					self.validation_pairs = int(self.num_samples/2)
			else:
				self.validation_pairs = int(z_codes.shape[0]/2)
				self.num_samples = 2 * self.validation_pairs

			self.fixed_source_w =  z_codes[:self.validation_pairs, :]
			self.fixed_target_w =  z_codes[self.validation_pairs:2*self.validation_pairs, :]			
		else:
			self.fixed_source_w = make_noise(self.validation_pairs, 512, None)
			self.fixed_target_w = make_noise(self.validation_pairs, 512, None)
			# Save random generated latent codes 
			save_path = './libs/configs/random_latent_codes_{}.npy'.format(self.validation_pairs)
			z_codes = torch.cat((self.fixed_source_w, self.fixed_target_w), dim = 0)
			np.save(save_path, z_codes.detach().cpu().numpy())

		self.transform = transforms.Compose([
				transforms.Resize((256, 256)),
				transforms.ToTensor(),
				transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

	
	def __len__(self):	
		return self.validation_pairs

	def __getitem__(self, index):
	
		source_w =  self.fixed_source_w[index]
		target_w =  self.fixed_target_w[index]
		sample = {
			'source_w':				source_w,
			'target_w':				target_w
		}
		return sample


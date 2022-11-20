"""
Script to extract the npy file with the min, max values of facial pose parameters (yaw, pitch, roll, jaw and expressions)
1. Generate a set of random synthetic images
2. Use DECA model to extract the facial shape and the corresponding parameters
3. Calculate min, max values
"""


import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
import json
import cv2
from tqdm import tqdm
import argparse
from torchvision import utils as torch_utils
import warnings
warnings.filterwarnings("ignore")

from libs.configs.config_models import *
from libs.utilities.utils import make_noise, make_path, calculate_shapemodel
from libs.DECA.estimate_DECA import DECA_model
from libs.models.StyleGAN2.model import Generator as StyleGAN2Generator



def extract_stats(statistics):

	num_stats = statistics.shape[1]
	statistics = np.transpose(statistics, (1, 0))
	ranges = []
	for i in range(statistics.shape[0]):
		pred = statistics[i, :]	
		max_ = np.amax(pred)	
		min_ = np.amin(pred)
		if i == 0:
			label = 'yaw'
		elif i == 1:
			label = 'pitch'
		elif i == 2:
			label = 'roll'
		elif i == 3:
			label = 'jaw'
		else:
			label = 'exp_{:02d}'.format(i)

		print('{}/{} Min {:.2f} Max {:.2f}'.format(label, i, min_, max_))
		
		ranges.append([min_, max_])

	return ranges

	

if __name__ == '__main__':

	num_images = 2000

	image_resolution = 1024
	dataset = 'FFHQ'

	output_path = './{}_stats'.format(dataset)
	make_path(output_path)

	gan_weights = stylegan2_ffhq_1024['gan_weights']
	channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']

	print('----- Load generator from {} -----'.format(gan_weights))
	truncation = 0.7		
	generator = StyleGAN2Generator(image_resolution, 512, 8, channel_multiplier= channel_multiplier)	
	generator.load_state_dict(torch.load(gan_weights)['g_ema'], strict = True)
	generator.cuda().eval()
	trunc = generator.mean_latent(4096).detach().clone()

	shape_model = DECA_model('cuda')
	face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))	
		
	statistics = []
	with torch.no_grad():
		for i in tqdm(range(num_images)):
			z = make_noise(1, 512).cuda()
			source_img = generator([z], return_latents = False, truncation = truncation, truncation_latent = trunc, input_is_latent = False)[0]
			source_img = face_pool(source_img)
			params_source, angles_source = calculate_shapemodel(shape_model, source_img)

			yaw = angles_source[:,0][0].detach().cpu().numpy() 
			pitch = angles_source[:,1][0].detach().cpu().numpy() 
			roll = angles_source[:, 2][0].detach().cpu().numpy() 
			exp = params_source['alpha_exp'][0].detach().cpu().numpy() 
			jaw =  params_source['pose'][0, 3].detach().cpu().numpy()
			
			tmp = np.zeros(54)
			tmp[0] = yaw
			tmp[1] = pitch
			tmp[2] = roll
			tmp[3] = jaw
			tmp[4:] = exp
			# np.save(os.path.join(output_path, '{:07d}.npy'.format(i)), tmp)
			statistics.append(tmp)

	statistics = np.asarray(statistics)
	np.save(os.path.join(output_path, 'stats_all.npy'), statistics)

	ranges = extract_stats(statistics)
	
	np.save(os.path.join(output_path, 'ranges_{}.npy'.format(dataset)), ranges)
	

	
	

	
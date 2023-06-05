import os
import datetime
import random
import sys
import argparse
from argparse import Namespace
import torch
from torch import nn
import numpy as np
import warnings
from tqdm import tqdm
warnings.filterwarnings("ignore")
sys.dont_write_bytecode = True

seed = 0
random.seed(seed)
import face_alignment

from libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.utils import make_noise, generate_image, generate_new_stylespace, save_image, save_grid, get_files_frompath
from libs.utilities.stylespace_utils import decoder
from libs.configs.config_models import stylegan2_ffhq_1024
from libs.utilities.utils_inference import preprocess_image, invert_image
from libs.utilities.image_utils import image_to_tensor
from libs.models.inversion.psp import pSp

class Inference_demo():

	def __init__(self, args):
		self.args = args
		
		self.device = 'cuda'
		self.output_path = args['output_path']
		arguments_json = os.path.join(self.output_path, 'arguments.json')
		self.masknet_path = args['masknet_path']
		self.image_resolution = args['image_resolution']
		self.dataset = args['dataset']

		self.source_path = args['source_path']
		self.target_path = args['target_path']
		self.num_pairs = args['num_pairs']

		self.save_grid = args['save_grid']
		self.save_image = args['save_image']
		self.resize_image = args['resize_image']
		
		if not os.path.exists(self.output_path):
			os.makedirs(self.output_path, exist_ok=True)
				
	def load_models(self, inversion):

		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		if self.dataset == 'ffhq' and self.image_resolution == 1024:
			self.generator_path = stylegan2_ffhq_1024['gan_weights'] 
			self.channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
			self.split_sections = stylegan2_ffhq_1024['split_sections']
			self.stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']
		else:
			print('Incorect dataset type {} and image resolution {}'.format(self.dataset, self.image_resolution))

		if os.path.exists(self.generator_path):
			print('----- Load generator from {} -----'.format(self.generator_path))
					
			self.G = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier = self.channel_multiplier)
			self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = True)
			self.G.cuda().eval()
			# use truncation 
			self.truncation = 0.7
			self.trunc =self.G.mean_latent(4096).detach().clone()			
			
		else:
			print('Please download the pretrained model for StyleGAN2 generator and save it into ./pretrained_models path')
			exit()

		if os.path.exists(self.masknet_path):
			print('----- Load mask network from {} -----'.format(self.masknet_path))
			ckpt = torch.load(self.masknet_path, map_location=torch.device('cpu'))
			self.num_layers_control = ckpt['num_layers_control']
			self.mask_net = nn.ModuleDict({})
			for layer_idx in range(self.num_layers_control):
				network_name_str = 'network_{:02d}'.format(layer_idx)

				# Net info
				stylespace_dim_layer = self.split_sections[layer_idx]	
				input_dim = stylespace_dim_layer
				output_dim = stylespace_dim_layer
				inner_dim = stylespace_dim_layer

				network_module = MaskPredictor(input_dim, output_dim, inner_dim = inner_dim)
				self.mask_net.update({network_name_str: network_module})
			self.mask_net.load_state_dict(ckpt['mask_net'])
			self.mask_net.cuda().eval()
		else:
			print('Please download the pretrained model for Mask network and save it into ./pretrained_models path')
			exit()
		
		if inversion:
			self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, device='cuda')
			### Load inversion model only when the input is image. ###
			self.encoder_path = stylegan2_ffhq_1024['e4e_inversion_model']
			print('----- Load e4e encoder from {} -----'.format(self.encoder_path))
			ckpt = torch.load(self.encoder_path, map_location='cpu')
			opts = ckpt['opts']
			opts['output_size'] = self.image_resolution
			opts['checkpoint_path'] = self.encoder_path
			opts['device'] = 'cuda'
			opts['channel_multiplier'] = self.channel_multiplier
			opts['dataset'] = self.dataset
			opts = Namespace(**opts)
			self.encoder = pSp(opts)
			self.encoder.cuda().eval()

	def load_samples(self, filepath):
		inversion = False
		if filepath is None:
			# Generate random latent code
			files_grabbed = []
			for i in range(self.num_pairs):
				files_grabbed.append(make_noise(1, 512))
		else:
			if os.path.isdir(filepath):
				## Check if files inside directory are images. Else check if latent codes
				files_grabbed = get_files_frompath(filepath, ['*.png', '*.jpg'])
				if len(files_grabbed) == 0:
					files_grabbed = get_files_frompath(filepath, ['*.npy'])
					if len(files_grabbed) == 0:
						print('Please specify correct path: folder with images (.png, .jpg) or latent codes (.npy)')
						exit()
					z_codes = []
					for file_ in files_grabbed:
						z_codes.append(torch.from_numpy(np.load(file_)).cuda())
					z_codes = torch.cat(z_codes).unsqueeze(0)	
					files_grabbed = z_codes
				else:
					inversion = True # invert real images

			elif os.path.isfile(filepath):

				head, tail = os.path.split(filepath)
				ext = tail.split('.')[-1]
				# Check if file is image
				if ext == 'png' or ext == 'jpg':
					files_grabbed = [filepath]
					inversion = True
				elif ext == 'npy':
					z_codes = torch.from_numpy(np.load(filepath)).unsqueeze(1)
					files_grabbed = z_codes	
				else:
					print('Wrong path. Expected file image (.png, .jpg) or latent code (.npy)')
					exit()
			else:
				print('Wrong path. Expected file image (.png, .jpg) or latent code (.npy)')
				exit()

		return files_grabbed, inversion

	def reenact_pair(self, source_code, target_code):
		
		with torch.no_grad():
			# Get source style space
			source_img, style_source, w_source, noise_source = generate_image(self.G, source_code, self.truncation, self.trunc, self.image_resolution, self.split_sections,
					input_is_latent = self.input_is_latent, return_latents= True, resize_image = self.resize_image)

			# Get target style space
			target_img, style_target, w_target, noise_target = generate_image(self.G, target_code, self.truncation, self.trunc, self.image_resolution, self.split_sections,
					input_is_latent = self.input_is_latent, return_latents= True, resize_image = self.resize_image)
		
			# Get reenacted image
			masks_per_layer = []
			for layer_idx in range(self.num_layers_control):
				network_name_str = 'network_{:02d}'.format(layer_idx)
				style_source_idx = style_source[layer_idx]
				style_target_idx = style_target[layer_idx]			
				styles = style_source_idx - style_target_idx
				mask_idx = self.mask_net[network_name_str](styles)
				masks_per_layer.append(mask_idx)

			mask = torch.cat(masks_per_layer, dim=1)
			style_source = torch.cat(style_source, dim=1)
			style_target = torch.cat(style_target, dim=1)

			new_style_space = generate_new_stylespace(style_source, style_target, mask, num_layers_control = self.num_layers_control)
			new_style_space = list(torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
			reenacted_img = decoder(self.G, new_style_space, w_source, noise_source, resize_image = self.resize_image)
			
		return source_img, target_img, reenacted_img

	def check_paths(self):
		assert type(self.target_path) == type(self.source_path), \
		 "Source path and target path should have the same type, None, files (.png, .jpg or .npy) or directories with files of type .png, .jpg or .npy"
		
		if self.source_path is not None and self.target_path is not None:
			if os.path.isdir(self.source_path):
				assert os.path.isdir(self.target_path), \
						"Source path and target path should have the same type, None, files (.png, .jpg or .npy) or directories with files of type .png, .jpg or .npy"

			if os.path.isfile(self.source_path):
				assert os.path.isfile(self.target_path), \
						"Source path and target path should have the same type, None, files (.png, .jpg or .npy) or directories with files of type .png, .jpg or .npy"
	
	def run(self):
		
		self.check_paths()
		source_samples, inversion = self.load_samples(self.source_path)
		target_samples, inversion = self.load_samples(self.target_path)

		assert len(source_samples) == len(target_samples), "Number of source samples should be the same with target samples"
		

		self.load_models(inversion)
		self.num_pairs = len(source_samples)
		
		print('Reenact {} pairs'.format(self.num_pairs))

		for i in tqdm(range(self.num_pairs)):
			if inversion: # Real image
				# Preprocess and invert real images into the W+ latent space using Encoder4Editing method
				cropped_image = preprocess_image(source_samples[i], self.fa, save_filename = None)	
				print(cropped_image.shape)
				source_img = image_to_tensor(cropped_image).unsqueeze(0).cuda()
				inv_image, source_code = invert_image(source_img, self.encoder, self.G, self.truncation, self.trunc)
				
				cropped_image = preprocess_image(target_samples[i], self.fa)		
				target_img = image_to_tensor(cropped_image).unsqueeze(0).cuda()
				inv_image, target_code = invert_image(target_img, self.encoder, self.G, self.truncation, self.trunc)
				self.input_is_latent = True		
			else: # synthetic latent code
				if self.source_path is not None:
					source_code = source_samples[i].cuda()
					target_code = target_samples[i].cuda()
					if source_code.ndim == 2:
						self.input_is_latent = False # Z space 1 X 512
					elif source_code.ndim == 3:
						self.truncation = 1.0 
						self.trunc = None 
						self.input_is_latent = True # W sapce 1 X 18 X 512
				else:
					source_code = source_samples[i].cuda()
					target_code = target_samples[i].cuda()
					self.input_is_latent = False # Z space

			source_img, target_img, reenacted_img = self.reenact_pair(source_code, target_code)

			if self.save_grid:
				save_grid(source_img, target_img, reenacted_img, os.path.join(self.output_path, 'grid_{:04d}.png').format(i))
			if self.save_image:
				save_image(reenacted_img, os.path.join(self.output_path, '{:04d}.png').format(i))
	
def main():
	"""
	Inference script.
	
	Options:
		######### General ###########
		--output_path		   				: path to save output images

		--source_path						: It can be either an image file, or a latent code or a directory with images or latent codes or None.
											   If source path is None then it will generate a random latent code.
		--target_path						: It can be either an image file, or a latent code or a directory with images or latent codes or None.
											   If target path is None then it will generate a random latent code.

		--masknet_path				   		: path to pretrained model for mask network
		--dataset							: dataset (ffhq)
		--image_resolution					: image resolution (1024)
		--num_pairs							: number of pairs to reenact

		########## Visualization ########## 
		--save_grid							: Generate figure with source, target and reenacted image
		--save_image						: Save only the reenacted image
		--resize_image						: Resize image from 1024 to 256

	
	python run_inference.py --output_path ./results --save_grid 

	"""
	parser = argparse.ArgumentParser(description="training script")

	######### General #########
	parser.add_argument('--output_path', type=str, required = True, help="path to save output images")
	parser.add_argument('--source_path', type=str, default = None, help='path to source samples (latent codes or images)')
	parser.add_argument('--target_path', type=str, default = None, help='path to target samples (latent codes or images)')

	parser.add_argument('--masknet_path', type=str, default = './pretrained_models/mask_network_1024.pt', help="path to pretrained model for mask network")
	parser.add_argument('--dataset', type=str, default = 'ffhq', help="dataset")
	parser.add_argument('--image_resolution', type=int, default = 1024, help="image resolution")

	parser.add_argument('--num_pairs', type=int, default = 4, help="number of random pairs to reenact")

	parser.add_argument('--save_grid', dest='save_grid', action='store_true', help="Generate figure with source, target and reenacted image")
	parser.set_defaults(save_grid=False)
	parser.add_argument('--save_image', dest='save_image', action='store_true', help="Save only the reenacted image")
	parser.set_defaults(save_image=False)
	parser.add_argument('--resize_image', dest='resize_image', action='store_true', help="Resize image from 1024 to 256")
	parser.set_defaults(resize_image=False)
	

	# Parse given arguments
	args = parser.parse_args()	
	args = vars(args) # convert to dictionary

	inf = Inference_demo(args)
	inf.run()

	


if __name__ == '__main__':
	main()

	
		

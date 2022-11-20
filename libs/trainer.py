"""
"""

import os
import json
import torch
import time
import numpy as np
import pdb
import cv2
import wandb
from torch import autograd
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from libs.utilities.utils import *
from libs.utilities.image_utils import *
from libs.DECA.estimate_DECA import DECA_model
from libs.models.StyleGAN2.model import Generator as StyleGAN2Generator
from libs.models.mask_predictor import MaskPredictor
from libs.utilities.stylespace_utils import decoder
from libs.configs.config_models import stylegan2_ffhq_1024
from libs.criteria.losses import Losses
from libs.criteria import id_loss
from libs.criteria.lpips.lpips import LPIPS
from libs.utilities.utils_inference import generate_grid_image, calculate_evaluation_metrics
from libs.utilities.dataloader import CustomDataset_validation

class Trainer(object):

	def __init__(self, args):
		
		self.args = args
		self.initialize_arguments(args)
		################# Initialize output paths #################
		make_path(self.output_path)
		self.log_dir = os.path.join(self.output_path, 'logs')
		make_path(self.log_dir)	
		self.models_dir = os.path.join(self.output_path, 'models')
		make_path(self.models_dir)
		####################################################################
		
		# save arguments file with params
		save_arguments_json(args, self.output_path, 'arguments.json')
	
	def initialize_arguments(self, args):

		self.output_path = args['experiment_path']
		self.use_wandb = args['use_wandb']
		self.log_images_wandb = args['log_images_wandb']
		self.project_wandb = args['project_wandb']
		self.resume_training_model = args['resume_training_model']

		self.image_resolution = args['image_resolution']
		self.dataset_type = args['dataset_type']
		self.synthetic_dataset_path = args['synthetic_dataset_path']

		self.lr = args['lr'] 
		self.num_layers_control = args['num_layers_control']
		self.max_iter = args['max_iter'] 
		self.batch_size = args['batch_size'] 
		self.test_batch_size = args['test_batch_size']
		self.workers = args['workers']

		# Weights
		self.lambda_identity = args['lambda_identity']
		self.lambda_perceptual = args['lambda_perceptual']
		self.lambda_shape = args['lambda_shape']
		self.use_recurrent_cycle_loss = args['use_recurrent_cycle_loss']
		
		self.steps_per_log = args['steps_per_log']
		self.steps_per_save_models = args['steps_per_save_models']
		self.steps_per_evaluation = args['steps_per_evaluation']
		self.validation_pairs = args['validation_pairs']
		self.num_pairs_log = args['num_pairs_log']
		# if self.num_pairs_log > self.validation_pairs:
		# 	self.num_pairs_log = self.validation_pairs

		

	def load_models(self):

		################## Initialize models #################
		print('-- Load DECA model ')
		self.deca = DECA_model('cuda')
		self.id_loss_ = id_loss.IDLoss().cuda().eval()
		self.lpips_loss = LPIPS(net_type='alex').cuda().eval()
		self.losses = Losses()
		####################################################################

		self.face_pool = torch.nn.AdaptiveAvgPool2d((256, 256))

		if self.dataset_type == 'ffhq' and self.image_resolution == 1024:
			self.generator_path = stylegan2_ffhq_1024['gan_weights'] 
			self.channel_multiplier = stylegan2_ffhq_1024['channel_multiplier']
			self.split_sections = stylegan2_ffhq_1024['split_sections']
			self.stylespace_dim = stylegan2_ffhq_1024['stylespace_dim']
			self.exp_ranges = np.load(stylegan2_ffhq_1024['expression_ranges'])
			
		else:
			print('Incorect dataset type {} and image resolution {}'.format(self.dataset_type, self.image_resolution))

		if self.num_layers_control is not None:
			self.num_nets = self.num_layers_control
		else:
			self.num_nets = len(self.split_sections)
			

		print('-- Load generator from {} '.format(self.generator_path))
		self.G = StyleGAN2Generator(self.image_resolution, 512, 8, channel_multiplier= self.channel_multiplier)
		if self.image_resolution == 256:
			self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = False)
		else:
			self.G.load_state_dict(torch.load(self.generator_path)['g_ema'], strict = True)
		self.G.cuda().eval()
		self.truncation = 0.7
		self.trunc = self.G.mean_latent(4096).detach().clone()


		print('-- Initialize mask predictor.')
		self.mask_net = nn.ModuleDict({})
		for layer_idx in range(self.num_nets):
			network_name_str = 'network_{:02d}'.format(layer_idx)
			# Net info
			stylespace_dim_layer = self.split_sections[layer_idx]
			
			input_dim = stylespace_dim_layer
			output_dim = stylespace_dim_layer
			inner_dim = stylespace_dim_layer
			network_module = MaskPredictor(input_dim, output_dim, inner_dim = inner_dim)
			self.mask_net.update({network_name_str: network_module})
			out_text = 'Network {}: ----> input_dim:{} - output_dim:{}'.format(layer_idx, input_dim, output_dim)
			print(out_text)
	
	def configure_dataset(self):
		self.test_dataset = CustomDataset_validation(synthetic_dataset_path = self.synthetic_dataset_path, validation_pairs = self.validation_pairs)	
		
		self.test_dataloader = DataLoader(self.test_dataset,
									batch_size=self.test_batch_size ,
									shuffle=False,
									num_workers=int(self.workers),
									drop_last=True)

	def start_from_checkpoint(self):
		step = 0
		if self.resume_training_model is not None: 
			if os.path.isfile(self.resume_training_model):
				print('Resuming training from {}'.format(self.resume_training_model))
				state_dict = torch.load(self.resume_training_model)
				if 'step' in state_dict:
					step = state_dict['step']
				if 'mask_net'  in state_dict:
					self.mask_net.load_state_dict(state_dict['mask_net'])	
		return step

	def get_shifted_image(self, style_source, style_target, w, noise):
		# Generate shift
		masks_per_layer = []
		for layer_idx in range(self.num_nets):
			network_name_str = 'network_{:02d}'.format(layer_idx)
			style_source_idx = style_source[layer_idx]
			style_target_idx = style_target[layer_idx]				
			styles = style_source_idx - style_target_idx
			mask_idx = self.mask_net[network_name_str](styles)
			masks_per_layer.append(mask_idx)

		style_source = torch.cat(style_source, dim=1)
		style_target = torch.cat(style_target, dim=1)
		mask = torch.cat(masks_per_layer, dim=1)
		new_style_space = generate_new_stylespace(style_source, style_target, mask, self.num_layers_control)
		
		new_style_space = list(torch.split(tensor=new_style_space, split_size_or_sections=self.split_sections, dim=1))
		imgs_shifted = decoder(self.G, new_style_space, w, noise, resize_image = True)
		
		return imgs_shifted, new_style_space

	def train(self):

		self.load_models()
		if self.use_wandb:
			#########################
			config = self.args
			wandb.init(
				project= self.project_wandb,
				notes="",
				tags=["debug"],
				config=config,
			)
			name = self.output_path.split('/')[-1]
			wandb.run.name = name
			wandb.watch(self.mask_net, log="all", log_freq=500)
			#######################
		self.configure_dataset()
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

		self.G.cuda().eval()
		self.mask_net.train().cuda() 
		optimizer = torch.optim.Adam(self.mask_net.parameters(), lr=self.lr, weight_decay=5e-4)

		self.truncation = 0.7
		latent_in = torch.randn(4096, 512).cuda()
		self.trunc = self.G.style(latent_in).mean(0, keepdim=True)
		input_is_latent = False

		recovered_step = self.start_from_checkpoint()
		if recovered_step != 0:
			print('Resume training from {}'.format(recovered_step))
		

		list_loss = []
		for step in range(recovered_step, self.max_iter):

			loss_dict = {}
			self.G.zero_grad()
			source_z = make_noise(self.batch_size, 512, None).cuda()				
			target_z = make_noise(self.batch_size, 512, None).cuda()	

			with torch.no_grad():
				######## Source images ########
				imgs_source, style_source, w_source, noise_source = generate_image(self.G, source_z, self.truncation, self.trunc, self.image_resolution, self.split_sections,
					input_is_latent = input_is_latent, return_latents= True, resize_image = True)
				params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
			
				######## Target images	 ########
				imgs_target, style_target, w_target, noise_target = generate_image(self.G, target_z, self.truncation, self.trunc, self.image_resolution, self.split_sections,
					input_is_latent = input_is_latent, return_latents= True, resize_image = True)
				params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)
				
			######## Generate reenacted image between source and target images ########
			imgs_shifted, new_style_space = self.get_shifted_image(style_source, style_target, w_source, noise_source)	
			params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)
			
				
			loss, loss_dict = self.calculate_loss(params_source, params_shifted, params_target, imgs_source, imgs_shifted)

			if self.use_recurrent_cycle_loss:
				########## Recurrent Cycle loss ##########
				with torch.no_grad():
					### Generate a new random target image ###
					target_z_cycle = make_noise(self.batch_size, 512, None).cuda()	
					imgs_target_cycle, style_target_cycle, w_target_cycle, noise_target_cycle = generate_image(self.G, target_z_cycle, self.truncation, self.trunc, self.image_resolution, self.split_sections,
						input_is_latent = input_is_latent, return_latents= True, resize_image = True)
					params_target_cycle, angles_target_cycle = calculate_shapemodel(self.deca, imgs_target_cycle)
				
				#### Reenact source image into the facial pose of target_z_cycle ####
				imgs_shifted_hat, new_style_space_hat = self.get_shifted_image(style_source, style_target_cycle, w_source, noise_source)
				params_shifted_hat, angles_shifted_hat = calculate_shapemodel(self.deca, imgs_shifted_hat)
				#####################################################################

				#### Reenact initial shifted image into the facial pose of target_z_cycle ####
				imgs_shifted_hat_2, new_style_space_hat_2 = self.get_shifted_image(new_style_space, style_target_cycle, w_source, noise_source)
				params_shifted_hat_2, angles_shifted_hat_2 = calculate_shapemodel(self.deca, imgs_shifted_hat_2)

				loss_cycle, loss_dict = self.calculate_recurrent_loss(params_source, params_target_cycle, params_shifted_hat, 
																	  params_shifted_hat_2, imgs_source, imgs_shifted_hat, imgs_shifted_hat_2, loss_dict)
				loss += loss_cycle

			############## Total loss ##############	
			list_loss.append(loss.data.item())
			self.mask_net.zero_grad()
			loss.backward()
			optimizer.step()
			

			######### Evaluate #########
			if step % self.steps_per_log == 0:
				out_text = '[step {}]'.format(step)
				for key, value in loss_dict.items():
					out_text += (' | {}: {:.2f}'.format(key, value))
				out_text += '| Mean Loss {:.2f}'.format(np.mean(np.array(list_loss)))
				print(out_text)
			
			if step % self.steps_per_save_models == 0 and step > 0:
				self.save_model(step)

			if step % self.steps_per_evaluation == 0:
				self.evaluate_model_reenactment(step)

			if step % 500 == 0 and step > 0:
				list_loss = []

			if self.use_wandb:
				wandb.log({
					'step': step,
				})
				wandb.log(loss_dict)

	def calculate_loss(self, params_source, params_shifted, params_target, imgs_source, imgs_shifted):
		loss_dict = {} 
		loss = 0
		
		############## Shape Loss ##############
		if self.lambda_shape !=0:

			coefficients_gt = {}	
			coefficients_gt['pose'] = params_target['pose']
			coefficients_gt['exp'] = params_target['alpha_exp']	
			coefficients_gt['cam'] = params_source['cam']
			coefficients_gt['cam'][:,:] = 0.
			coefficients_gt['cam'][:,0] = 8
			coefficients_gt['shape'] = params_source['alpha_shp']
			landmarks2d_gt, landmarks3d_gt, shape_gt = self.deca.calculate_shape(coefficients_gt)

			coefficients_reen = {}
			coefficients_reen['pose'] = params_shifted['pose']
			coefficients_reen['shape'] = params_shifted['alpha_shp']
			coefficients_reen['exp'] = params_shifted['alpha_exp']
			coefficients_reen['cam'] = params_shifted['cam']
			coefficients_reen['cam'][:,:] = 0.
			coefficients_reen['cam'][:,0] = 8
			landmarks2d_reenacted, landmarks3d_reenacted, shape_reenacted = self.deca.calculate_shape(coefficients_reen)
			
			loss_shape = self.lambda_shape *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth = self.lambda_shape *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye = self.lambda_shape * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)
			
			loss_dict['loss_shape'] = loss_shape.data.item()
			loss_dict['loss_eye'] = loss_eye.data.item()
			loss_dict['loss_mouth'] = loss_mouth.data.item()

			loss += loss_mouth
			loss += loss_shape
			loss += loss_eye
		####################################################

		############## Identity losses ##############	
		if self.lambda_identity != 0:
			loss_identity = self.lambda_identity * self.id_loss_(imgs_shifted, imgs_source.detach())
			loss_dict['loss_identity'] = loss_identity.data.item()
			loss += loss_identity

		if self.lambda_perceptual != 0:
			imgs_source_255 = tensor_to_255(imgs_source)
			imgs_shifted_255 = tensor_to_255(imgs_shifted)
			loss_perceptual = self.lambda_perceptual * self.lpips_loss(imgs_shifted_255, imgs_source_255.detach())
			loss_dict['loss_perceptual'] = loss_perceptual.data.item()
			loss += loss_perceptual

		loss_dict['loss'] = loss.data.item()
		return loss, loss_dict

	def calculate_recurrent_loss(self, params_source, params_target_cycle, params_shifted_hat, params_shifted_hat_2, imgs_source, imgs_shifted_hat, imgs_shifted_hat_2, loss_dict):

		loss = 0
		############## Shape Loss ##############
		if self.lambda_shape > 0:
			# 1
			coefficients_gt = {}
			coefficients_gt['pose'] = params_target_cycle['pose']
			coefficients_gt['exp'] = params_target_cycle['alpha_exp']	
			coefficients_gt['cam'] = params_source['cam']
			coefficients_gt['cam'][:,:] = 0.
			coefficients_gt['cam'][:,0] = 8
			coefficients_gt['shape'] = params_source['alpha_shp']
			landmarks2d_gt, landmarks3d_gt, shape_gt = self.deca.calculate_shape(coefficients_gt)

			coefficients_reen = {}
			coefficients_reen['pose'] = params_shifted_hat['pose']
			coefficients_reen['shape'] = params_shifted_hat['alpha_shp']
			coefficients_reen['exp'] = params_shifted_hat['alpha_exp']
			coefficients_reen['cam'] = params_shifted_hat['cam']
			coefficients_reen['cam'][:,:] = 0.
			coefficients_reen['cam'][:,0] = 8
			landmarks2d_reenacted, landmarks3d_reenacted, shape_reenacted = self.deca.calculate_shape(coefficients_reen)
			

			loss_shape = self.lambda_shape *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth = self.lambda_shape *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye = self.lambda_shape * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)

			# 2
			coefficients_gt = {}
			coefficients_gt['pose'] = params_target_cycle['pose']
			coefficients_gt['exp'] = params_target_cycle['alpha_exp']	
			coefficients_gt['cam'] = params_source['cam']
			coefficients_gt['cam'][:,:] = 0.
			coefficients_gt['cam'][:,0] = 8
			coefficients_gt['shape'] = params_source['alpha_shp']
			landmarks2d_gt, landmarks3d_gt, shape_gt = self.deca.calculate_shape(coefficients_gt)

			coefficients_reen = {}
			coefficients_reen['pose'] = params_shifted_hat_2['pose']
			coefficients_reen['shape'] = params_shifted_hat_2['alpha_shp']
			coefficients_reen['exp'] = params_shifted_hat_2['alpha_exp']
			coefficients_reen['cam'] = params_shifted_hat_2['cam']
			coefficients_reen['cam'][:,:] = 0.
			coefficients_reen['cam'][:,0] = 8
			landmarks2d_reenacted, landmarks3d_reenacted, shape_reenacted = self.deca.calculate_shape(coefficients_reen)
			

			loss_shape += self.lambda_shape *  self.losses.calculate_shape_loss(shape_gt, shape_reenacted, normalize = False)
			loss_mouth += self.lambda_shape *  self.losses.calculate_mouth_loss(landmarks2d_gt, landmarks2d_reenacted) 
			loss_eye += self.lambda_shape * self.losses.calculate_eye_loss(landmarks2d_gt, landmarks2d_reenacted)
			
			loss_dict['loss_shape_cycle'] = loss_shape.data.item()
			loss_dict['loss_eye_cycle'] = loss_eye.data.item()
			loss_dict['loss_mouth_cycle'] = loss_mouth.data.item()

			loss += loss_mouth
			loss += loss_shape
			loss += loss_eye

		############## Identity losses ##############	
		if self.lambda_identity != 0:

			loss_identity = self.lambda_identity * self.id_loss_(imgs_shifted_hat, imgs_source.detach())
			loss_identity += self.lambda_identity* self.id_loss_(imgs_shifted_hat_2, imgs_source.detach())
			loss_dict['loss_identity_cycle'] = loss_identity.data.item()
			loss += loss_identity

		if self.lambda_perceptual != 0:
			imgs_shifted_hat_255 = tensor_to_255(imgs_shifted_hat)
			imgs_shifted_hat_2_255 = tensor_to_255(imgs_shifted_hat_2)
			loss_perceptual = self.lambda_perceptual * self.lpips_loss(imgs_shifted_hat_255, imgs_shifted_hat_2_255)
			loss_dict['loss_perceptual_cycle'] = loss_perceptual.data.item()
			loss += loss_perceptual

		loss_dict['loss_cycle'] = loss.data.item()

		return loss, loss_dict

	def save_model(self, step):
		state_dict = {
			'step': 				step,
			'mask_net': 			self.mask_net.state_dict(),
			'num_layers_control': 	self.num_layers_control
		}
		checkpoint_path = os.path.join(self.models_dir, 'mask_net_{:06d}.pt'.format(step))
		torch.save(state_dict, checkpoint_path)

	'Evaluate models for face reenactment and save reenactment figure'
	def evaluate_model_reenactment(self, step):

		input_is_latent = False
		self.mask_net.eval()
		exp_error = 0; pose_error = 0; csim_total = 0; count = 0
		counter_logs = 0

		source_images = torch.zeros((self.num_pairs_log, 3, 256, 256))
		target_images = torch.zeros((self.num_pairs_log, 3, 256, 256))
		reenacted_images = torch.zeros((self.num_pairs_log, 3, 256, 256))
		
		for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):

			with torch.no_grad():
				sample_batch = batch

				source_w = sample_batch['source_w'].cuda()
				target_w = sample_batch['target_w'].cuda()

				imgs_source, style_source, w_source, noise_source = generate_image(self.G, source_w, self.truncation, self.trunc, self.image_resolution, self.split_sections,
					input_is_latent = input_is_latent, return_latents= True, resize_image = True)
				params_source, angles_source = calculate_shapemodel(self.deca, imgs_source)
				
				imgs_target, style_target, w_target, noise_target = generate_image(self.G, target_w, self.truncation, self.trunc, self.image_resolution, self.split_sections,
					input_is_latent = input_is_latent, return_latents= True, resize_image = True)	
				params_target, angles_target = calculate_shapemodel(self.deca, imgs_target)

				imgs_shifted, new_style_space = self.get_shifted_image(style_source, style_target, w_source, noise_source)	
				params_shifted, angles_shifted = calculate_shapemodel(self.deca, imgs_shifted)
			
				csim, pose, exp = calculate_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source, self.id_loss_, self.exp_ranges)
				exp_error += exp
				csim_total += csim
				pose_error += pose
				count += 1

				if counter_logs < self.num_pairs_log:
					if (self.num_pairs_log - counter_logs) % source_w.shape[0] == 0:
						source_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_source.detach().cpu()
						target_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_target.detach().cpu()
						reenacted_images[counter_logs:counter_logs+source_w.shape[0]] = imgs_shifted.detach().cpu()
					else:
						num = self.num_pairs_log - counter_logs
						source_images[counter_logs:counter_logs+num]  = imgs_source[:num].detach().cpu()
						target_images[counter_logs:counter_logs+num]  = imgs_target[:num].detach().cpu()
						reenacted_images[counter_logs:counter_logs+num]  = imgs_shifted[:num].detach().cpu()
					counter_logs += source_w.shape[0]

			
		sample = generate_grid_image(source_images, target_images, reenacted_images)
		save_image(sample, os.path.join(self.log_dir, '{:06d}.png'.format(step)))
		if self.use_wandb and self.log_images_wandb:
			image_array = sample.detach().cpu().numpy()
			image_array = np.transpose(image_array, (1, 2, 0))
			images = wandb.Image(image_array)
			wandb.log({"images": images})

		print('*************** Validation ***************')
		print('Expression Error: {:.4f}\t Pose Error: {:.2f}\t CSIM: {:.2f}'.format(exp_error/count, pose_error/count, csim_total/count))
		print('*************** Validation ***************')
		
		if self.use_wandb:
			wandb.log({
				'expression_error': exp_error/count,
				'pose_error': pose_error/count,
				'csim': csim_total/count,
			})

		self.mask_net.train()
				
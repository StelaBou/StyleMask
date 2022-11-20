import os
import numpy as np
import torch
from torchvision import utils as torch_utils
import cv2
from skimage import io

from libs.utilities.image_utils import read_image_opencv, torch_image_resize
from libs.utilities.ffhq_cropping import align_crop_image

def calculate_evaluation_metrics(params_shifted, params_target, angles_shifted, angles_target, imgs_shifted, imgs_source, id_loss_, exp_ranges):

	
	############ Evaluation ############
	yaw_reenacted = angles_shifted[:,0][0].detach().cpu().numpy() 
	pitch_reenacted = angles_shifted[:,1][0].detach().cpu().numpy() 
	roll_reenacted = angles_shifted[:,2][0].detach().cpu().numpy()
	exp_reenacted = params_shifted['alpha_exp'][0].detach().cpu().numpy() 
	jaw_reenacted = params_shifted['pose'][0, 3].detach().cpu().numpy() 
	
	yaw_target = angles_target[:,0][0].detach().cpu().numpy() 
	pitch_target = angles_target[:,1][0].detach().cpu().numpy() 
	roll_target = angles_target[:,2][0].detach().cpu().numpy()
	exp_target = params_target['alpha_exp'][0].detach().cpu().numpy() 
	jaw_target = params_target['pose'][0, 3].detach().cpu().numpy()

	exp_error = []	
	num_expressions = 20
	max_range = exp_ranges[3][1]
	min_range = exp_ranges[3][0]		
	jaw_target = (jaw_target - min_range)/(max_range-min_range)
	jaw_reenacted = (jaw_reenacted - min_range)/(max_range-min_range)
	exp_error.append(abs(jaw_reenacted - jaw_target))			
			
	for j  in range(num_expressions):
		max_range = exp_ranges[j+4][1]
		min_range = exp_ranges[j+4][0]
		target = (exp_target[j] - min_range)/(max_range-min_range)
		reenacted = (exp_reenacted[j] - min_range)/(max_range-min_range)
		exp_error.append(abs(reenacted - target) )
	exp_error = np.mean(exp_error)

	## normalize exp coef in [0,1]
	# exp_error = []	
	# num_expressions = 12	 # len(exp_target)
	# for j in range(num_expressions):
	# 	exp_error.append(abs(exp_reenacted[j] - exp_target[j]) )
	# exp_error.append(abs(jaw_reenacted - jaw_target))	
	# exp_error = np.mean(exp_error)
	
	pose = (abs(yaw_reenacted-yaw_target) + abs(pitch_reenacted-pitch_target) + abs(roll_reenacted-roll_target))/3
	################################################

	###### CSIM ######
	loss_identity = id_loss_(imgs_shifted, imgs_source) 
	csim = 1 - loss_identity.data.item()

	return csim, pose, exp_error

def generate_grid_image(source, target, reenacted):
	num_images = source.shape[0] # batch size
	width = 256; height = 256
	grid_image = torch.zeros((3, num_images*height, 3*width))
	for i in range(num_images):
		s = i*height
		e = s + height
		grid_image[:, s:e, :width] = source[i, :, :, :]
		grid_image[:, s:e, width:2*width] = target[i, :, :, :]	
		grid_image[:, s:e, 2*width:] = reenacted[i, :, :, :]
	
	if grid_image.shape[1] > 1000: # height
		grid_image = torch_image_resize(grid_image, height = 800)
	return grid_image

" Crop images using facial landmarks like FFHQ "
def preprocess_image(image_path, landmarks_est, save_filename = None):

	image = read_image_opencv(image_path)
	landmarks = landmarks_est.get_landmarks(image)[0]
	landmarks = np.asarray(landmarks)
	
	img = align_crop_image(image, landmarks)
	
	if img is not None and save_filename is not None:
		cv2.imwrite(save_filename, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))
	if img is not None:
		return img
	else:
		print('Error with image preprocessing')
		exit()

" Invert real image into the latent space of StyleGAN2 "
def invert_image(image, encoder, generator, truncation, trunc, save_path = None, save_name = None):
	with torch.no_grad():
		latent_codes = encoder(image)
		inverted_images, _ = generator([latent_codes], input_is_latent=True, return_latents = False, truncation= truncation, truncation_latent=trunc)

	if save_path is not None and save_name is not None:
		grid = torch_utils.save_image(
						inverted_images,
						os.path.join(save_path, '{}.png'.format(save_name)),
						normalize=True,
						range=(-1, 1),
					)
		# Latent code
		latent_code = latent_codes[0].detach().cpu().numpy()
		save_dir = os.path.join(save_path, '{}.npy'.format(save_name))
		np.save(save_dir, latent_code)

	return inverted_images, latent_codes
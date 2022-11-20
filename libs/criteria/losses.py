import torch
import numpy as np

"""
Calculate shape losses
"""

class Losses():
	def __init__(self):   
		self.criterion_mse = torch.nn.MSELoss()
		self.criterion_l1 = torch.nn.L1Loss()
		self.image_deca_size = 224

	def calculate_pixel_wise_loss(self, images_shifted, images):

		pixel_wise_loss = self.criterion_l1(images, images_shifted) 

		return pixel_wise_loss

	def calculate_shape_loss(self, shape_gt, shape_reenacted, normalize = False):
		criterion_l1 = torch.nn.L1Loss()
		if normalize:
			shape_gt_norm = shape_gt/200 #self.image_deca_size
			shape_reenacted_norm = shape_reenacted/200 #self.image_deca_size
			loss = criterion_l1(shape_gt_norm, shape_reenacted_norm)
		else:
			loss = criterion_l1(shape_gt, shape_reenacted)
		return loss

	def calculate_eye_loss(self, shape_gt, shape_reenacted):
		criterion_l1 = torch.nn.L1Loss()
		shape_gt_norm = shape_gt.clone()
		shape_reenacted_norm = shape_reenacted.clone()
		# shape_gt_norm = shape_gt_norm/self.image_deca_size
		# shape_reenacted_norm = shape_reenacted_norm/self.image_deca_size
		eye_pairs = [(36, 39), (37, 41), (38, 40), (42, 45), (43, 47), (44, 46)]
		loss = 0
		for i in range(len(eye_pairs)):
			pair = eye_pairs[i]
			d_gt = abs(shape_gt[:, pair[0],:] - shape_gt[:, pair[1],:])
			d_e = abs(shape_reenacted[:, pair[0],:] - shape_reenacted[:, pair[1],:])
			loss += criterion_l1(d_gt, d_e)
		
		loss = loss/len(eye_pairs)
		return loss
	
	def calculate_mouth_loss(self, shape_gt, shape_reenacted):
		criterion_l1 = torch.nn.L1Loss()
		shape_gt_norm = shape_gt.clone()
		shape_reenacted_norm = shape_reenacted.clone()
		# shape_gt_norm = shape_gt_norm/self.image_deca_size
		# shape_reenacted_norm = shape_reenacted_norm/self.image_deca_size
		mouth_pairs = [(48, 54), (49, 59), (50, 58), (51, 57), (52, 56), (53, 55), (60, 64), (61, 67), (62, 66), (63, 65)]
		loss = 0
		for i in range(len(mouth_pairs)):
			pair = mouth_pairs[i]
			d_gt = abs(shape_gt[:, pair[0],:] - shape_gt[:, pair[1],:])
			d_e = abs(shape_reenacted[:, pair[0],:] - shape_reenacted[:, pair[1],:])
			loss += criterion_l1(d_gt, d_e)

		loss = loss/len(mouth_pairs)
		return loss
		
	
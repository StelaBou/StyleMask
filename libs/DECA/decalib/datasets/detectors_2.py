"""
Calculate euler angles yaw pitch roll using deep network HopeNet
https://github.com/natanielruiz/deep-head-pose

The face detector used is SFD (taken from face-alignment FAN) https://github.com/1adrianb/face-alignment

"""
import os 
import numpy as np
import sys
from matplotlib import pyplot as plt
import cv2
from enum import Enum

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.model_zoo import load_url

from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image

# from .image_utils import imshow, imshow_nparray, image_resize
# from .visualization import print_values , draw_detected_face

from libs.pose_estimation.sfd.sfd_detector import SFDDetector as FaceDetector
from libs.pose_estimation.fan_model.models import FAN, ResNetDepth
from libs.pose_estimation.fan_model.utils import *


class LandmarksType(Enum):
	"""Enum class defining the type of landmarks to detect.

	``_2D`` - the detected points ``(x,y)`` are detected in a 2D space and follow the visible contour of the face
	``_2halfD`` - this points represent the projection of the 3D points into 3D
	``_3D`` - detect the points ``(x,y,z)``` in a 3D space

	"""
	_2D = 1
	_2halfD = 2
	_3D = 3


class NetworkSize(Enum):
	# TINY = 1
	# SMALL = 2
	# MEDIUM = 3
	LARGE = 4

	def __new__(cls, value):
		member = object.__new__(cls)
		member._value_ = value
		return member

	def __int__(self):
		return self.value

models_urls = {
	'2DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/2DFAN4-11f355bf06.pth.tar',
	'3DFAN-4': 'https://www.adrianbulat.com/downloads/python-fan/3DFAN4-7835d9f11d.pth.tar',
	'depth': 'https://www.adrianbulat.com/downloads/python-fan/depth-2a464da4ea.pth.tar',
}

def get_preds_fromhm(hm, center=None, scale=None):
	"""Obtain (x,y) coordinates given a set of N heatmaps. If the center
	and the scale is provided the function will return the points also in
	the original coordinate frame.

	Arguments:
		hm {torch.tensor} -- the predicted heatmaps, of shape [B, N, W, H]

	Keyword Arguments:
		center {torch.tensor} -- the center of the bounding box (default: {None})
		scale {float} -- face scale (default: {None})
	"""
	max, idx = torch.max(
		hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
	idx = idx + 1
	preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
	preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
	preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

	for i in range(preds.size(0)):
		for j in range(preds.size(1)):
			hm_ = hm[i, j, :]
			pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
			if pX > 0 and pX < 63 and pY > 0 and pY < 63:
				diff = torch.FloatTensor(
					[hm_[pY, pX + 1] - hm_[pY, pX - 1],
					 hm_[pY + 1, pX] - hm_[pY - 1, pX]])
				preds[i, j].add_(diff.sign_().mul_(.25))

	preds.add_(-.5)

	preds_orig = torch.zeros(preds.size())
	if center is not None and scale is not None:
		for i in range(hm.size(0)):
			for j in range(hm.size(1)):
				preds_orig[i, j] = transform(
					preds[i, j], center, scale, hm.size(2), True)

	return preds, preds_orig

def draw_detected_face(img, face):
	# for i, d in enumerate(face):
	x_min = int(face[0])
	y_min = int(face[1])
	x_max = int(face[2])
	y_max = int(face[3])
		# # print(x_min,y_min,x_max,y_max)
		# bbox_width = abs(x_max - x_min)
		# bbox_height = abs(y_max - y_min)
		# x_min -= 50
		# x_max += 50
		# y_min -= 50
		# y_max += 30
		# x_min = max(x_min, 0)
		# y_min = max(y_min, 0)
		# # print(img.shape)
		# x_max = min(img.shape[1], x_max) 
		# y_max = min(img.shape[0], y_max)
		
		# Crop image
		# img = image[:, :, y_min:y_max, x_min:x_max]
		# print(x_min,y_min,x_max,y_max)
		# img = img[int(y_min):int(y_max),int(x_min):int(x_max)]
	cv2.rectangle(img, (int(x_min),int(y_min)), (int(x_max),int(y_max)), (255,0,0), 2)
	
	return img
	
from os.path import abspath, dirname
current_file_directory = dirname(abspath(__file__))

class LandmarksEstimation():

	def __init__(self):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
		# Load all needed models - Face detector and Pose detector

		network_size = NetworkSize.LARGE
		network_size = int(network_size)	
		self.landmarks_type = LandmarksType._2D
		self.flip_input = False
		# SFD face detection
		path_to_detector = './libs/pose_estimation/sfd/model/s3fd-619a316812.pth'
		if not os.path.exists(path_to_detector):
			'Search on scratch'
			path_to_detector = '../../../scratch/k2033759/Finding_directions/pretrained_models/s3fd-619a316812.pth'

		face_detector = 'sfd'
		self.face_detector = FaceDetector(device='cuda', verbose=False,path_to_detector = path_to_detector)
		

		self.transformations_image = transforms.Compose([transforms.Resize(224),
								transforms.CenterCrop(224), transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
		self.transformations = transforms.Compose([transforms.Resize(224),
								transforms.CenterCrop(224),
								transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

		# Initialise the face alignemnt networks
		self.face_alignment_net = FAN(network_size)
		network_name = '2DFAN-' + str(network_size)
		fan_weights = load_url(models_urls[network_name], map_location=lambda storage, loc: storage)
		self.face_alignment_net.load_state_dict(fan_weights)
		self.face_alignment_net.to(self.device)
		self.face_alignment_net.eval()

	def detect_landmarks_torch(self, images):
		"""
		images: torch Tensor B x C x W x H
		detected_faces: B X 1 x 5
		"""
		detected_faces, error, error_index = self.face_detector.detect_from_batch(images)
		
		faces = []
		for i in range(images.shape[0]):
			box = detected_faces[i]
			if len(box) > 1:
				max_conf = -1
				max_ind = -1
				for j in range(len(box)):
					conf = box[j][4]
					if conf > max_conf:
						max_conf = conf
						max_ind = j
				box_new = box[max_ind]
				box = box_new
				faces.append(box)
			else:
				faces.append(box[0])

		faces = np.asarray(faces)
		bboxes = []
		for i in range(faces.shape[0]):
			kpt = self.find_landmarks_torch(faces[i], images[i])
			kpt = kpt[0].detach().cpu().numpy()
			left = np.min(kpt[:,0])
			right = np.max(kpt[:,0])
			top = np.min(kpt[:,1])
			bottom = np.max(kpt[:,1])
			bbox = [left, top, right, bottom]
			bboxes.append(bbox)

		return bboxes, 'kpt68'


	def find_landmarks_torch(self, face, image):

		center = torch.FloatTensor(
			[(face[2] + face[0]) / 2.0,
				(face[3] + face[1]) / 2.0])

		center[1] = center[1] - (face[3] - face[1]) * 0.12
		scale = (face[2] - face[0] + face[3] - face[1]) / self.face_detector.reference_scale
		
		inp = crop_torch(image.unsqueeze(0), center, scale).float().cuda()

		# print(inp.shape)
		# imshow(inp.squeeze(0))

		inp = inp.div(255.0)
		

		out = self.face_alignment_net(inp)[-1]

		
		if self.flip_input:
			out = out + flip(self.face_alignment_net(flip(inp))
						[-1],  is_label=True)  # patched inp_batch undefined variable error
		out = out.cpu()

		pts, pts_img = get_preds_fromhm(out, center, scale)
		
		
		pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)
	
		return pts_img


	def find_landmarks(self, face, image):

		# face = face[0]

		center = torch.FloatTensor(
			[(face[2] + face[0]) / 2.0,
				(face[3] + face[1]) / 2.0])

		center[1] = center[1] - (face[3] - face[1]) * 0.12
		scale = (face[2] - face[0] + face[3] - face[1]) / self.face_detector.reference_scale
		
		inp = crop_torch(image.unsqueeze(0), center, scale).float().cuda()

		# print(inp.shape)
		# imshow(inp.squeeze(0))

		inp = inp.div(255.0)
		

		out = self.face_alignment_net(inp)[-1]

		
		if self.flip_input:
			out = out + flip(self.face_alignment_net(flip(inp))
						[-1],  is_label=True)  # patched inp_batch undefined variable error
		out = out.cpu()

		pts, pts_img = get_preds_fromhm(out, center, scale)
		out = out.cuda()
		
		# Added 3D landmark support
		if self.landmarks_type == LandmarksType._3D:
			pts, pts_img = pts.view(68, 2) * 4, pts_img.view(68, 2)
			heatmaps = torch.zeros((68,256,256), dtype=torch.float32)
			for i in range(68):
				if pts[i, 0] > 0:
					heatmaps[i] = draw_gaussian(
						heatmaps[i], pts[i], 2)
		
			heatmaps = heatmaps.unsqueeze(0)
			
			heatmaps = heatmaps.to(self.device)
			if inp.shape[2] != heatmaps.shape[2] or inp.shape[3] != heatmaps.shape[3]:
				print(inp.shape)
				print(heatmaps.shape)

			depth_pred = self.depth_prediciton_net(
				torch.cat((inp, heatmaps), 1)).view(68, 1)  #.data.cpu().view(68, 1)
			# print(depth_pred.view(68, 1).shape)
			pts_img = pts_img.cuda()
			pts_img = torch.cat(
				(pts_img, depth_pred * (1.0 / (256.0 / (200.0 * scale)))), 1)

			
		else:
			pts, pts_img = pts.view(-1, 68, 2) * 4, pts_img.view(-1, 68, 2)
		
		# if pts_img.requires_grad:
		# 	pts_img.register_hook(lambda grad: print('pts_img',grad))
		# 	print(pts_img.requires_grad)
		return pts_img, out


	def face_detection(self, image, save_path, image_path):
		image_tensor = torch.tensor(np.transpose(image,(2,0,1))).float().cuda()
		if len(image_tensor.shape) == 3:
			image_tensor = image_tensor.unsqueeze(0).cuda()
			detected_faces,error,error_index = self.face_detector.detect_from_batch(image_tensor)
		else:
			detected_faces,error,error_index = self.face_detector.detect_from_batch(image_tensor)

		faces_num = 0
		if len(detected_faces[0]) == 0:
			return image
		for face in detected_faces[0]:
			conf = face[4]
			# print('Conf {:.2f}'.format(conf))
			if conf > 0.9:
				x1 = face[0]
				y1 = face[1]
				x2 = face[2]
				y2 = face[3]
				w = x2-x1
				h = y2-y1
			
				cx = int(x1+w/2)
				cy = int(y1+h/2)

				if h>w:
					w = h
					x1_hat = cx - int(w/2)
					if x1_hat < 0:
						x1_hat = 0
					x2_hat = x1_hat + w

				else:
					h = w
					y1_hat = cy - int(h/2)
					if y1_hat < 0:
						y1_hat = 0
					y2_hat = y1_hat + h

				# print(int(w), int(h))
				# quit()
				# w = 100
				# h = 100
				w_hat = int(w*1.6)
				h_hat = int(h*1.6)
				x1_hat = cx - int(w_hat/2)
				if x1_hat < 0:
					x1_hat = 0
				y1_hat = cy - int(h_hat/2)
				if y1_hat < 0:
					y1_hat = 0
				x2_hat = x1_hat + w_hat
				y2_hat = y1_hat + h_hat
				crop = image.copy()
				# print(y1_hat, y2_hat, x1_hat, x2_hat)
				crop = crop[ y1_hat:y2_hat, x1_hat:x2_hat]

				# print(w_hat, h_hat)
				crop, scale = image_resize(crop, 256, 256)

				# x2 = x1 + w
		# y2 = y1 + h
		# cx = int(x1+w/2)
		# cy = int(y1+h/2)
		# w_hat = int(w*1.6)
		# h_hat = int(h*1.6)
		# x1_hat = cx - int(w_hat/2)
		# if x1_hat < 0:
		# 	x1_hat = 0
		# y1_hat = cy - int(h_hat/2)
		# if y1_hat < 0:
		# 	y1_hat = 0
		# x2_hat = x1_hat + w_hat
		# y2_hat = y1_hat + h_hat
		# crop = image[ y1_hat:y2_hat, x1_hat:x2_hat]
		# # cv2.imwrite('./test.png', cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR))
		# crop, scale  = image_resize(crop , resize_, resize_)

				# print(scale)
				# img = draw_detected_face(image, face)
				# image_name = image_path.split('/')[-1]
				# filename = os.path.join(save_path, 'cropped_' +image_name)
				# cv2.imwrite(filename, cv2.cvtColor(crop.copy(), cv2.COLOR_RGB2BGR))
				# filename_2 = os.path.join(save_path, 'face_' + image_name)
				# # img, scale = image_resize(image, 256)
				# cv2.imwrite(filename_2, cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR))

		
		return crop
	
	@torch.no_grad()		
	def detect_landmarks(self, image, detected_faces = None, draw_face = False):
		twoface = False
		# image.register_hook(lambda grad: print('images',grad))
		if detected_faces is None:
			if len(image.shape) == 3:
				image = image.unsqueeze(0).cuda()
				detected_faces,error,error_index = self.face_detector.detect_from_batch(image)
			else:
				detected_faces,error,error_index = self.face_detector.detect_from_batch(image)
		
		twoface = False
		
		batch = 0
		num_faces = 0
		em_max = -1
		index_face = 0
		for face in detected_faces[0]:
			conf = face[4]
			w = face[2] - face[0]
			h = face[3] - face[1]
			em = w*h
			if em>em_max:
				em_max = em
				index_face = num_faces
			# print(face)
			# print(w*h)
			# print('Conf {:.2f}'.format(conf))
			num_faces += 1

		# # print(num_faces)
		# if num_faces > 1:
		# 	face_final = detected_faces[0]
		# quit()

		size = len(detected_faces[0])
		if self.landmarks_type == LandmarksType._3D:
			landmarks = torch.empty((1, 68, 3), requires_grad=True).cuda()
		else:
			landmarks = torch.empty((1, 68, 2), requires_grad=True).cuda()
		
		counter = 0
		for face in detected_faces[0]:
			# print(face)

		# if len(detected_faces[0]) >1:
		# 	# print(detected_faces)
		# 	# img_np = image.clone()
		# 	# img_np = img_np.squeeze(0)
		# 	# img_np = img_np.detach().cpu().numpy()		
		# 	# img_np = np.transpose(img_np, (1, 2, 0))
		# 	# print(detected_faces)
		# 	# img_face  = draw_detected_face(img_np, detected_faces[0])
		# 	# cv2.imwrite('test_face.png',img_face)

		# 	# img_face  = draw_detected_face(img_np, detected_faces[1])
		# 	# cv2.imwrite('test_face_1.png',img_face)

		# 	# quit()
		# 	twoface = True
		# 	return [], twoface
		# else:
			# if len(detected_faces) == 0:
			# 	print("Warning: No faces were detected.")
			# 	return None

			# # ### Draw detected face
			
			# if draw_face:
			# 	img_np = image.clone()
			# 	img_np = img_np.squeeze(0)
			# 	img_np = img_np.detach().cpu().numpy()		
			# 	img_np = np.transpose(img_np, (1, 2, 0))
			# 	print(detected_faces)
			# 	img_face  = draw_detected_face(img_np, detected_faces[0])
			# 	cv2.imwrite('test_face.png',img_face)
			# 	# print_values(img_face)
			# 	# imshow_nparray(img_face)
			
			# error_flag = []

			conf = face[4]
			
			if conf > 0.99 and counter == index_face:
				# print(index_face)
				# print(face)
				# print('Conf {:.2f}'.format(conf))
				pts_img, heatmaps = self.find_landmarks(face, image[0])
				landmarks[batch] = pts_img.cuda()
				batch += 1

			counter += 1

		if batch > 1:
			twoface = True
		return landmarks, twoface, detected_faces
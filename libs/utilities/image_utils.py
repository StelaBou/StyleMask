import torch
import numpy as np
import cv2
import torchvision
import os
	
" Read image from path"
def read_image_opencv(image_path):
	img = cv2.imread(image_path, cv2.IMREAD_COLOR) # BGR order!!!!
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 

	return img.astype('uint8')

" Load image from file path to tensor [-1,1] range "
def image_to_tensor(image_file):
	max_val = 1
	min_val = -1
	if os.path.isfile(image_file):
		image = cv2.imread(image_file, cv2.IMREAD_COLOR) # BGR order
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype('uint8')
	else:
		image = image_file
	if image.shape[0]>256:
		image, _ = image_resize(image, 256)
	image_tensor = torch.tensor(np.transpose(image,(2,0,1))).float().div(255.0)	
	image_tensor = image_tensor * (max_val - min_val) + min_val

	return image_tensor

def tensor_to_255(image):
	img_tmp = image.clone()
	min_val = -1
	max_val = 1
	img_tmp.clamp_(min=min_val, max=max_val)
	img_tmp.add_(-min_val).div_(max_val - min_val + 1e-5)
	img_tmp = img_tmp.mul(255.0).add(0.0) 
	return img_tmp

def torch_image_resize(image, width = None, height = None):
	dim = None
	(h, w) = image.shape[1:]
	# if both the width and height are None, then return the
	# original image
	if width is None and height is None:
		return image

	# check to see if the width is None
	if width is None:
		# calculate the ratio of the height and construct the
		# dimensions
		r = height / float(h)
		dim = (height, int(w * r))
		scale = r
	# otherwise, the height is None
	else:
		# calculate the ratio of the width and construct the
		# dimensions
		r = width / float(w)
		dim = (int(h * r), width)
		scale = r
	image = image.unsqueeze(0)
	image = torch.nn.functional.interpolate(image, size=dim, mode='bilinear')
	return image.squeeze(0)
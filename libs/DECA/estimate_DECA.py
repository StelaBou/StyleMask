"""
"""

import torch
import numpy as np
import cv2
import os

from .decalib.deca import DECA
from .decalib.datasets import datasets 
from .decalib.utils import util
from .decalib.utils.config import cfg as deca_cfg
from .decalib.utils.rotation_converter import *


class DECA_model():

    def __init__(self, device):
        deca_cfg.model.use_tex = False
        dir_path = os.path.dirname(os.path.realpath(__file__))
        models_path =  os.path.join(dir_path, 'data')
        if not os.path.exists(models_path):
            print('Please download the required data for DECA model. See Readme.')
            exit()
        self.deca = DECA(config = deca_cfg, device=device)
        self.data = datasets.TestData()

    'Batch torch tensor'
    def extract_DECA_params(self, images):
       
        p_tensor = torch.zeros(images.shape[0], 6).cuda()
        alpha_shp_tensor = torch.zeros(images.shape[0], 100).cuda()
        alpha_exp_tensor = torch.zeros(images.shape[0], 50).cuda()
        angles = torch.zeros(images.shape[0], 3).cuda()
        cam = torch.zeros(images.shape[0], 3).cuda()
        for batch in range(images.shape[0]):            
            image_prepro, error_flag = self.data.get_image_tensor(images[batch].clone())
            if not error_flag:
                codedict = self.deca.encode(image_prepro.unsqueeze(0).cuda())
                pose = codedict['pose'][:,:3]
                pose = rad2deg(batch_axis2euler(pose))
                p_tensor[batch] = codedict['pose'][0]
                alpha_shp_tensor[batch] = codedict['shape'][0]
                alpha_exp_tensor[batch] = codedict['exp'][0]
                cam[batch] = codedict['cam'][0]
                angles[batch] = pose
            else:
                angles[batch][0] = -180
                angles[batch][1] = -180
                angles[batch][2] = -180

        return p_tensor, alpha_shp_tensor, alpha_exp_tensor, angles, cam

    def calculate_shape(self, coefficients, image = None, save_path = None, prefix = None):   
        landmarks2d, landmarks3d, points = self.deca.decode(coefficients)
        return landmarks2d, landmarks3d, points

    
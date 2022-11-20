import torch
from torch import nn
from .model_irse import Backbone
import os
import torch.backends.cudnn as cudnn

class IDLoss(nn.Module):
    def __init__(self, pretrained_model_path = './pretrained_models/model_ir_se50.pth'):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace for identity loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        if not os.path.exists(pretrained_model_path):
            print('ir_se50 model does not exist in {}'.format(pretrained_model_path))
            exit()
        self.facenet.load_state_dict(torch.load(pretrained_model_path))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.criterion = nn.CosineSimilarity(dim=1, eps=1e-6)

    def extract_feats(self, x, crop = True):
        if crop:
            x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def forward(self, y_hat, y, crop = True):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y, crop) 
        y_hat_feats = self.extract_feats(y_hat, crop)
        cosine_sim = self.criterion(y_hat_feats, y_feats.detach())
        loss = 1 - cosine_sim
        loss = torch.mean(loss)
        return loss

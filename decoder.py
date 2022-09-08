import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *

class PointPyramidDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(PointPyramidDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.fc1 = nn.Linear(self.latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

    def forward(self, x):
        # get input latent vector of each scale
        x_det = F.relu(self.fc1(x))
        x_sec = F.relu(self.fc2(x_det))
        x_pri = F.relu(self.fc3(x_sec))

        # get prediction of each scale
        

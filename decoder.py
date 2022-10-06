import torch
import torch.nn as nn
import torch.nn.functional as F
from module import *

# ----------------------------------------------------------------------------------------
# hierarchical decoder
class PointPyramidDecoder(nn.Module):
    def __init__(self, latent_dim, final_num_points):
        super(PointPyramidDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.M = final_num_points
        self.M2 = self.M//16
        self.M1 = self.M2//2
        #self.M2 = 256
        #self.M1 = 128

        # to obtain a latent vector for each scale
        self.fc1 = nn.Linear(self.latent_dim, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)

        # to obtain shape for each scale
        # primary
        self.fc_pri = nn.Linear(256, 3*self.M1)
        # secondary
        self.fc_sec = nn.Linear(512, self.M1*self.M2)
        self.conv_sec = nn.Conv1d(self.M2, int(3*self.M2/self.M1), 1)
        # detail(original)
        #self.fc_det = nn.Linear(1024, self.M*self.M2)
        #self.conv1_det = nn.Conv1d(self.M, self.M, 1)
        #self.conv2_det = nn.Conv1d(self.M, 256, 1)
        #self.conv3_det = nn.Conv1d(256, int(3*self.M/self.M2), 1)

        # detail(change to solve error)
        self.fc_det = nn.Linear(1024, self.M2)
        self.conv1_det = nn.Conv1d(1, 64, 1)
        self.conv2_det = nn.Conv1d(64, 256, 1)
        self.conv3_det = nn.Conv1d(256, int(3*self.M/self.M2), 1)


    def forward(self, x):
        """decoder to get completion result

        Args:
            x (tensor): feature vector (B, latent_dim)

        Returns:
            3 resolution result: (B, 3, num_pri), (B, 3, num_sec), (B, 3, num_det)
        """
        # get input latent vector of each scale
        x_det = F.relu(self.fc1(x)) # (B, 1024)
        x_sec = F.relu(self.fc2(x_det)) # (B, 512)
        x_pri = F.relu(self.fc3(x_sec)) # (B, 256)

        #out = self.conv_test(x_det)


        # get prediction of each scale
        # primary
        x_pri = self.fc_pri(x_pri) # (B, 3*self.M)
        out_pri = x_pri.view(-1, self.M1, 3)
        out_pri_expnad = torch.unsqueeze(out_pri, dim=2)

        # secondary
        x_sec = F.relu(self.fc_sec(x_sec)) # (B, self.M1*self.M2)
        out_sec = x_sec.view(-1, self.M2, self.M1)
        out_sec = self.conv_sec(out_sec)
        out_sec = out_sec.view(-1, self.M1, int(self.M2/self.M1), 3)
        out_sec = out_sec + out_pri_expnad
        out_sec = out_sec.view(-1, self.M2, 3)
        out_sec_expand = torch.unsqueeze(out_sec, dim=2)

        # finary(original)
        #x_det = F.relu(self.fc_det(x_det)) # (B, self.M*self.M2)
        #out_det = x_det.view(-1, self.M, self.M2)
        #out_det = F.relu(self.conv1_det(out_det))
        #out_det = F.relu(self.conv2_det(out_det))
        #out_det = self.conv3_det(out_det)
        #out_det = out_det.view(-1, self.M2, int(self.M/self.M2), 3)
        #out_det = out_det + out_sec_expand
        #out_det = out_det.view(-1, self.M, 3)

        # finary(change to solve error)
        x_det = F.relu(self.fc_det(x_det)) # (B, self.M*self.M2)
        out_det = torch.unsqueeze(x_det, dim=1)
        out_det = F.relu(self.conv1_det(out_det))
        out_det = F.relu(self.conv2_det(out_det))
        out_det = self.conv3_det(out_det)
        out_det = out_det.view(-1, self.M2, int(self.M/self.M2), 3)
        out_det = out_det + out_sec_expand
        out_det = out_det.view(-1, self.M, 3)

        return out_pri, out_sec, out_det

# discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.CMLP = CombinedMLP(3, 448)
        self.fc = nn.Sequential(
            nn.Linear(448, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        """Discriminator

        Args:
            x (tensor): (B, 3, num_points)

        Returns:
            tensor: (B, 0 or 1) 0 and 1 are fake and real, respectively
        """
        x = self.CMLP(x)
        x = x.view(-1, 448)
        out = self.fc(x)

        return out # There is sigmoid in loss, so no need to add sigmoid in layer

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# modules for discriminator
class CombinedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CombinedMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.MLP1 = nn.Sequential(
            SharedMLP(self.in_channels, 64),
            SharedMLP(64, 64)
        )
        self.MLP2 = SharedMLP(64, 128)
        self.MLP3 = SharedMLP(128, 256)
        self.MLP4 = SharedMLP(256, self.out_channels - (64+128))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x_64 = self.MLP1(x)
        x_128 = self.MLP2(x_64)
        x_256 = self.MLP3(x_128)

        feature_64 = torch.max(x_64, dim=2, keepdim=True)[0]
        feature_128 = torch.max(x_128, dim=2, keepdim=True)[0]
        feature_256 = torch.max(x_256, dim=2, keepdim=True)[0]

        feature = torch.concat([feature_64,
                                feature_128,
                                feature_256], dim=1)

        return feature
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# test
if __name__ == "__main__":
    device = "cpu"
    x = torch.randn(10, 1920, device=device)
    decoder = PointPyramidDecoder(latent_dim=1920, final_num_points=12384)
    pri, sec, det = decoder(x)

    print(pri.shape)
    print(sec.shape)
    print(det.shape)

    netD = Discriminator()
    out = netD(det)
    print(out.shape)

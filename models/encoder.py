import torch
import torch.nn as nn
from models.module import SharedMLP
from models.stn import *

class MultiResolutionEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MultiResolutionEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.CMLP_det = CombinedMLP(3, self.latent_dim)
        self.CMLP = nn.ModuleList([CombinedMLP(3, self.latent_dim) for _ in range(2)])
        self.MLP = SharedMLP(3, 1)

    def forward(self, x):
        """Encoder

        Args:
            x (list): [xyz_pri, xyz_sec, xyz_det]
        Returns:
            tensor: latent vector (B, latent_dim)
        """
        x_det = x[2]
        # feature, trans_3d = self.CMLP_det(x_det, 2)
        feature = self.CMLP_det(x_det, 1)
        # trans_PriSec = trans_3d.clone().detach()
        features = [feature]
        for i in range(2):
            x_PriSec = x[i].permute(0, 2, 1)

            # apply stn got in det CMLP
            # trans_x = torch.bmm(x_PriSec, trans_PriSec)
            # trans_x = trans_x.permute(0, 2, 1)
            # feature = self.CMLP[i](trans_x, i) # (B, latent_dim, 1)

            feature = self.CMLP[i](x_PriSec, i) # (B, latent_dim, 1)
            features.append(feature)

        x = torch.cat(features, dim=2) # (B, latent_dim, 3)
        x = x.transpose(1, 2).contiguous() # (B, 3, latent_dim)
        encode_result = self.MLP(x) # (B, 1, latent_dim)
        encode_result = encode_result.view(-1, self.latent_dim)

        # return encode_result, trans_PriSec
        return encode_result


class CombinedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CombinedMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.STN3d = STNkd(3)
        self.MLP1 = nn.Sequential(
            SharedMLP(self.in_channels, 64),
            SharedMLP(64, 64),
            SharedMLP(64, 128)
        )
        self.MLP2 = SharedMLP(128, 256)
        self.MLP3 = SharedMLP(256, 512)
        self.MLP4 = SharedMLP(512, self.out_channels - (128+256+512))

    def forward(self, x, id):
        """Combined Shared MLP

        Args:
            x (tensor): (B, C, N)
        Returns:
            tensor: feature vector(B, latent_dim, 1)
        """
        if id == 2:
            # apply first stn
            trans_3d = self.STN3d(x)
            x = x.permute(0, 2, 1)
            x = torch.bmm(x, trans_3d)
            x = x.permute(0, 2, 1)

        x_128 = self.MLP1(x)
        x_256 = self.MLP2(x_128)
        x_512 = self.MLP3(x_256)
        x_1024 = self.MLP4(x_512)

        feature_128 = torch.max(x_128, dim=2, keepdim=True)[0]
        feature_256 = torch.max(x_256, dim=2, keepdim=True)[0]
        feature_512 = torch.max(x_512, dim=2, keepdim=True)[0]
        feature_1024 = torch.max(x_1024, dim=2, keepdim=True)[0]

        feature = torch.cat([feature_128,
                             feature_256,
                             feature_512,
                             feature_1024], dim=1)
        if id == 2:
            return feature, trans_3d
        else:
            return feature

if __name__=="__main__":
    device = "cuda"
    x_pri = torch.randn(10, 3, 400, device=device)
    x_sec = torch.randn(10, 3, 800, device=device)
    x_det = torch.randn(10, 3, 4000, device=device)
    x = [x_pri, x_sec, x_det]
    encoder = MultiResolutionEncoder(latent_dim=1920).to(device)
    out, trans = encoder(x)

    print(out.shape)
    print(trans.shape)

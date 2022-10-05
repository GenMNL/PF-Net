import torch
import torch.nn as nn
from module import *
from encoder import *
from decoder import *

class PFNet(nn.Module):
    def __init__(self, latent_dim, final_num_points):
        super(PFNet, self).__init__()
        self.latent_dim = latent_dim
        self.final_num_points = final_num_points
        self.Encoder = MultiResolutionEncoder(self.latent_dim)
        self.Decoder = PointPyramidDecoder(self.latent_dim, self.final_num_points)

    def forward(self, x):
        """full model that contain encoder and decoder

        Args:
            x (list): (x_pri, x_sec, x_det)

        Returns:
            3 resolution result: (B, 3, num_pri), (B, 3, num_sec), (B, 3, num_det)
        """
        feature_v = self.Encoder(x)
        out_pri, out_sec, out_det = self.Decoder(feature_v)

        return out_pri, out_sec, out_det


# ----------------------------------------------------------------------------------------
if __name__=="__main__":
    device = 'cuda'
    x_det = torch.randn(10, 3, 4000, device=device)
    x_sec = torch.randn(10, 3, 800, device=device)
    x_pri = torch.randn(10, 3, 400, device=device)
    x = [x_pri, x_sec, x_det]

    model = PFNet(1924, 12384).to(device)
    out_pri, out_sec, out_det= model(x)
    print(out_det.shape)

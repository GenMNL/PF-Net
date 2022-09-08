import torch
import torch.nn as nn
from module import SharedMLP

class MultiResolutionEncoder(nn.Module):
    def __init__(self, latent_dim):
        super(MultiResolutionEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.CMLP = nn.ModuleList([CombinedMLP(3, self.latent_dim) for _ in range(3)])
        self.MLP = SharedMLP(3, 1)

    def forward(self, x):
        """Encoder

        Args:
            x (list): [xyz_det, xyz_sec, xyz_pri]

        Returns:
            tensor: latent vector (B, feature_vector)
        """
        features = []
        for i in range(3):
            feature = self.CMLP[i](x[i])
            features.append(feature)

        x = torch.concat(features, dim=2)
        x = x.transpose(1, 2).contiguous()
        encode_result = self.MLP(x)
        encode_result = encode_result.view(-1, self.latent_dim)

        return encode_result


class CombinedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CombinedMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.MLP1 = nn.Sequential(
            SharedMLP(self.in_channels, 64),
            SharedMLP(64, 64),
            SharedMLP(64, 128)
        )
        self.MLP2 = SharedMLP(128, 256)
        self.MLP3 = SharedMLP(256, 512)
        self.MLP4 = SharedMLP(512, self.out_channels - (128+256+512))

    def forward(self, x):
        x_128 = self.MLP1(x)
        x_256 = self.MLP2(x_128)
        x_512 = self.MLP3(x_256)
        x_1024 = self.MLP4(x_512)

        feature_128 = torch.max(x_128, dim=2, keepdim=True)[0]
        feature_256 = torch.max(x_256, dim=2, keepdim=True)[0]
        feature_512 = torch.max(x_512, dim=2, keepdim=True)[0]
        feature_1024 = torch.max(x_1024, dim=2, keepdim=True)[0]

        feature = torch.concat([feature_128,
                                feature_256,
                                feature_512,
                                feature_1024], dim=1)
        
        return feature

if __name__=="__main__":
    device = "cpu"
    x_det = torch.randn(10, 3, 100, device=device)
    x_sec = torch.randn(10, 3, 50, device=device)
    x_pri = torch.randn(10, 3, 10, device=device)
    x = [x_det, x_pri, x_sec]
    encoder = MultiResolutionEncoder(latent_dim=1920)
    out = encoder(x)

    print(out.shape)

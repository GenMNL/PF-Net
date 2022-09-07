import torch
import torch.nn as nn

# ----------------------------------------------------------------------------------------
# modules for network
class SharedMLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SharedMLP, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.main = nn.Sequential(
            nn.Conv1d(self.in_channels, self.out_channels, 1),
            nn.BatchNorm1d(self.out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.main(x)
        return out
# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
# modules for sampling
def farthest_point_sampling(xyz, num_smpling):
    """function to get indices of farthest point sampling

    Args:
        xyz (torch.tensor): (B, C, N)
        num_smpling (int): number of sampling

    Returns:
        torch.tensor(dtype=torch.long): (B, num_sumpling) This is indices of FPS sampling
    """
    device = xyz.device
    B, C, N = xyz.shape

    centroids = torch.zeros((B, num_smpling), dtype=torch.long, device=device) # initialization of list for centroids
    farthest = torch.randint(0, N, (B,), device=device) # making initial centroids
    distance = torch.ones((B, N), dtype=torch.long, device=device)*1e10 # initialozation of the nearest points list

    batch_indices = torch.arange(B, dtype=torch.long, device=device) # This is used to specify batch index.

    for i in range(num_smpling):
        centroids[:, i] = farthest # updating list for centroids
        centroid = xyz[batch_indices, :, farthest] # centroid has points cordinate of farthest
        centroid = centroid.view(B, C, 1) # reshape for compute distance between centroid and points in xyz
        dist = torch.sum((centroid - xyz)**2, dim=1) # computing distance
        mask = dist < distance # make boolean list
        distance[mask] = dist[mask] # update nearest list
        farthest = torch.max(distance, dim=1)[1] # update farthest ([0] means indices)

    return centroids

def index2point_converter(xyz, indices):
    device = xyz.device
    B, C, N = xyz.shape
    num_new_points = indices.shape[1]

    batch_indices = torch.arange(B, device=device)
    batch_indices = batch_indices.view([B, 1])
    batch_indices = batch_indices.repeat([1, num_new_points])

    new_xyz = xyz[batch_indices, :, indices]
    return new_xyz.permute(0, 2, 1)

# ----------------------------------------------------------------------------------------
if __name__=="__main__":
    device = 'cpu'
    x = torch.randn(2, 3, 5, device=device)
    # mlp = SharedMLP(3, 5)
    # out = mlp(x)
    # print(out.shape)
    print(x)
    out = farthest_point_sampling(x, 2)
    print(out)
    out = index2point_converter(x, out)
    print(out)

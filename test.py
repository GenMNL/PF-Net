import torch

def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device) # the tensor which contains indicies of centroid
    distance = torch.ones(B, N).to(device) * 1e10 # the list of the nearest point distances
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device) # initial chosed indicies of each batch
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    # initial chosed point cordinate is written as xyz[batch_indices, farthest, :] which contain each batch info.
    for i in range(npoint):
        centroids[:, i] = farthest # the farthest points is next centroid
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1) # calculate the distance from now centroid
        mask = dist < distance # mask is boolean list
        distance[mask] = dist[mask] # updating the distance list. This way of writting is original of numpy array.
        farthest = torch.max(distance, -1)[1] # update the farthest point indicies
    return centroids

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    print(batch_indices.shape)
    print(idx.shape)
    new_points = points[batch_indices, idx, :]
    return new_points

if __name__=="__main__":
    x = torch.randn(2, 10, 3)
    # print(x)
    npoint = 3
    out = farthest_point_sample(x, npoint)
    # print(out)

    out = index_points(x, out)
    # print(out)

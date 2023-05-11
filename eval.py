
import torch
from torch.utils.data import DataLoader
import numpy as np
from pytorch3d.loss import chamfer_distance
import os
import open3d as o3d
from utils.data import MakeDataset, OriginalCollate
from models.module import *
from models.model import PFNet
from utils.options import make_parser
# ----------------------------------------------------------------------------------------
# make function
# ----------------------------------------------------------------------------------------
# for export ply
def export_ply(dir_path, file_name, type, point_cloud):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(point_cloud)
    path = os.path.join(dir_path, type, str(file_name)+".ply")
    o3d.io.write_point_cloud(path, pc)

def resize(ary, max, min):
    max = max[0,:]
    min = min[0,:]
    ary[:,0] *= (max[0] - min[0])
    ary[:,1] *= (max[1] - min[1])
    ary[:,2] *= (max[2] - min[2])
    ary[:,0] += min[0]
    ary[:,1] += min[1]
    ary[:,2] += min[2]

    return ary

# ----------------------------------------------------------------------------------------
def test(model_G, dataloader, save_dir):
    model_G.eval()

    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, points in enumerate(dataloader):
            diff = points[0]
            comp = points[1]
            partial = points[2].permute(0, 2, 1)
            diff_max, diff_min, comp_max, comp_min, partial_max, partial_min = points[3], points[4], points[5], points[6], points[7], points[8]

            # get 3 resolution partial point cloud list
            input_pri_idx = farthest_point_sampling(partial, 400)
            input_pri = index2point_converter(partial, input_pri_idx)
            input_sec_idx = farthest_point_sampling(partial, 800)
            input_sec = index2point_converter(partial, input_sec_idx)
            input_det = partial.clone().detach()
            input_list = [input_pri, input_sec, input_det]

            # get prediction
            # _, _, pre_det, trans_3d = model_G(input_list)
            _, _, pre_det = model_G(input_list)

            # get chanmfer distance loss
            # CD_loss = chamfer_distance(pre_comp, comp)
            # test_loss += CD_loss[0]

            comp = comp.detach().cpu().numpy()
            comp = comp.reshape(-1, 3)
            comp = resize(comp, comp_max, comp_min)
            partial = partial.permute(0, 2, 1).detach().cpu().numpy()
            partial = partial.reshape(-1, 3)
            partial = resize(partial, partial_max, partial_min)
            diff = diff.detach().cpu().numpy()
            diff = diff.reshape(-1, 3)
            diff = resize(diff, diff_max, diff_min)
            pre_det = pre_det.detach().cpu().numpy()
            pre_det = pre_det.reshape(-1, 3)
            pre_det = resize(pre_det, comp_max, comp_min)
            pre_comp = np.concatenate([pre_det, partial], axis=0)
            export_ply(save_dir, i+1, "comp", comp) # save point cloud of comp
            export_ply(save_dir, i+1, "partial", partial) # save point cloud of partial
            export_ply(save_dir, i+1, "diff", diff) # save point cloud of fine
            export_ply(save_dir, i+1, "pre_det", pre_det) # save point cloud of coarse
            export_ply(save_dir, i+1, "pre_comp", pre_comp) # save point cloud of coarse

            count += 1

    # test_loss /= count
    return test_loss

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()
    # make result dirctory

    result_dir = os.path.join(args.result_dir, args.subset)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make dataloader
    # training data
    test_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                                eval="test", num_partial_pattern=1, device=args.device)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1,
                                  collate_fn=OriginalCollate(args.device)) # DataLoader is iterable object.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # prepare model and optimaizer
    model_G = PFNet(latent_dim=args.latent_dim, final_num_points=args.final_num_points).to(args.device)
    tar_path = os.path.join(args.save_dir, "all", args.year, args.date, args.select_result + "_weight.tar")

    checkpoint = torch.load(tar_path)
    model_G.load_state_dict(checkpoint["model_G_state_dict"])

    result_dir = 'result'
    result_txt = os.path.join(result_dir, 'result.txt')
    with open(result_txt, 'w') as f:
        f.write('train_data: {}\n'.format(args.date))
        f.write('epoch: {}\n'.format(checkpoint['epoch']))
        f.write('loss : {}\n'.format(checkpoint['loss']))

    result_dir = os.path.join(args.result_dir, args.result_subset)
    test_loss = test(model_G, test_dataloader, result_dir)

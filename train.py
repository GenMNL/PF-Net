import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
from module import *
from model import *
from decoder import Discriminator
from options import *

# ----------------------------------------------------------------------------------------
def train_one_epoch(model_G, model_D, dataloader, alpha1, alpha2, optim_D, optim_G, criterion):
    device = model_G.device

    model_G.train()
    model_D.train()

    real_label = torch.ones((args.batch_size, 1), dtype=torch.float, device=device) # make labels used for discriminator
    fake_label = torch.zeros((args.batch_size, 1), dtype=float, device=device)

    sum_loss_D = 0.0
    sum_loss_G = 0.0
    sum_loss = 0.0
    count = 0

    for i, points in enumerate(tqdm(dataloader, desc="train")):
        comp = points[0]
        partial = points[1]

        # get 3 resolution partial point cloud list
        input_pri_idx = farthest_point_sampling(partial, 1000)
        input_pri = index2point_converter(partial, input_pri_idx)
        input_sec_idx = farthest_point_sampling(partial, 2000)
        input_sec = index2point_converter(partial, input_sec_idx)
        input_det = partial.clone().detach()
        input_list = [input_pri, input_sec, input_det]

        # get prediction of G
        pre_pri, pre_sec, pre_det = model_G(input_list)

        # optim D
        # Get D result of real data
        D_prediction_real = model_D(comp)
        loss_D_real = criterion(D_prediction_real, real_label)
        # Get D result of fake data
        D_prediction_fake = model_D(pre_det)
        loss_D_fake = criterion(D_prediction_fake, fake_label)
        # backward
        loss_D = loss_D_real + loss_D_fake
        model_D.zero_grad()
        loss_D.backward()
        optim_D.step()

        # optim G using chamfer distance
        CD_pri = chamfer_distance(pre_pri, comp)
        CD_sec = chamfer_distance(pre_sec, comp)
        CD_det = chamfer_distance(pre_det, comp)
        CD_loss = CD_det[0] + alpha1*CD_sec[0] + alpha2*CD_pri[0]
        # backward
        loss_G = (1 - args.weight_G_loss)*loss_D_fake + args.weight_G_loss*CD_loss
        model_G.zero_grad()
        loss_G.backward()
        optim_G.step()

        sum_loss_D += loss_D
        sum_loss_G += loss_G
        count += 1

    sum_loss_D /= count
    sum_loss_G /= count
    sum_loss = sum_loss_D + sum_loss_G

    return sum_loss_D, sum_loss_G, sum_loss

def val_one_epoch(model_G, dataloader):
    device = model_G.device

    model_D.eval()

    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, points in enumerate(dataloader):
            comp = points[0]
            partial = points[1]

            # get 3 resolution partial point cloud list
            input_pri_idx = farthest_point_sampling(partial, 1000)
            input_pri = index2point_converter(partial, input_pri_idx)
            input_sec_idx = farthest_point_sampling(partial, 2000)
            input_sec = index2point_converter(partial, input_sec_idx)
            input_det = partial.clone().detach()
            input_list = [input_pri, input_sec, input_det]

            # get prediction
            _, _, pre_det = model_G(input_list)
            # get chanmfer distance loss
            CD_loss = chamfer_distance(pre_det, comp)
            val_loss += CD_loss

    val_loss /= count
    return val_loss

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make dataloader
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # prepare model and optimaizer
    model_G = PFNet(latent_dim=args.latent_dim, final_num_points=args.final_num_points).to(args.device)
    model_D = Discriminator().to(args.device)
    criterion = nn.BCEWithLogitsLoss().to(args.device)

    if args.optimizer == "Adam":
        optim_G = torch.optim.Adam(model_G.parameters(), lr=args.lr, betas=[0.9, 0.999])
        optim_D = torch.optim.Adam(model_D.parameters(), lr=args.lr, betas=[0.9, 0.999])
    elif args.optimizer == "SGD":
        optim_G = torch.optim.SGD(model_G.parameters(), lr=args.lr, momentum=0.6)
        optim_D = torch.optim.SGD(model_D.parameters(), lr=args.lr, momentum=0.6)

    if args.lr_schdualer:
        lr_schdualer_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=int(args.epochs/4), gamma=0.2)
        lr_schdualer_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=int(args.epochs/4), gamma=0.2)

    writter = SummaryWriter()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # main loop
    best_loss = np.inf
    for epoch in tqdm(args.epochs, desc="main loop"):
        if epoch < 30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        # get loss
        loss_D, loss_G, train_loss = train_one_epoch(model_G, model_D, train_dataloader,
                                                     alpha1, alpha2, optim_D, optim_G, criterion)

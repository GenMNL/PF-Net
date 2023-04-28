import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch3d.loss import chamfer_distance
from tensorboardX import SummaryWriter
import numpy as np
from tqdm import tqdm
import os
import datetime
from utils.data import *
from models.module import *
from models.model import PFNet
from models.decoder import Discriminator
from utils.options import *

# ----------------------------------------------------------------------------------------
def train_one_epoch(model_G, model_D, dataloader, alpha1, alpha2, optim_D, optim_G, criterion):
    device = args.device

    model_G.train()
    model_D.train()

    real_label = torch.ones((args.batch_size, 1), dtype=torch.float, device=device) # make labels used for discriminator
    fake_label = torch.zeros((args.batch_size, 1), dtype=float, device=device)

    sum_loss_D = 0.0
    sum_loss_G = 0.0
    sum_loss = 0.0
    count = 0

    for _, points in enumerate(tqdm(dataloader, desc="train")):
        diff = points[0]
        partial = points[2].permute(0, 2, 1)

        # get 3 resolution partial point cloud list
        input_pri_idx = farthest_point_sampling(partial, 400)
        input_pri = index2point_converter(partial, input_pri_idx)
        input_sec_idx = farthest_point_sampling(partial, 800)
        input_sec = index2point_converter(partial, input_sec_idx)
        input_det = partial.clone().detach()
        input_list = [input_pri, input_sec, input_det]
        
        # get prediction of G
        # pre_pri, pre_sec, pre_det, trans_3d = model_G(input_list)
        pre_pri, pre_sec, pre_det = model_G(input_list)

        # diff = torch.bmm(diff, trans_3d)

        # optim D
        model_D.zero_grad()
        # Get D result of real data
        D_prediction_real = model_D(diff)
        loss_D_real = criterion(D_prediction_real, real_label)
        # Get D result of fake data
        D_prediction_fake = model_D(pre_det.detach())
        loss_D_fake = criterion(D_prediction_fake, fake_label)
        # backward
        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optim_D.step()

        # optim G using chamfer distance
        model_G.zero_grad()
        # cal chamfer distance
        CD_pri = chamfer_distance(pre_pri, diff)
        CD_sec = chamfer_distance(pre_sec, diff)
        CD_det = chamfer_distance(pre_det, diff)
        CD_loss = CD_det[0] + alpha1*CD_sec[0] + alpha2*CD_pri[0]
        # get prediction of D
        D_prediction_fake = model_D(pre_det)
        loss_D_fake = criterion(D_prediction_fake, fake_label)
        # backward
        loss_G = (1 - args.weight_G_loss)*loss_D_fake + args.weight_G_loss*CD_loss
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
    model_G.eval()

    val_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, points in enumerate(dataloader):
            diff = points[0]
            partial = points[2].permute(0, 2, 1)

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

            # diff = torch.bmm(diff, trans_3d)

            # get chanmfer distance loss
            CD_loss = chamfer_distance(pre_det, diff)
            val_loss += CD_loss[0]

            count += 1

    val_loss /= count
    return val_loss

# ----------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------
if __name__ == "__main__":
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # get options
    parser = make_parser()
    args = parser.parse_args()

    # make path of save params
    dt_now = datetime.datetime.now()
    save_date = str(dt_now.month) + str(dt_now.day) + "-" + str(dt_now.hour) + "-" + str(dt_now.minute)
    save_dir = os.path.join(args.save_dir, args.subset, str(dt_now.year), save_date)
    save_normal_path = os.path.join(save_dir, "normal_weight.tar")
    save_best_path = os.path.join(save_dir, "best_weight.tar")
    os.mkdir(save_dir)
    # make condition file
    with open(os.path.join(save_dir, "conditions.txt"), 'w') as f:
        f.write('')

    writter = SummaryWriter()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # make dataloader
    # training data
    train_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                                eval="train", num_partial_pattern=4, device=args.device)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=True, drop_last=True,
                                  collate_fn=OriginalCollate(args.device)) # DataLoader is iterable object.
    # validation data
    val_dataset = MakeDataset(dataset_path=args.dataset_dir, subset=args.subset,
                              eval="val", num_partial_pattern=4, device=args.device)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=2, shuffle=True,
                                drop_last=True, collate_fn=OriginalCollate(args.device))
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

    if args.lr_schduler:
        lr_schdualer_G = torch.optim.lr_scheduler.StepLR(optim_G, step_size=int(args.epochs/4), gamma=0.2)
        lr_schdualer_D = torch.optim.lr_scheduler.StepLR(optim_D, step_size=int(args.epochs/4), gamma=0.2)

    torch.autograd.set_detect_anomaly(True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # main loop
    best_loss = np.inf
    for epoch in tqdm(range(1, args.epochs+1), desc="main loop"):
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
        val_loss = val_one_epoch(model_G, val_dataloader)

        writter.add_scalar("D_loss", loss_D, epoch)
        writter.add_scalar("G_loss", loss_G, epoch)
        writter.add_scalar("train_loss", train_loss, epoch)
        writter.add_scalar("validation_loss", val_loss, epoch)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                        'epoch':epoch,
                        'model_G_state_dict':model_G.state_dict(), 
                        'model_D_state_dict':model_D.state_dict(), 
                        'optimizer_G_state_dict':optim_G.state_dict(),
                        'optimizer_D_state_dict':optim_D.state_dict(),
                        'loss':best_loss
                        }, save_best_path)

        # save normal weight 
        torch.save({
                    'epoch':epoch,
                    'model_G_state_dict':model_G.state_dict(), 
                    'model_D_state_dict':model_D.state_dict(), 
                    'optimizer_G_state_dict':optim_G.state_dict(),
                    'optimizer_D_state_dict':optim_D.state_dict(),
                    'loss':val_loss
                    }, save_normal_path)
        # lr_schdual.step()

    # close writter
    writter.close()


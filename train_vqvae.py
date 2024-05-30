import argparse
import sys
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from torchvision import datasets, transforms, utils

from tqdm import tqdm

from vqvae import VQVAE
from scheduler import CycleScheduler
import distributed as dist

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

from torchinfo import summary
import wandb
import matplotlib.pyplot as plt
from dataset import Dataset_EKG, NoOverlap_Dataset_EKG


def train(epoch, loader, model, optimizer, scheduler, device):

    if args.use_wandb:
        wandb.init(
                # set the wandb project where this run will be logged
                project="EKG-VQVAE",
                name=args.run_name,

                # track hyperparameters and run metadata
                config={
                "dataset": "EKG",
                }
            )

    if dist.is_primary():
        loader = tqdm(loader)

    criterion = nn.MSELoss()

    latent_loss_weight = 0.25
    sample_size = 25

    mse_sum = 0
    mse_n = 0

    for i, (batch_x, batch_y) in enumerate(loader):
        ekg = batch_x
        #print('ekg.shape: ', ekg.shape)
        #print('ekg type: ', type(ekg))

        model.zero_grad()

        ekg = ekg.to(device)

        out, latent_loss = model(ekg)
        recon_loss = criterion(out, ekg)
        latent_loss = latent_loss.mean()
        loss = recon_loss + latent_loss_weight * latent_loss
        loss.backward()

        if scheduler is not None:
            scheduler.step()
        optimizer.step()

        part_mse_sum = recon_loss.item() * ekg.shape[0]
        part_mse_n = ekg.shape[0]
        comm = {"mse_sum": part_mse_sum, "mse_n": part_mse_n}
        comm = dist.all_gather(comm)

        for part in comm:
            mse_sum += part["mse_sum"]
            mse_n += part["mse_n"]

        if i % 1000 == 0:
            if args.use_wandb:
                wandb.log({'recon_loss': recon_loss, 'latent_loss': latent_loss, 'loss': loss})

        if dist.is_primary():
            lr = optimizer.param_groups[0]["lr"]

            loader.set_description(
                (
                    f"epoch: {epoch + 1}; mse: {recon_loss.item():.5f}; "
                    f"latent: {latent_loss.item():.3f}; avg mse: {mse_sum / mse_n:.5f}; "
                    f"lr: {lr:.5f}"
                )
            )

            if i % 10000 == 0:
                model.eval()

                sample = ekg[:1]
                #print(sample.shape) # 25, 96, 1

                with torch.no_grad():
                    out, _ = model(sample)

                #print(out.shape) # 25, 96, 1

                save_img = torch.cat([sample, out], 1)
                save_img = save_img.detach().cpu().numpy()      # (1, 192, 1)
                save_img = np.squeeze(save_img)                 # 192
                ar = np.arange(len(save_img))
                plt.plot(ar, save_img, color = 'k', linestyle = 'solid',label = "EKG Data")
                plt.axvline(x = args.seq_len, color = 'b')
                os.makedirs(f'tests/{args.model_name}/', exist_ok=True)
                plt.savefig(f'tests/{args.model_name}/epoch{epoch+1}_step{i}_visualization.png')
                plt.close()

                model.train()


#######################################################################################
## Main Code
#######################################################################################

def main(args):
    device = "cuda"

    args.distributed = dist.get_world_size() > 1

    '''
    dataset = NoOverlap_Dataset_EKG(
        root_path='/home/bryanswkim/mamba/ekg-vqvae/',
        data_path='./data/example_ekg.csv',
        size=[args.seq_len, args.seq_len],
        dataset_len = 17220000
    )
    '''

    dataset = Dataset_EKG(
        root_path='/home/bryanswkim/mamba/ekg-vqvae/data/',
        size=[args.seq_len, args.seq_len],
    )

    print('dataset length: ', len(dataset))     # 

    sampler = dist.data_sampler(dataset, shuffle=True, distributed=args.distributed)

    loader = DataLoader(
        dataset,
        sampler=sampler,
        batch_size=16 // args.n_gpu,
        num_workers=2
    )

    model = VQVAE().to(device)

    if args.distributed:
        model = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[dist.get_local_rank()],
            output_device=dist.get_local_rank(),
        )

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = None
    if args.sched == "cycle":
        scheduler = CycleScheduler(
            optimizer,
            args.lr,
            n_iter=len(loader) * args.epoch,
            momentum=None,
            warmup_proportion=0.05,
        )

    summary(model, (16, 1024, 1))

    for i in range(args.epoch):
        train(i, loader, model, optimizer, scheduler, device)

        if dist.is_primary(): #and (i+1)%50 == 0:
            torch.save(model.state_dict(), f"checkpoint/vqvae_{args.model_name}_{str(i + 1).zfill(3)}.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_gpu", type=int, default=1)

    port = (
        2 ** 15
        + 2 ** 14
        + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    )
    parser.add_argument("--dist_url", default=f"tcp://127.0.0.1:{port}")

    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--epoch", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--sched", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--use_wandb", type=int, default=0)
    parser.add_argument("--run_name", type=str, default='NoRunName')
    #parser.add_argument("path", type=str)

    args = parser.parse_args()

    print(args)

    dist.launch(main, args.n_gpu, 1, 0, args.dist_url, args=(args,))

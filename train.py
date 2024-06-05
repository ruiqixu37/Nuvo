import torch
import trimesh
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

from network import Nuvo
from utils import sample_points_on_mesh, sample_uv_points
from loss import compute_loss


def main(config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = OmegaConf.load(config_path)
    model = Nuvo(**conf.model).to(device)
    sigma = nn.Parameter(torch.tensor(1.0))
    texture_map_res = int(256 * ((2 / conf.model.num_charts) ** 0.5))
    texture_maps = nn.Parameter(
        torch.randn(conf.model.num_charts, texture_map_res, texture_map_res, 6)
    )

    # optimizers and schedulers
    T_max = conf.train.epochs * conf.train.iters
    optimizer_nuvo = torch.optim.Adam(model.parameters(), lr=conf.optimizer.nuvo_lr)
    scheduler_nuvo = CosineAnnealingLR(optimizer_nuvo, T_max=T_max)
    optimizer_sigma = torch.optim.Adam([sigma], lr=conf.optimizer.sigma_lr)
    scheduler_sigma = CosineAnnealingLR(optimizer_sigma, T_max=T_max)
    optimizer_texture_maps = torch.optim.Adam(
        [texture_maps], lr=conf.optimizer.normal_grids_lr
    )
    scheduler_texture_maps = CosineAnnealingLR(optimizer_texture_maps, T_max=T_max)

    # load mesh
    mesh = trimesh.load_mesh(conf.train.mesh_path)

    # train loop
    for epoch in range(conf.train.epochs):
        for i in tqdm(range(conf.train.iters)):
            optimizer_nuvo.zero_grad()
            optimizer_sigma.zero_grad()
            optimizer_texture_maps.zero_grad()
            
            # sample points on mesh and UV points
            points, normals = sample_points_on_mesh(mesh, conf.train.G_num)
            uvs = sample_uv_points(conf.train.G_num)
            points = torch.tensor(points, dtype=torch.float32, device=device)
            normals = torch.tensor(normals, dtype=torch.float32, device=device)
            uvs = torch.tensor(uvs, dtype=torch.float32, device=device)

            # compute losses
            loss = compute_loss(
                conf,
                points,
                normals,
                uvs,
                model,
                sigma,
                texture_maps,
            )
            
            loss.backward()
            optimizer_nuvo.step()
            optimizer_sigma.step()
            optimizer_texture_maps.step()
            scheduler_nuvo.step()
            scheduler_sigma.step()
            scheduler_texture_maps.step()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="configs/nefertiti.yaml")
    args = args.parse_args()

    main(args.config)

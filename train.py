import torch
import trimesh
import wandb
import os
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.utils import make_grid
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

from network import Nuvo
from utils import (
    sample_points_on_mesh,
    sample_uv_points,
    create_rgb_maps,
    set_all_seeds,
    normalize_mesh,
    create_wandb_object,
)
from loss import compute_loss


def main(config_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conf = OmegaConf.load(config_path)
    set_all_seeds(conf.train.seed)
    model = Nuvo(**conf.model).to(device)
    sigma = nn.Parameter(torch.tensor(1.0, device=device))
    texture_map_res = int(256 * ((2 / conf.model.num_charts) ** 0.5))
    normal_maps = nn.Parameter(
        torch.zeros(
            conf.model.num_charts, texture_map_res, texture_map_res, 3, device=device
        )
    )

    # optimizers and schedulers
    T_max = conf.train.epochs * conf.train.iters
    optimizer_nuvo = torch.optim.Adam(model.parameters(), lr=conf.optimizer.nuvo_lr)
    scheduler_nuvo = CosineAnnealingLR(optimizer_nuvo, T_max=T_max)
    optimizer_sigma = torch.optim.Adam([sigma], lr=conf.optimizer.sigma_lr)
    scheduler_sigma = CosineAnnealingLR(optimizer_sigma, T_max=T_max)
    optimizer_normal_maps = torch.optim.Adam(
        [normal_maps], lr=conf.optimizer.normal_grids_lr
    )
    scheduler_normal_maps = CosineAnnealingLR(optimizer_normal_maps, T_max=T_max)

    # optional wandb logging
    if conf.train.use_wandb:
        wandb.config = OmegaConf.to_container(conf, resolve=True)
        wandb.init(project="Nuvo", name=conf.train.name)

    # load mesh
    mesh = trimesh.load_mesh(conf.train.mesh_path)
    mesh = normalize_mesh(mesh)

    # train loop
    for epoch in range(conf.train.epochs):
        for i in tqdm(range(conf.train.iters)):
            optimizer_nuvo.zero_grad()
            optimizer_sigma.zero_grad()
            optimizer_normal_maps.zero_grad()

            # sample points on mesh and UV points
            points, normals = sample_points_on_mesh(mesh, conf.train.G_num)
            uvs = sample_uv_points(conf.train.G_num)
            points = torch.tensor(points, dtype=torch.float32, device=device)
            normals = torch.tensor(normals, dtype=torch.float32, device=device)
            uvs = torch.tensor(uvs, dtype=torch.float32, device=device)

            # compute losses
            loss_dict = compute_loss(
                conf,
                points,
                normals,
                uvs,
                model,
                sigma,
                normal_maps,
            )

            loss_dict["loss_combined"].backward()
            optimizer_nuvo.step()
            optimizer_sigma.step()
            optimizer_normal_maps.step()
            scheduler_nuvo.step()
            scheduler_sigma.step()
            scheduler_normal_maps.step()

            # logging
            if conf.train.use_wandb:
                wandb.log({k: v.item() for k, v in loss_dict.items()})
                wandb.log({"sigma": sigma.item()})

                if (i + 1) % conf.train.texture_map_save_interval == 0:
                    val_normal_maps = normal_maps.permute(0, 3, 1, 2)
                    val_normal_maps = make_grid(val_normal_maps, nrow=2)
                    wandb.log({"normal_maps": [wandb.Image(val_normal_maps)]})
                    val_wandb_object, new_mesh = create_wandb_object(
                        mesh, device, model
                    )
                    wandb.log(
                        {
                            "segmented mesh w.r.t chart distribution": val_wandb_object,
                        }
                    )

                    # use wandb to save the mesh
                    mesh_path = os.path.join(wandb.run.dir, f"mesh_{epoch}_{i}.obj")
                    new_mesh.export(mesh_path)
            else:
                print(
                    f"Epoch: {epoch}, Iter: {i}, Total Loss: {loss_dict['loss_combined'].item()}"
                )

    # save model 
    if conf.train.use_wandb:
        model_path = os.path.join(wandb.run.dir, "final_model.ckpt")
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="configs/nefertiti.yaml")
    args = args.parse_args()

    main(args.config)

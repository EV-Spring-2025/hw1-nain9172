import os
import random
import torch
import wandb
from omegaconf import OmegaConf
from torch_3dgs.data import read_data
from torch_3dgs.trainer import Trainer
from torch_3dgs.model import GaussianModel
from torch_3dgs.point import *   
from torch_3dgs.utils import dict_to_device

def bonus(data, num_samples):
    N = len(data["camera"])
    if num_samples >= N:
        return data  # no need to subset
    indices = random.sample(range(N), k=num_samples)
    sub_data = {}
    for key, val in data.items():
        if isinstance(val, torch.Tensor):
            sub_data[key] = val[indices]
        elif isinstance(val, list):
            # if you have any data stored as a list
            sub_data[key] = [val[i] for i in indices]
        else:
            sub_data[key] = val  # or skip if it doesn't vary by index
    return sub_data
    
if __name__ == "__main__":
    config = OmegaConf.load("config.yaml")
    os.makedirs(config.output_folder, exist_ok=True)
    device = torch.device(config.device)

    data = read_data(config.data_folder, resize_scale=config.resize_scale)
    data = dict_to_device(data, device)
    sizes = [200, 100, 10, 2]

    for size in sizes:
        sub_data = bonus(data, size)
        sub_data = dict_to_device(sub_data, device)

        points = get_point_clouds(
            sub_data["camera"],
            sub_data["depth"],
            sub_data["alpha"],
            sub_data["rgb"],
        )
        raw_points = points.random_sample(config.num_points)
        
        model = GaussianModel(sh_degree=4, debug=False)
        model.create_from_pcd(pcd=raw_points)

        out_dir = f"{config.output_folder}_s{size}"
        os.makedirs(out_dir, exist_ok=True)

        wandb.init(
            project="EV-HW1",
            name=f"subset_{size}",
            config=OmegaConf.to_container(config, resolve=True),
        )

        trainer = Trainer(
            data=sub_data,
            model=model,
            device=device,
            num_steps=config.num_steps,
            eval_interval=config.eval_interval,
            l1_weight=config.l1_weight,
            dssim_weight=config.dssim_weight,
            depth_weight=config.depth_weight,
            lr=config.lr,
            results_folder=out_dir,
            render_kwargs={"tile_size": config.render.tile_size},
            logger=wandb,
        )

        print(f"\n===== TRAINING with {size} samples =====")
        trainer.train()

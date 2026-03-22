import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


DATASET_ROOT = Path(r"E:\Python Projects\image_disease_classification\nerf_synthetic")


def load_blender_data(scene: str, split: str = "train", white_bg: bool = True):
    scene_path = DATASET_ROOT / scene
    json_path = scene_path / f"transforms_{split}.json"

    with open(json_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    images = []
    poses = []

    for frame in meta["frames"]:
        image_path = scene_path / f"{frame['file_path']}.png"

        image = Image.open(image_path).convert("RGBA")
        image = np.array(image).astype(np.float32) / 255.0

        if white_bg:
            rgb = image[..., :3] * image[..., 3:4] + (1.0 - image[..., 3:4])
        else:
            rgb = image[..., :3]

        pose = np.array(frame["transform_matrix"], dtype=np.float32)

        images.append(rgb)
        poses.append(pose)

    images = torch.from_numpy(np.stack(images, axis=0)).float()
    poses = torch.from_numpy(np.stack(poses, axis=0)).float()

    H, W = images.shape[1], images.shape[2]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * W / math.tan(0.5 * camera_angle_x)

    return images, poses, focal, H, W

def get_rays(H, W, focal, transform_matrix):
    i, j = torch.meshgrid(
        torch.arange(W), torch.arange(H), indexing='xy'
    )
    i = i.float()
    j = j.float()

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)

    rays_d = torch.sum(dirs[..., None, :] * transform_matrix[:3, :3], dim=-1)
    rays_o = transform_matrix[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


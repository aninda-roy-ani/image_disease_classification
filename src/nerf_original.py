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

def sample_points(rays_o, rays_d, near, far, num_samples):
    t_vals = torch.linspace(near, far, steps=num_samples)
    t_vals = t_vals.view(1, 1, num_samples, 1)
    points = rays_o[..., None, :] + t_vals * rays_d[..., None, :]
    return points, t_vals

def positional_encoding(x, num_freqs):
    out = [x]
    for i in range(num_freqs):
        out.append(torch.sin((2.0 ** i) * x))
        out.append(torch.cos((2.0 ** i) * x))
    return torch.cat(out, dim=-1)


import torch.nn as nn


class NerfModel(nn.Module):
    def __init__(self, pos_dim, dir_dim, hidden_dim=128):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU()

    def forward(self, x, d):
        h = self.block1(x)
        h = self.block2(torch.cat([h, x], dim=-1))

        sigma = self.relu(h[..., 0])
        features = h[..., 1:]

        h = self.block3(torch.cat([features, d], dim=-1))
        rgb = self.block4(h)

        return rgb, sigma
    

def volume_render(rgb, sigma, t_vals):
    deltas = t_vals[..., 1:, 0] - t_vals[..., :-1, 0]
    delta_inf = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, delta_inf], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * deltas)
    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[..., :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[..., :-1]
    weights = alpha * trans

    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=-2)
    depth_map = torch.sum(weights * t_vals[..., 0], dim=-1)
    acc_map = torch.sum(weights, dim=-1)

    return rendered_rgb, depth_map, acc_map, weights


def train_nerf(model, images, poses, focal, H, W, num_epochs=100, lr=5e-4,
               near=2.0, far=6.0, num_samples=64, pos_freqs=10, dir_freqs=4,
               device="cpu"):
    model = model.to(device)
    images = images.to(device)
    poses = poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        total_loss = 0.0

        for idx in range(images.shape[0]):
            image = images[idx]
            transform_matrix = poses[idx]

            rays_o, rays_d = get_rays(H, W, focal, transform_matrix)
            rays_o = rays_o.to(device)
            rays_d = rays_d.to(device)

            points, t_vals = sample_points(rays_o, rays_d, near=near, far=far, num_samples=num_samples)
            points = points.to(device)
            t_vals = t_vals.to(device)

            encoded_points = positional_encoding(points, num_freqs=pos_freqs)

            view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            view_dirs = view_dirs[..., None, :].expand_as(points)
            encoded_dirs = positional_encoding(view_dirs, num_freqs=dir_freqs)

            rgb, sigma = model(encoded_points, encoded_dirs)
            rendered_rgb, depth_map, acc_map, weights = volume_render(rgb, sigma, t_vals)

            loss = torch.mean((rendered_rgb - image) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / images.shape[0]
        print(f"Epoch {epoch + 1}/{num_epochs} | Loss: {avg_loss:.6f}")

    return model

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scene = "chair"
    split = "train"
    pos_freqs = 10
    dir_freqs = 4
    hidden_dim = 128
    num_samples = 64
    near = 2.0
    far = 6.0
    num_epochs = 10
    lr = 5e-4

    images, poses, focal, H, W = load_blender_data(scene, split=split)

    sample_points_for_shape, _ = sample_points(
        *get_rays(H, W, focal, poses[0]),
        near=near,
        far=far,
        num_samples=num_samples
    )

    sample_rays_o, sample_rays_d = get_rays(H, W, focal, poses[0])
    sample_view_dirs = sample_rays_d / torch.norm(sample_rays_d, dim=-1, keepdim=True)
    sample_view_dirs = sample_view_dirs[..., None, :].expand_as(sample_points_for_shape)

    encoded_points = positional_encoding(sample_points_for_shape, num_freqs=pos_freqs)
    encoded_dirs = positional_encoding(sample_view_dirs, num_freqs=dir_freqs)

    pos_dim = encoded_points.shape[-1]
    dir_dim = encoded_dirs.shape[-1]

    model = NerfModel(pos_dim=pos_dim, dir_dim=dir_dim, hidden_dim=hidden_dim)

    model = train_nerf(
        model=model,
        images=images,
        poses=poses,
        focal=focal,
        H=H,
        W=W,
        num_epochs=num_epochs,
        lr=lr,
        near=near,
        far=far,
        num_samples=num_samples,
        pos_freqs=pos_freqs,
        dir_freqs=dir_freqs,
        device=device
    )

    image = images[0].to(device)
    transform_matrix = poses[0].to(device)

    rays_o, rays_d = get_rays(H, W, focal, transform_matrix)
    rays_o = rays_o.to(device)
    rays_d = rays_d.to(device)

    points, t_vals = sample_points(rays_o, rays_d, near=near, far=far, num_samples=num_samples)
    points = points.to(device)
    t_vals = t_vals.to(device)

    encoded_points = positional_encoding(points, num_freqs=pos_freqs)

    view_dirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    view_dirs = view_dirs[..., None, :].expand_as(points)
    encoded_dirs = positional_encoding(view_dirs, num_freqs=dir_freqs)

    with torch.no_grad():
        rgb, sigma = model(encoded_points, encoded_dirs)
        rendered_rgb, depth_map, acc_map, weights = volume_render(rgb, sigma, t_vals)
        loss = torch.mean((rendered_rgb - image) ** 2)

    print("rendered_rgb:", rendered_rgb.shape)
    print("depth_map:", depth_map.shape)
    print("acc_map:", acc_map.shape)
    print("weights:", weights.shape)
    print("final loss:", loss.item())

    torch.save(model.state_dict(), "nerf_model.pth")
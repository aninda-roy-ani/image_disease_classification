import json
import math
import time
from pathlib import Path
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from einops import repeat

from src.mip import sample_along_rays, integrated_pos_enc, pos_enc, volumetric_rendering, resample_along_rays


DATASET_ROOT = Path(r"E:\Python Projects\image_disease_classification\nerf_synthetic")
SCENE_NAME = "chair"

EPOCHS = 100
LR = 5e-4
N_RAND = 1024
CHUNK = 1024
WHITE_BKGD = True

NUM_SAMPLES = 64
NUM_LEVELS = 2
NET_DEPTH = 8
NET_WIDTH = 256
NET_DEPTH_CONDITION = 2
NET_WIDTH_CONDITION = 128
SKIP_INDEX = 4

Rays = namedtuple(
    "Rays",
    ("origins", "directions", "viewdirs", "radii", "lossmult", "near", "far")
)


def load_blender_data(scene, split="train", white_bg=True):
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


def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )

    i = i.float()
    j = j.float()

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)

    dx = torch.sqrt(torch.sum((rays_d[:, :-1, :] - rays_d[:, 1:, :]) ** 2, dim=-1))
    dx = torch.cat([dx, dx[:, -2:-1]], dim=1)
    radii = dx[..., None] * 2.0 / math.sqrt(12.0)

    viewdirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-9)

    return rays_o, rays_d, viewdirs, radii


def _xavier_init(linear):
    torch.nn.init.xavier_uniform_(linear.weight.data)


class MLP(torch.nn.Module):
    def __init__(
        self,
        net_depth,
        net_width,
        net_depth_condition,
        net_width_condition,
        skip_index,
        num_rgb_channels,
        num_density_channels,
        activation,
        xyz_dim,
        view_dim
    ):
        super().__init__()

        self.skip_index = skip_index
        self.xyz_dim = xyz_dim

        self.layers = torch.nn.ModuleList()

        for i in range(net_depth):
            if i == 0:
                dim_in = xyz_dim
            elif i % skip_index == 0:
                dim_in = net_width + xyz_dim
            else:
                dim_in = net_width

            linear = torch.nn.Linear(dim_in, net_width)
            _xavier_init(linear)

            if activation == "relu":
                block = torch.nn.Sequential(linear, torch.nn.ReLU(inplace=False))
            else:
                raise NotImplementedError

            self.layers.append(block)

        self.density_layer = torch.nn.Linear(net_width, num_density_channels)
        _xavier_init(self.density_layer)

        self.extra_layer = torch.nn.Linear(net_width, net_width)
        _xavier_init(self.extra_layer)

        self.view_layers = torch.nn.ModuleList()

        for i in range(net_depth_condition):
            if i == 0:
                dim_in = net_width + view_dim
            else:
                dim_in = net_width_condition

            linear = torch.nn.Linear(dim_in, net_width_condition)
            _xavier_init(linear)

            if activation == "relu":
                block = torch.nn.Sequential(linear, torch.nn.ReLU(inplace=False))
            else:
                raise NotImplementedError

            self.view_layers.append(block)

        self.color_layer = torch.nn.Linear(net_width_condition, num_rgb_channels)
        _xavier_init(self.color_layer)

    def forward(self, x, view_direction=None):
        num_samples = x.shape[1]
        inputs = x

        for i, layer in enumerate(self.layers):
            if i > 0 and i % self.skip_index == 0:
                x = torch.cat([x, inputs], dim=-1)

            x = layer(x)

        raw_density = self.density_layer(x)

        if view_direction is not None:
            bottleneck = self.extra_layer(x)

            view_direction = repeat(
                view_direction,
                "batch feature -> batch sample feature",
                sample=num_samples
            )

            x = torch.cat([bottleneck, view_direction], dim=-1)

            for layer in self.view_layers:
                x = layer(x)

        raw_rgb = self.color_layer(x)

        return raw_rgb, raw_density


class MipNerf(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.num_levels = NUM_LEVELS
        self.num_samples = NUM_SAMPLES
        self.disparity = False
        self.ray_shape = "cone"
        self.disable_integration = False
        self.min_deg_point = 0
        self.max_deg_point = 16
        self.use_viewdirs = True
        self.deg_view = 4
        self.density_noise = 0.0
        self.density_bias = -1.0
        self.rgb_padding = 0.001
        self.resample_padding = 0.01
        self.stop_resample_grad = True

        xyz_dim = (self.max_deg_point - self.min_deg_point) * 3 * 2
        view_dim = self.deg_view * 3 * 2 + 3

        self.mlp = MLP(
            NET_DEPTH,
            NET_WIDTH,
            NET_DEPTH_CONDITION,
            NET_WIDTH_CONDITION,
            SKIP_INDEX,
            3,
            1,
            "relu",
            xyz_dim,
            view_dim
        )

        self.rgb_activation = torch.nn.Sigmoid()
        self.density_activation = torch.nn.Softplus()

    def forward(self, rays, randomized=True, white_bkgd=True):
        ret = []
        t_samples = None
        weights = None

        for i_level in range(self.num_levels):
            if i_level == 0:
                t_samples, means_covs = sample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    self.num_samples,
                    rays.near,
                    rays.far,
                    randomized,
                    self.disparity,
                    self.ray_shape
                )
            else:
                t_samples, means_covs = resample_along_rays(
                    rays.origins,
                    rays.directions,
                    rays.radii,
                    t_samples,
                    weights,
                    randomized,
                    self.ray_shape,
                    self.stop_resample_grad,
                    resample_padding=self.resample_padding
                )

            samples_enc = integrated_pos_enc(
                means_covs,
                self.min_deg_point,
                self.max_deg_point
            )

            viewdirs_enc = pos_enc(
                rays.viewdirs,
                min_deg=0,
                max_deg=self.deg_view,
                append_identity=True
            )

            raw_rgb, raw_density = self.mlp(samples_enc, viewdirs_enc)

            if randomized and self.density_noise > 0:
                raw_density = raw_density + self.density_noise * torch.randn_like(raw_density)

            rgb = self.rgb_activation(raw_rgb)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding
            density = self.density_activation(raw_density + self.density_bias)

            comp_rgb, distance, acc, weights = volumetric_rendering(
                rgb,
                density,
                t_samples,
                rays.directions,
                white_bkgd=white_bkgd
            )

            ret.append((comp_rgb, distance, acc))

        return ret


def create_ray_batch(rays_o, rays_d, viewdirs, radii, target, near=2.0, far=6.0):
    near = near * torch.ones_like(rays_o[..., :1])
    far = far * torch.ones_like(rays_o[..., :1])
    lossmult = torch.ones_like(rays_o[..., :1])

    rays = Rays(
        origins=rays_o,
        directions=rays_d,
        viewdirs=viewdirs,
        radii=radii,
        lossmult=lossmult,
        near=near,
        far=far
    )

    return rays, target


def psnr_from_mse(mse):
    return -10.0 * torch.log10(mse + 1e-10)


def train(model, images, poses, focal, H, W, device):
    model.to(device)
    images = images.to(device)
    poses = poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    ), dim=-1).reshape(-1, 2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        start_time = time.time()

        for i in range(images.shape[0]):
            image = images[i]
            pose = poses[i]

            rays_o_full, rays_d_full, viewdirs_full, radii_full = get_rays(H, W, focal, pose, device)

            select = coords[torch.randint(0, coords.shape[0], (N_RAND,), device=device)]

            rays_o = rays_o_full[select[:, 0], select[:, 1]]
            rays_d = rays_d_full[select[:, 0], select[:, 1]]
            viewdirs = viewdirs_full[select[:, 0], select[:, 1]]
            radii = radii_full[select[:, 0], select[:, 1]]
            target = image[select[:, 0], select[:, 1]]

            rays, target = create_ray_batch(rays_o, rays_d, viewdirs, radii, target)

            out = model(rays, randomized=True, white_bkgd=WHITE_BKGD)
            pred = out[-1][0]

            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / images.shape[0]
        train_psnr = psnr_from_mse(torch.tensor(avg_loss)).item()
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_time = time.time() - start_time

        print(
            f"Iter {epoch + 1}/{EPOCHS} | "
            f"Loss {avg_loss:.6f} | "
            f"PSNR {train_psnr:.4f} | "
            f"LR {current_lr:.8f} | "
            f"Time {epoch_time:.3f}s"
        )

    return model


def evaluate(model, images, poses, focal, H, W, device, test_index=0):
    model.eval()
    images = images.to(device)
    poses = poses.to(device)

    with torch.no_grad():
        image = images[test_index]
        pose = poses[test_index]

        rays_o_full, rays_d_full, viewdirs_full, radii_full = get_rays(H, W, focal, pose, device)

        rays_o = rays_o_full.reshape(-1, 3)
        rays_d = rays_d_full.reshape(-1, 3)
        viewdirs = viewdirs_full.reshape(-1, 3)
        radii = radii_full.reshape(-1, 1)
        target = image.reshape(-1, 3)

        preds = []

        for start in range(0, rays_o.shape[0], CHUNK):
            end = start + CHUNK

            rays, _ = create_ray_batch(
                rays_o[start:end],
                rays_d[start:end],
                viewdirs[start:end],
                radii[start:end],
                target[start:end]
            )

            out = model(rays, randomized=False, white_bkgd=WHITE_BKGD)
            preds.append(out[-1][0])

        pred = torch.cat(preds, dim=0)
        mse = torch.mean((pred - target) ** 2)
        psnr = psnr_from_mse(mse).item()

    return psnr


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_images, train_poses, focal, H, W = load_blender_data(SCENE_NAME, split="train")
    test_images, test_poses, _, _, _ = load_blender_data(SCENE_NAME, split="test")

    model = MipNerf()

    model = train(model, train_images, train_poses, focal, H, W, device)

    print("\nFinal Evaluation Started...\n")

    final_psnr = evaluate(model, test_images, test_poses, focal, H, W, device)

    print(f"Scene: {SCENE_NAME}")
    print(f"Dataset: Blender")
    print(f"Final PSNR: {final_psnr:.4f}")

    torch.save(model.state_dict(), "mipnerf_blender_relu.pth")
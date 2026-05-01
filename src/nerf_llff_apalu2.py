import os
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio

from src.activations import APALU


DATASET_ROOT = Path(r"E:\Python Projects\image_disease_classification\llff")


def load_llff_data(scene: str, factor: int = 8):
    scene_path = DATASET_ROOT / scene
    images_dir = scene_path / f"images_{factor}"
    poses_bounds_path = scene_path / "poses_bounds.npy"

    poses_bounds = np.load(poses_bounds_path)

    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]

    poses = poses[:, :, :4]

    image_files = sorted([
        images_dir / f
        for f in os.listdir(images_dir)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ])

    images = []
    for f in image_files:
        image = imageio.imread(f).astype(np.float32) / 255.0
        images.append(image[..., :3])

    images = torch.from_numpy(np.stack(images, axis=0)).float()
    poses = torch.from_numpy(poses).float()

    H, W = images.shape[1], images.shape[2]
    focal = float(poses_bounds[0, -1])

    near = float(np.min(bounds) * 0.9)
    far = float(np.max(bounds) * 1.0)

    return images, poses, focal, H, W, near, far


def get_rays(H, W, focal, transform_matrix, device):
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

    rays_d = torch.sum(dirs[..., None, :] * transform_matrix[:3, :3], dim=-1)
    rays_o = transform_matrix[:3, 3].expand(rays_d.shape)

    return rays_o, rays_d


def sample_points(rays_o, rays_d, near, far, num_samples):
    t_vals = torch.linspace(
        near,
        far,
        steps=num_samples,
        device=rays_o.device,
        dtype=rays_o.dtype
    )

    t_vals = t_vals.view(1, num_samples, 1)
    points = rays_o[:, None, :] + t_vals * rays_d[:, None, :]

    return points, t_vals


def positional_encoding(x, num_freqs):
    out = [x]

    for i in range(num_freqs):
        freq = 2.0 ** i
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))

    return torch.cat(out, dim=-1)


class NerfModel(nn.Module):
    def __init__(self, pos_dim, dir_dim, hidden_dim=128):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            APALU(),
            nn.Linear(hidden_dim, hidden_dim),
            APALU(),
            nn.Linear(hidden_dim, hidden_dim),
            APALU(),
            nn.Linear(hidden_dim, hidden_dim),
            APALU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            APALU(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            APALU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

        self.sigma_activation = nn.Softplus()

    def forward(self, x, d):
        h = self.block1(x)
        h = self.block2(torch.cat([h, x], dim=-1))

        sigma = self.sigma_activation(h[..., 0:1]).squeeze(-1)
        features = h[..., 1:]

        h = self.block3(torch.cat([features, d], dim=-1))
        rgb = self.block4(h)

        return rgb, sigma


def volume_render(rgb, sigma, t_vals):
    deltas = t_vals[:, 1:, 0] - t_vals[:, :-1, 0]
    delta_inf = torch.full_like(deltas[:, :1], 1e10)
    deltas = torch.cat([deltas, delta_inf], dim=-1)

    sigma = torch.clamp(sigma, min=0.0, max=100.0)

    alpha = 1.0 - torch.exp(-sigma * deltas)
    alpha = torch.clamp(alpha, 0.0, 1.0)

    trans = torch.cumprod(
        torch.cat(
            [torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10],
            dim=-1
        ),
        dim=-1
    )[:, :-1]

    weights = alpha * trans
    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=1)

    return rendered_rgb


def render_rays(
    model,
    rays_o,
    rays_d,
    near,
    far,
    num_samples=32,
    pos_freqs=6,
    dir_freqs=4
):
    points, t_vals = sample_points(rays_o, rays_d, near, far, num_samples)

    encoded_points = positional_encoding(points, pos_freqs)

    view_dirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-9)
    view_dirs = view_dirs[:, None, :].expand(-1, num_samples, -1)
    encoded_dirs = positional_encoding(view_dirs, dir_freqs)

    rgb, sigma = model(encoded_points, encoded_dirs)
    rendered = volume_render(rgb, sigma, t_vals)

    return rendered


def psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    return -10.0 * torch.log10(mse + 1e-10)


def train(model, images, poses, focal, H, W, near, far, epochs=100, lr=5e-4, device="cpu"):
    model.to(device)
    images = images.to(device)
    poses = poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    n_rand = 1024
    num_samples = 32
    pos_freqs = 6
    dir_freqs = 4
    grad_clip = 1.0

    coords = torch.stack(
        torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij"
        ),
        dim=-1
    ).reshape(-1, 2)

    for epoch in range(epochs):
        total_loss = 0.0

        for i in range(images.shape[0]):
            image = images[i]
            pose = poses[i]

            rays_o_full, rays_d_full = get_rays(H, W, focal, pose, device)

            select = coords[
                torch.randint(
                    0,
                    coords.shape[0],
                    (n_rand,),
                    device=device
                )
            ]

            rays_o = rays_o_full[select[:, 0], select[:, 1]]
            rays_d = rays_d_full[select[:, 0], select[:, 1]]
            target = image[select[:, 0], select[:, 1]]

            rendered = render_rays(
                model,
                rays_o,
                rays_d,
                near,
                far,
                num_samples=num_samples,
                pos_freqs=pos_freqs,
                dir_freqs=dir_freqs
            )

            loss = mse_loss(rendered, target)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at epoch {epoch + 1}, image {i + 1}")
                return model, None

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            total_loss += loss.item()

        avg_epoch_loss = total_loss / len(images)
        print(f"Epoch {epoch + 1} Loss: {avg_epoch_loss:.6f}")

    return model, avg_epoch_loss


def evaluate_one_test_image(
    model,
    images,
    poses,
    focal,
    H,
    W,
    near,
    far,
    device="cpu",
    chunk_size=1024,
    test_index=0
):
    model.eval()
    images = images.to(device)
    poses = poses.to(device)

    with torch.no_grad():
        image = images[test_index]
        pose = poses[test_index]

        rays_o_full, rays_d_full = get_rays(H, W, focal, pose, device)

        rays_o = rays_o_full.reshape(-1, 3)
        rays_d = rays_d_full.reshape(-1, 3)
        target = image.reshape(-1, 3)

        rendered_chunks = []

        for start in range(0, rays_o.shape[0], chunk_size):
            end = start + chunk_size

            rendered_chunk = render_rays(
                model,
                rays_o[start:end],
                rays_d[start:end],
                near,
                far
            )

            rendered_chunks.append(rendered_chunk)

        rendered = torch.cat(rendered_chunks, dim=0)
        scene_psnr = psnr(rendered, target).item()

    return scene_psnr


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scene_name = "fern"

    images, poses, focal, H, W, near, far = load_llff_data(
        scene=scene_name,
        factor=8
    )

    dummy_rays_o = torch.zeros((1, 3), device=device)
    dummy_rays_d = torch.tensor([[0.0, 0.0, -1.0]], device=device)

    dummy_points, _ = sample_points(dummy_rays_o, dummy_rays_d, near, far, 32)
    pos_dim = positional_encoding(dummy_points, 6).shape[-1]

    dummy_view_dirs = dummy_rays_d / (
        torch.norm(dummy_rays_d, dim=-1, keepdim=True) + 1e-9
    )
    dummy_view_dirs = dummy_view_dirs[:, None, :].expand(-1, 32, -1)
    dir_dim = positional_encoding(dummy_view_dirs, 4).shape[-1]

    model = NerfModel(
        pos_dim=pos_dim,
        dir_dim=dir_dim,
        hidden_dim=128
    )

    model, final_train_loss = train(
        model,
        images,
        poses,
        focal,
        H,
        W,
        near,
        far,
        epochs=100,
        lr=5e-4,
        device=device
    )

    if final_train_loss is not None:
        scene_psnr = evaluate_one_test_image(
            model,
            images,
            poses,
            focal,
            H,
            W,
            near,
            far,
            device=device,
            chunk_size=1024,
            test_index=0
        )

        print(f"{scene_name} | Final Train Loss: {final_train_loss:.6f} | PSNR: {scene_psnr:.4f}")
        torch.save(model.state_dict(), "nerf_llff_apalu_block123.pth")
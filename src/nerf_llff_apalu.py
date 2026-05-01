import os
import math
import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio

from src.activations import APALU


BASE_DIR = r"E:\Python Projects\image_disease_classification\llff"
SCENE_NAME = "fern"
FACTOR = 8
LLFF_HOLD = 8
EPOCHS = 100
LR = 6e-4
LR_GAMMA = 1
N_RAND = 1024
N_SAMPLES = 32
POS_FREQS = 6
DIR_FREQS = 4
HIDDEN_DIM = 128
CHUNK = 1024
GRAD_CLIP = 1.0


def normalize(x):
    return x / (np.linalg.norm(x) + 1e-9)


def viewmatrix(z, up, pos):
    z = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, z))
    vec1 = normalize(np.cross(z, vec0))
    return np.stack([vec0, vec1, z, pos], axis=1)


def poses_avg(poses):
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    return viewmatrix(vec2, up, center)


def recenter_poses(poses):
    c2w = poses_avg(poses)
    c2w_h = np.eye(4, dtype=np.float32)
    c2w_h[:3, :4] = c2w

    bottom = np.broadcast_to(np.array([0, 0, 0, 1], dtype=np.float32), (poses.shape[0], 1, 4))
    poses_h = np.concatenate([poses, bottom], axis=1)

    poses_recentered = np.linalg.inv(c2w_h) @ poses_h
    return poses_recentered[:, :3, :4]


def load_llff_scene(base_dir, scene_name, factor=8, llffhold=8):
    scene_path = os.path.join(base_dir, scene_name)
    images_dir = os.path.join(scene_path, f"images_{factor}")

    poses_bounds = np.load(os.path.join(scene_path, "poses_bounds.npy"))
    poses = poses_bounds[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
    bounds = poses_bounds[:, -2:].astype(np.float32)

    hwf = poses[0, :, 4]
    H, W, focal = int(hwf[0]), int(hwf[1]), float(hwf[2])
    poses = poses[:, :, :4]

    image_files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.lower().endswith(("jpg", "jpeg", "png"))
    ])

    images = np.stack([
        imageio.imread(f).astype(np.float32) / 255.0
        for f in image_files
    ], axis=0)

    H, W = images.shape[1], images.shape[2]
    poses = recenter_poses(poses)

    scale = 1.0 / (bounds.min() * 0.75 + 1e-9)
    poses[:, :3, 3] *= scale
    bounds *= scale

    i_test = np.arange(images.shape[0])[::llffhold]
    i_train = np.array([i for i in range(images.shape[0]) if i not in i_test])

    train_bounds = bounds[i_train]
    near = max(float(np.percentile(train_bounds[:, 0], 5) * 0.9), 0.1)
    far = float(np.percentile(train_bounds[:, 1], 95))

    return (
        torch.from_numpy(images[i_train]).float(),
        torch.from_numpy(poses[i_train]).float(),
        torch.from_numpy(images[i_test]).float(),
        torch.from_numpy(poses[i_test]).float(),
        focal, H, W, near, far
    )


def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        indexing="xy"
    )

    dirs = torch.stack([
        (i - W * 0.5) / focal,
        -(j - H * 0.5) / focal,
        -torch.ones_like(i)
    ], dim=-1)

    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], dim=-1)
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def sample_points(rays_o, rays_d, near, far, n_samples):
    t_vals = torch.linspace(near, far, steps=n_samples, device=rays_o.device)
    t_vals = t_vals.view(1, n_samples, 1)
    return rays_o[:, None, :] + t_vals * rays_d[:, None, :], t_vals


def positional_encoding(x, L):
    out = [x]
    for i in range(L):
        freq = 2.0 ** i
        out += [torch.sin(freq * x), torch.cos(freq * x)]
    return torch.cat(out, dim=-1)


class NerfModel(nn.Module):
    def __init__(self, pos_dim, dir_dim):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(pos_dim, HIDDEN_DIM), APALU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), APALU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), APALU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM), APALU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(HIDDEN_DIM + pos_dim, HIDDEN_DIM), APALU(),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(HIDDEN_DIM + dir_dim, HIDDEN_DIM // 2), APALU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(HIDDEN_DIM // 2, 3), nn.Sigmoid()
        )

        self.sigma_activation = nn.Softplus()

    def forward(self, x, d):
        h = self.block1(x)
        h = self.block2(torch.cat([h, x], dim=-1))
        sigma = self.sigma_activation(h[..., 0:1]).squeeze(-1)
        feat = h[..., 1:]
        h = self.block3(torch.cat([feat, d], dim=-1))
        return self.block4(h), sigma


def volume_render(rgb, sigma, t_vals):
    deltas = t_vals[:, 1:, 0] - t_vals[:, :-1, 0]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)

    alpha = 1.0 - torch.exp(-torch.clamp(sigma, 0.0, 100.0) * deltas)
    trans = torch.cumprod(torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    return torch.sum((alpha * trans)[..., None] * rgb, dim=1)


def render_rays(model, rays_o, rays_d, near, far):
    pts, t_vals = sample_points(rays_o, rays_d, near, far, N_SAMPLES)
    enc_pts = positional_encoding(pts, POS_FREQS)

    dirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-9)
    dirs = dirs[:, None, :].expand(-1, N_SAMPLES, -1)
    enc_dirs = positional_encoding(dirs, DIR_FREQS)

    rgb, sigma = model(enc_pts, enc_dirs)
    return volume_render(rgb, sigma, t_vals)


def train(model, images, poses, focal, H, W, near, far, device):
    model.to(device)
    images, poses = images.to(device), poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    ), -1).reshape(-1, 2)

    for epoch in range(EPOCHS):
        total_loss = 0

        for i in range(len(images)):
            rays_o, rays_d = get_rays(H, W, focal, poses[i], device)

            sel = coords[torch.randint(0, coords.shape[0], (N_RAND,), device=device)]
            pred = render_rays(model,
                               rays_o[sel[:, 0], sel[:, 1]],
                               rays_d[sel[:, 0], sel[:, 1]],
                               near, far)

            loss = torch.mean((pred - images[i][sel[:, 0], sel[:, 1]]) ** 2)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        print(f"Epoch {epoch+1} Loss: {total_loss/len(images):.6f}")

    return model


def evaluate(model, images, poses, focal, H, W, near, far, device):
    model.eval()
    images, poses = images.to(device), poses.to(device)

    with torch.no_grad():
        rays_o, rays_d = get_rays(H, W, focal, poses[0], device)

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        target = images[0].reshape(-1, 3)

        preds = []
        for i in range(0, len(rays_o), CHUNK):
            preds.append(render_rays(model, rays_o[i:i+CHUNK], rays_d[i:i+CHUNK], near, far))

        pred = torch.cat(preds, 0)
        mse = torch.mean((pred - target) ** 2)
        return (-10 * torch.log10(mse + 1e-10)).item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_imgs, train_poses, test_imgs, test_poses, focal, H, W, near, far = load_llff_scene(
        BASE_DIR, SCENE_NAME, FACTOR, LLFF_HOLD
    )

    dummy_o = torch.zeros((1, 3), device=device)
    dummy_d = torch.tensor([[0, 0, -1]], device=device)

    pos_dim = positional_encoding(sample_points(dummy_o, dummy_d, near, far, N_SAMPLES)[0], POS_FREQS).shape[-1]
    dir_dim = positional_encoding(dummy_d[:, None, :].expand(-1, N_SAMPLES, -1), DIR_FREQS).shape[-1]

    model = NerfModel(pos_dim, dir_dim)
    model = train(model, train_imgs, train_poses, focal, H, W, near, far, device)

    psnr = evaluate(model, test_imgs, test_poses, focal, H, W, near, far, device)
    print(f"PSNR: {psnr:.4f}")
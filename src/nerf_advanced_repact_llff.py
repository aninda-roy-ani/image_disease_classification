import os
import math
import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio

from src.activations import RepAct_Softmax


BASE_DIR = r"E:\Python Projects\image_disease_classification\llff"
SCENE_NAME = "fern"
ACTIVATION_MODE = "block1"
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
    m = np.stack([vec0, vec1, z, pos], axis=1)
    return m


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
    if not os.path.isdir(images_dir):
        images_dir = os.path.join(scene_path, "images")

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

    images = []
    for f in image_files:
        img = imageio.imread(f).astype(np.float32) / 255.0
        images.append(img)
    images = np.stack(images, axis=0)

    H = images.shape[1]
    W = images.shape[2]

    poses = recenter_poses(poses)

    scale = 1.0 / (bounds.min() * 0.75 + 1e-9)
    poses[:, :3, 3] *= scale
    bounds *= scale

    i_test = np.arange(images.shape[0])[::llffhold]
    i_train = np.array([i for i in range(images.shape[0]) if i not in i_test])

    train_bounds = bounds[i_train]
    near = max(float(np.percentile(train_bounds[:, 0], 5) * 0.9), 0.1)
    far = float(np.percentile(train_bounds[:, 1], 95) * 1.0)

    train_images = torch.from_numpy(images[i_train]).float()
    train_poses = torch.from_numpy(poses[i_train]).float()
    test_images = torch.from_numpy(images[i_test]).float()
    test_poses = torch.from_numpy(poses[i_test]).float()

    return train_images, train_poses, test_images, test_poses, focal, H, W, near, far


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
    t_vals = torch.linspace(near, far, steps=n_samples, device=rays_o.device, dtype=rays_o.dtype)
    t_vals = t_vals.view(1, n_samples, 1)
    points = rays_o[:, None, :] + t_vals * rays_d[:, None, :]
    return points, t_vals


def positional_encoding(x, L):
    out = [x]
    for i in range(L):
        freq = 2.0 ** i
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


def get_act(use_repact):
    return RepAct_Softmax() if use_repact else nn.ReLU()


class NerfModel(nn.Module):
    def __init__(self, pos_dim, dir_dim, activation_mode="relu", hidden_dim=128):
        super().__init__()

        use_b1 = activation_mode in {"block1", "block12", "block123"}
        use_b2 = activation_mode in {"block2", "block12", "block123"}
        use_b3 = activation_mode in {"block3", "block123"}

        self.block1 = nn.Sequential(
            nn.Linear(pos_dim, hidden_dim),
            get_act(use_b1),
            nn.Linear(hidden_dim, hidden_dim),
            get_act(use_b1),
            nn.Linear(hidden_dim, hidden_dim),
            get_act(use_b1),
            nn.Linear(hidden_dim, hidden_dim),
            get_act(use_b1)
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            get_act(use_b2),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            get_act(use_b3)
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
        feat = h[..., 1:]
        h = self.block3(torch.cat([feat, d], dim=-1))
        rgb = self.block4(h)
        return rgb, sigma


def volume_render(rgb, sigma, t_vals):
    deltas = t_vals[:, 1:, 0] - t_vals[:, :-1, 0]
    deltas = torch.cat([deltas, torch.full_like(deltas[:, :1], 1e10)], dim=-1)

    sigma = torch.clamp(sigma, 0.0, 100.0)
    alpha = 1.0 - torch.exp(-sigma * deltas)
    alpha = torch.clamp(alpha, 0.0, 1.0)

    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]

    weights = alpha * trans
    return torch.sum(weights[..., None] * rgb, dim=1)


def render_rays(model, rays_o, rays_d, near, far):
    pts, t_vals = sample_points(rays_o, rays_d, near, far, N_SAMPLES)
    enc_pts = positional_encoding(pts, POS_FREQS)

    view_dirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-9)
    view_dirs = view_dirs[:, None, :].expand(-1, N_SAMPLES, -1)
    enc_dirs = positional_encoding(view_dirs, DIR_FREQS)

    rgb, sigma = model(enc_pts, enc_dirs)
    return volume_render(rgb, sigma, t_vals)


def train(model, images, poses, focal, H, W, near, far, device):
    model.to(device)
    images = images.to(device)
    poses = poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)

    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    ), -1).reshape(-1, 2)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for i in range(len(images)):
            img = images[i]
            pose = poses[i]

            rays_o, rays_d = get_rays(H, W, focal, pose, device)

            sel = coords[torch.randint(0, coords.shape[0], (N_RAND,), device=device)]
            rays_o = rays_o[sel[:, 0], sel[:, 1]]
            rays_d = rays_d[sel[:, 0], sel[:, 1]]
            target = img[sel[:, 0], sel[:, 1]]

            pred = render_rays(model, rays_o, rays_d, near, far)
            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            total_loss += loss.item()

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1} Loss: {total_loss / len(images):.6f} LR: {current_lr:.8f}")

    return model


def psnr_from_mse(mse):
    return -10.0 * torch.log10(mse + 1e-10)


def evaluate(model, images, poses, focal, H, W, near, far, device):
    model.eval()
    images = images.to(device)
    poses = poses.to(device)

    psnrs = []

    with torch.no_grad():
        for idx in range(len(images)):
            image = images[idx]
            pose = poses[idx]

            rays_o, rays_d = get_rays(H, W, focal, pose, device)
            rays_o = rays_o.reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)
            target = image.reshape(-1, 3)

            outputs = []
            for i in range(0, rays_o.shape[0], CHUNK):
                pred = render_rays(model, rays_o[i:i + CHUNK], rays_d[i:i + CHUNK], near, far)
                outputs.append(pred)

            pred = torch.cat(outputs, dim=0)
            mse = torch.mean((pred - target) ** 2)
            psnrs.append(psnr_from_mse(mse).item())

    return float(np.mean(psnrs))


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_images, train_poses, test_images, test_poses, focal, H, W, near, far = load_llff_scene(
        BASE_DIR,
        SCENE_NAME,
        factor=FACTOR,
        llffhold=LLFF_HOLD
    )

    dummy_o = torch.zeros((1, 3), device=device)
    dummy_d = torch.tensor([[0.0, 0.0, -1.0]], device=device)

    pts, _ = sample_points(dummy_o, dummy_d, near, far, N_SAMPLES)
    pos_dim = positional_encoding(pts, POS_FREQS).shape[-1]

    dirs = dummy_d[:, None, :].expand(-1, N_SAMPLES, -1)
    dir_dim = positional_encoding(dirs, DIR_FREQS).shape[-1]

    model = NerfModel(
        pos_dim=pos_dim,
        dir_dim=dir_dim,
        activation_mode=ACTIVATION_MODE,
        hidden_dim=HIDDEN_DIM
    )

    model = train(model, train_images, train_poses, focal, H, W, near, far, device)
    test_psnr = evaluate(model, test_images, test_poses, focal, H, W, near, far, device)

    print(f"Scene: {SCENE_NAME}")
    print(f"Activation mode: {ACTIVATION_MODE}")
    print(f"Near: {near:.4f} Far: {far:.4f}")
    print(f"Test PSNR: {test_psnr:.4f}")
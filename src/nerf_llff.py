import os
import math
import numpy as np
import torch
import torch.nn as nn
import imageio.v2 as imageio


BASE_DIR = r"E:\Python Projects\image_disease_classification\llff"


def load_llff_data(scene_path, factor=8):
    images_dir = os.path.join(scene_path, f"images_{factor}")
    poses_bounds = np.load(os.path.join(scene_path, "poses_bounds.npy"))

    poses = poses_bounds[:, :-2].reshape([-1, 3, 5])
    bounds = poses_bounds[:, -2:]
    poses = poses[:, :, :4]

    image_files = sorted([
        os.path.join(images_dir, f)
        for f in os.listdir(images_dir)
        if f.endswith("jpg") or f.endswith("png")
    ])

    images = []
    for f in image_files:
        img = imageio.imread(f).astype(np.float32) / 255.0
        images.append(img)

    images = np.stack(images, axis=0)

    H = images.shape[1]
    W = images.shape[2]
    focal = poses_bounds[0, -1]

    images = torch.from_numpy(images).float()
    poses = torch.from_numpy(poses).float()

    near = np.min(bounds) * 0.9
    far = np.max(bounds) * 1.0

    return images, poses, focal, H, W, near, far


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
    points = rays_o[:, None, :] + t_vals * rays_d[:, None, :]
    return points, t_vals


def positional_encoding(x, L):
    out = [x]
    for i in range(L):
        freq = 2.0 ** i
        out.append(torch.sin(freq * x))
        out.append(torch.cos(freq * x))
    return torch.cat(out, dim=-1)


class NerfModel(nn.Module):
    def __init__(self, pos_dim, dir_dim):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Linear(pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.block2 = nn.Sequential(
            nn.Linear(128 + pos_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 129)
        )

        self.block3 = nn.Sequential(
            nn.Linear(128 + dir_dim, 64),
            nn.ReLU()
        )

        self.block4 = nn.Sequential(
            nn.Linear(64, 3),
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

    trans = torch.cumprod(
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]

    weights = alpha * trans
    return torch.sum(weights[..., None] * rgb, dim=1)


def render_rays(model, rays_o, rays_d, near, far):
    pts, t_vals = sample_points(rays_o, rays_d, near, far, 32)

    enc_pts = positional_encoding(pts, 6)

    view_dirs = rays_d / (torch.norm(rays_d, dim=-1, keepdim=True) + 1e-9)
    view_dirs = view_dirs[:, None, :].expand(-1, 32, -1)
    enc_dirs = positional_encoding(view_dirs, 4)

    rgb, sigma = model(enc_pts, enc_dirs)
    return volume_render(rgb, sigma, t_vals)


def train(model, images, poses, focal, H, W, near, far, device):
    model.to(device)
    images = images.to(device)
    poses = poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    coords = torch.stack(torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij"
    ), -1).reshape(-1, 2)

    for epoch in range(100):
        total_loss = 0.0

        for i in range(len(images)):
            img = images[i]
            pose = poses[i]

            rays_o, rays_d = get_rays(H, W, focal, pose, device)

            sel = coords[torch.randint(0, coords.shape[0], (1024,), device=device)]
            rays_o = rays_o[sel[:, 0], sel[:, 1]]
            rays_d = rays_d[sel[:, 0], sel[:, 1]]
            target = img[sel[:, 0], sel[:, 1]]

            pred = render_rays(model, rays_o, rays_d, near, far)
            loss = torch.mean((pred - target) ** 2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {total_loss / len(images):.6f}")

    return model


def evaluate(model, images, poses, focal, H, W, near, far, device):
    model.eval()

    with torch.no_grad():
        image = images[0].to(device)
        pose = poses[0].to(device)

        rays_o, rays_d = get_rays(H, W, focal, pose, device)

        rays_o = rays_o.reshape(-1, 3)
        rays_d = rays_d.reshape(-1, 3)
        target = image.reshape(-1, 3)

        outputs = []
        chunk = 1024

        for i in range(0, rays_o.shape[0], chunk):
            pred = render_rays(
                model,
                rays_o[i:i + chunk],
                rays_d[i:i + chunk],
                near,
                far
            )
            outputs.append(pred)

        pred = torch.cat(outputs, dim=0)

        mse = torch.mean((pred - target) ** 2)
        psnr = -10.0 * torch.log10(mse + 1e-10)

    return psnr.item()


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scene_path = r"E:\Python Projects\image_disease_classification\llff\fern"

    images, poses, focal, H, W, near, far = load_llff_data(scene_path)

    dummy_o = torch.zeros((1, 3), device=device)
    dummy_d = torch.tensor([[0.0, 0.0, -1.0]], device=device)

    pts, _ = sample_points(dummy_o, dummy_d, near, far, 32)
    pos_dim = positional_encoding(pts, 6).shape[-1]

    dirs = dummy_d[:, None, :].expand(-1, 32, -1)
    dir_dim = positional_encoding(dirs, 4).shape[-1]

    model = NerfModel(pos_dim, dir_dim)

    model = train(model, images, poses, focal, H, W, near, far, device)

    psnr_value = evaluate(model, images, poses, focal, H, W, near, far, device)
    print(f"PSNR: {psnr_value:.4f}")

    print("Training complete")
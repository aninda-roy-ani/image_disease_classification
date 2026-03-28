import csv
import gc
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.activations import RepAct_Softmax


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
            RepAct_Softmax(),
            nn.Linear(hidden_dim, hidden_dim),
            RepAct_Softmax(),
            nn.Linear(hidden_dim, hidden_dim),
            RepAct_Softmax(),
            nn.Linear(hidden_dim, hidden_dim),
            RepAct_Softmax()
        )

        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim + pos_dim, hidden_dim),
            RepAct_Softmax(),
            nn.Linear(hidden_dim, hidden_dim + 1)
        )

        self.block3 = nn.Sequential(
            nn.Linear(hidden_dim + dir_dim, hidden_dim // 2),
            RepAct_Softmax()
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
        torch.cat([torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1),
        dim=-1
    )[:, :-1]

    weights = alpha * trans
    rendered_rgb = torch.sum(weights[..., None] * rgb, dim=1)

    return rendered_rgb


def render_rays(model, rays_o, rays_d, num_samples=32, pos_freqs=6, dir_freqs=4, near=2.0, far=6.0):
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


def train(model, images, poses, focal, H, W, epochs=20, lr=5e-4, device="cpu"):
    model.to(device)
    images = images.to(device)
    poses = poses.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    N_rand = 1024
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

            select = coords[torch.randint(0, coords.shape[0], (N_rand,), device=device)]
            rays_o = rays_o_full[select[:, 0], select[:, 1]]
            rays_d = rays_d_full[select[:, 0], select[:, 1]]
            target = image[select[:, 0], select[:, 1]]

            rendered = render_rays(
                model,
                rays_o,
                rays_d,
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


def evaluate_one_test_image(model, images, poses, focal, H, W, device="cpu", chunk_size=1024, test_index=0):
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
            rendered_chunk = render_rays(model, rays_o[start:end], rays_d[start:end])
            rendered_chunks.append(rendered_chunk)

        rendered = torch.cat(rendered_chunks, dim=0)
        scene_psnr = psnr(rendered, target).item()

    return scene_psnr


def get_scene_names():
    scenes = []
    for p in DATASET_ROOT.iterdir():
        if p.is_dir() and (p / "transforms_train.json").exists() and (p / "transforms_test.json").exists():
            scenes.append(p.name)
    scenes.sort()
    return scenes


def build_model_dims(device):
    dummy_rays_o = torch.zeros((1, 3), device=device)
    dummy_rays_d = torch.tensor([[0.0, 0.0, -1.0]], device=device)

    dummy_points, _ = sample_points(dummy_rays_o, dummy_rays_d, 2.0, 6.0, 32)
    pos_dim = positional_encoding(dummy_points, 6).shape[-1]

    dummy_view_dirs = dummy_rays_d / (torch.norm(dummy_rays_d, dim=-1, keepdim=True) + 1e-9)
    dummy_view_dirs = dummy_view_dirs[:, None, :].expand(-1, 32, -1)
    dir_dim = positional_encoding(dummy_view_dirs, 4).shape[-1]

    return pos_dim, dir_dim


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scenes = get_scene_names()

    print("Scenes found:", scenes)

    pos_dim, dir_dim = build_model_dims(device)

    results = []

    for scene_name in scenes:
        print(f"\nRunning scene: {scene_name}")

        train_images, train_poses, focal, H, W = load_blender_data(scene_name, split="train")
        test_images, test_poses, _, _, _ = load_blender_data(scene_name, split="test")

        model = NerfModel(
            pos_dim=pos_dim,
            dir_dim=dir_dim,
            hidden_dim=128
        )

        model, final_train_loss = train(
            model,
            train_images,
            train_poses,
            focal,
            H,
            W,
            epochs=10,
            lr=5e-4,
            device=device
        )

        if final_train_loss is None:
            print(f"Training failed for {scene_name}")
            continue

        scene_psnr = evaluate_one_test_image(
            model,
            test_images,
            test_poses,
            focal,
            H,
            W,
            device=device,
            chunk_size=1024,
            test_index=0
        )

        torch.save(model.state_dict(), f"nerf_model_repact_softmax_{scene_name}.pth")

        print(f"{scene_name} | Final Train Loss: {final_train_loss:.6f} | PSNR: {scene_psnr:.4f}")

        results.append({
            "scene": scene_name,
            "final_train_loss": final_train_loss,
            "psnr": scene_psnr
        })

        del model
        del train_images
        del train_poses
        del test_images
        del test_poses
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    avg_train_loss = float(np.mean([r["final_train_loss"] for r in results])) if results else 0.0
    avg_psnr = float(np.mean([r["psnr"] for r in results])) if results else 0.0

    print("\nAll scene results:")
    for r in results:
        print(f"{r['scene']} | Final Train Loss: {r['final_train_loss']:.6f} | PSNR: {r['psnr']:.4f}")

    print(f"\nAverage Final Train Loss across all scenes: {avg_train_loss:.6f}")
    print(f"Average PSNR across all scenes: {avg_psnr:.4f}")

    with open("repact_softmax_fast_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scene", "final_train_loss", "psnr"])
        for r in results:
            writer.writerow([r["scene"], r["final_train_loss"], r["psnr"]])
        writer.writerow(["average", avg_train_loss, avg_psnr])

    with open("repact_softmax_fast_results.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "results": results,
                "average_final_train_loss": avg_train_loss,
                "average_psnr": avg_psnr
            },
            f,
            indent=2
        )
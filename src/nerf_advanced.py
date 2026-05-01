import os
import json
import math
import time
import random
from dataclasses import dataclass, asdict
from typing import Optional, Tuple, Dict, Any, List

import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm


@dataclass
class Config:
    dataset_type: str = "llff"
    data_dir: str = "./data/fern"
    exp_name: str = "nerf_run"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    llff_hold: int = 8
    llff_factor: int = 4
    llff_recenter: bool = True
    llff_bd_factor: float = 0.75
    use_ndc: bool = True

    blender_half_res: bool = True
    blender_white_bkgd: bool = True
    blender_testskip: int = 1

    H: Optional[int] = None
    W: Optional[int] = None
    focal: Optional[float] = None

    near: float = 0.0
    far: float = 1.0
    near_percentile: float = 5.0
    far_percentile: float = 95.0

    N_iters: int = 100
    rays_per_batch: int = 1024
    N_samples: int = 64
    N_importance: int = 64
    perturb: float = 1.0
    lindisp: bool = False
    raw_noise_std: float = 0.0

    multires: int = 10
    multires_views: int = 4
    use_viewdirs: bool = True

    netdepth: int = 8
    netwidth: int = 256
    netdepth_fine: int = 8
    netwidth_fine: int = 256
    skips: Tuple[int, ...] = (4,)

    lr: float = 5e-4
    lr_scheduler: str = "two_stage"
    lr_gamma: float = 0.995
    lr_gamma_stage2: float = 0.98
    lr_warmup_iters: int = 0
    lr_decay_start: int = 20
    lr_stage2_start: int = 70
    step_lr_size: int = 20
    cosine_min_lr: float = 5e-5

    grad_clip: float = 1.0
    amp: bool = True

    precrop_iters: int = 10
    precrop_frac: float = 0.5

    render_factor: int = 0
    i_print: int = 1
    i_weights: int = 10
    i_testset: int = -1

    ckpt_path: Optional[str] = None
    output_dir: str = "./outputs"

    activation_name: str = "relu"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class RePAct(nn.Module):
    def __init__(self, mode: str = "softmax", channels: int = 1):
        super().__init__()
        self.mode = mode
        self.idn = nn.Identity()
        self.relu = nn.ReLU()
        self.prelu = nn.PReLU(num_parameters=channels)
        self.hswish = nn.Hardswish()
        self.w = nn.Parameter(torch.ones(4))

    def forward(self, x):
        acts = [
            self.idn(x),
            self.relu(x),
            self.prelu(x),
            self.hswish(x),
        ]
        if self.mode == "origin":
            w = self.w
        elif self.mode == "softmax":
            w = torch.softmax(self.w, dim=0)
        elif self.mode == "bn":
            w = self.w / (self.w.norm(p=2) + 1e-8)
        else:
            w = torch.softmax(self.w, dim=0)
        y = 0.0
        for i in range(4):
            y = y + w[i] * acts[i]
        return y


def build_activation(name: str, width: int):
    n = name.lower()
    if n == "relu":
        return nn.ReLU()
    if n == "prelu":
        return nn.PReLU(num_parameters=width)
    if n == "hardswish":
        return nn.Hardswish()
    if n == "gelu":
        return nn.GELU()
    if n == "repact_origin":
        return RePAct("origin", width)
    if n == "repact_softmax":
        return RePAct("softmax", width)
    if n == "repact_bn":
        return RePAct("bn", width)
    raise ValueError(f"Unknown activation: {name}")


class Embedder:
    def __init__(self, input_dims=3, max_freq_log2=9, N_freqs=10, include_input=True, log_sampling=True):
        self.input_dims = input_dims
        self.include_input = include_input
        self.max_freq_log2 = max_freq_log2
        self.N_freqs = N_freqs
        self.log_sampling = log_sampling
        self.out_dim = 0
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.input_dims
        out_dim = 0
        if self.include_input:
            embed_fns.append(lambda x: x)
            out_dim += d
        if self.log_sampling:
            freq_bands = 2.0 ** torch.linspace(0.0, self.max_freq_log2, steps=self.N_freqs)
        else:
            freq_bands = torch.linspace(2.0 ** 0.0, 2.0 ** self.max_freq_log2, steps=self.N_freqs)
        for freq in freq_bands:
            embed_fns.append(lambda x, freq=freq: torch.sin(x * freq))
            embed_fns.append(lambda x, freq=freq: torch.cos(x * freq))
            out_dim += 2 * d
        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class NeRF(nn.Module):
    def __init__(
        self,
        D=8,
        W=256,
        input_ch=63,
        input_ch_views=27,
        skips=(4,),
        use_viewdirs=True,
        activation_name="relu",
    ):
        super().__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = set(skips)
        self.use_viewdirs = use_viewdirs

        pts_linears = []
        for i in range(D):
            if i == 0:
                in_ch = input_ch
            elif i in self.skips:
                in_ch = W + input_ch
            else:
                in_ch = W
            pts_linears.append(nn.Linear(in_ch, W))
        self.pts_linears = nn.ModuleList(pts_linears)
        self.pts_acts = nn.ModuleList([build_activation(activation_name, W) for _ in range(D)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.view_linear = nn.Linear(W + input_ch_views, W // 2)
            self.view_act = build_activation(activation_name, W // 2)
            self.rgb_linear = nn.Linear(W // 2, 3)
        else:
            self.output_linear = nn.Linear(W, 4)

    def forward(self, x):
        if self.use_viewdirs:
            input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        else:
            input_pts = x
            input_views = None

        h = input_pts
        for i, (lin, act) in enumerate(zip(self.pts_linears, self.pts_acts)):
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)
            h = act(lin(h))

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)
            h = self.view_act(self.view_linear(h))
            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log10(x + 1e-10)


def to8b(x):
    return (255 * np.clip(x, 0, 1)).astype(np.uint8)


def get_rays(H, W, focal, c2w, device):
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32, device=device),
        torch.arange(H, dtype=torch.float32, device=device),
        indexing="xy",
    )
    dirs = torch.stack([(i - W * 0.5) / focal, -(j - H * 0.5) / focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def ndc_rays(H, W, focal, near, rays_o, rays_d):
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    o0 = -1.0 / (W / (2.0 * focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1.0 / (H / (2.0 * focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1.0 + 2.0 * near / rays_o[..., 2]

    d0 = -1.0 / (W / (2.0 * focal)) * (rays_d[..., 0] / rays_d[..., 2] - rays_o[..., 0] / rays_o[..., 2])
    d1 = -1.0 / (H / (2.0 * focal)) * (rays_d[..., 1] / rays_d[..., 2] - rays_o[..., 1] / rays_o[..., 2])
    d2 = -2.0 * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)
    return rays_o, rays_d


def sample_pdf(bins, weights, N_samples, det=False):
    weights = weights + 1e-5
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)

    if det:
        u = torch.linspace(0.0, 1.0, steps=N_samples, device=bins.device)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples], device=bins.device)

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp(inds - 1, min=0)
    above = torch.clamp(inds, max=cdf.shape[-1] - 1)
    inds_g = torch.stack([below, above], -1)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
    return samples


def raw2outputs(raw, z_vals, rays_d, raw_noise_std=0.0, white_bkgd=False):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, 1e10 * torch.ones_like(dists[..., :1])], -1)
    dists = dists * torch.norm(rays_d[..., None, :], dim=-1)

    rgb = torch.sigmoid(raw[..., :3])
    noise = 0.0
    if raw_noise_std > 0.0:
        noise = torch.randn(raw[..., 3].shape, device=raw.device) * raw_noise_std

    sigma = F.relu(raw[..., 3] + noise)
    alpha = 1.0 - torch.exp(-sigma * dists)
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=alpha.device), 1.0 - alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    depth_map = torch.sum(weights * z_vals, -1)
    acc_map = torch.sum(weights, -1)
    disp_map = 1.0 / torch.max(1e-10 * torch.ones_like(depth_map), depth_map / acc_map.clamp(min=1e-10))

    if white_bkgd:
        rgb_map = rgb_map + (1.0 - acc_map[..., None])

    return rgb_map, disp_map, acc_map, weights, depth_map


def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=65536):
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn.embed(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:, None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn.embed(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = []
    for i in range(0, embedded.shape[0], netchunk):
        outputs_flat.append(fn(embedded[i:i + netchunk]))
    outputs_flat = torch.cat(outputs_flat, 0)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs


def render_rays(
    ray_batch,
    network_fn,
    network_fine,
    embed_fn,
    embeddirs_fn,
    N_samples,
    retraw=False,
    lindisp=False,
    perturb=0.0,
    N_importance=0,
    network_chunk=65536,
    raw_noise_std=0.0,
    white_bkgd=False,
):
    N_rays = ray_batch.shape[0]
    rays_o, rays_d = ray_batch[:, 0:3], ray_batch[:, 3:6]
    viewdirs = ray_batch[:, -3:] if ray_batch.shape[-1] > 8 else None
    bounds = torch.reshape(ray_batch[..., 6:8], [-1, 1, 2])
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0.0, 1.0, steps=N_samples, device=ray_batch.device)
    if not lindisp:
        z_vals = near * (1.0 - t_vals) + far * t_vals
    else:
        z_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    z_vals = z_vals.expand([N_rays, N_samples])

    if perturb > 0.0:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        t_rand = torch.rand(z_vals.shape, device=ray_batch.device)
        z_vals = lower + (upper - lower) * t_rand

    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    raw = run_network(pts, viewdirs, network_fn, embed_fn, embeddirs_fn, network_chunk)
    rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

    if N_importance > 0:
        rgb_map_0, disp_map_0, acc_map_0 = rgb_map, disp_map, acc_map
        z_vals_mid = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(z_vals_mid, weights[..., 1:-1].detach(), N_importance, det=(perturb == 0.0))
        z_samples = z_samples.detach()
        z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
        run_fn = network_fine if network_fine is not None else network_fn
        raw = run_network(pts, viewdirs, run_fn, embed_fn, embeddirs_fn, network_chunk)
        rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd)

        ret = {
            "rgb_map": rgb_map,
            "disp_map": disp_map,
            "acc_map": acc_map,
            "rgb0": rgb_map_0,
            "disp0": disp_map_0,
            "acc0": acc_map_0,
            "depth_map": depth_map,
        }
    else:
        ret = {
            "rgb_map": rgb_map,
            "disp_map": disp_map,
            "acc_map": acc_map,
            "depth_map": depth_map,
        }

    if retraw:
        ret["raw"] = raw
    return ret


@torch.no_grad()
def render_image(H, W, focal, c2w, chunk, cfg, near, far, network_fn, network_fine, embed_fn, embeddirs_fn, white_bkgd):
    device = c2w.device
    rays_o, rays_d = get_rays(H, W, focal, c2w, device)

    if cfg.use_viewdirs:
        viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    else:
        viewdirs = None

    if cfg.dataset_type == "llff" and cfg.use_ndc:
        rays_o, rays_d = ndc_rays(H, W, focal, 1.0, rays_o, rays_d)

    rays_o = rays_o.reshape(-1, 3)
    rays_d = rays_d.reshape(-1, 3)
    near_full = near * torch.ones_like(rays_d[:, :1])
    far_full = far * torch.ones_like(rays_d[:, :1])

    if viewdirs is not None:
        viewdirs = viewdirs.reshape(-1, 3)
        rays = torch.cat([rays_o, rays_d, near_full, far_full, viewdirs], -1)
    else:
        rays = torch.cat([rays_o, rays_d, near_full, far_full], -1)

    all_ret = {"rgb_map": [], "disp_map": [], "acc_map": []}
    for i in range(0, rays.shape[0], chunk):
        ret = render_rays(
            rays[i:i + chunk],
            network_fn=network_fn,
            network_fine=network_fine,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            N_samples=cfg.N_samples,
            lindisp=cfg.lindisp,
            perturb=0.0,
            N_importance=cfg.N_importance,
            raw_noise_std=0.0,
            white_bkgd=white_bkgd,
        )
        for k in all_ret:
            all_ret[k].append(ret[k])

    rgb = torch.cat(all_ret["rgb_map"], 0).reshape(H, W, 3)
    disp = torch.cat(all_ret["disp_map"], 0).reshape(H, W)
    acc = torch.cat(all_ret["acc_map"], 0).reshape(H, W)
    return rgb, disp, acc


def normalize(x):
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def poses_avg(poses):
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w


def recenter_poses(poses):
    poses_ = poses.copy()
    bottom = np.reshape([0, 0, 0, 1.0], [1, 4])
    c2w = poses_avg(poses)
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])
    poses_h = np.concatenate([poses[:, :3, :4], bottom], -2)
    poses = np.linalg.inv(c2w) @ poses_h
    poses_[:, :3, :4] = poses[:, :3, :4]
    return poses_


def load_blender_data(basedir, half_res=True, testskip=1, white_bkgd=True):
    splits = ["train", "val", "test"]
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, f"transforms_{s}.json"), "r") as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        skip = 1 if s == "train" or testskip == 0 else testskip
        for frame in meta["frames"][::skip]:
            fname = os.path.join(basedir, frame["file_path"] + ".png")
            img = imageio.imread(fname)
            imgs.append((np.array(img) / 255.0).astype(np.float32))
            poses.append(np.array(frame["transform_matrix"]).astype(np.float32))
        imgs = np.array(imgs)
        poses = np.array(poses)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]
    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(metas["train"]["camera_angle_x"])
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

    if white_bkgd:
        imgs = imgs[..., :3] * imgs[..., -1:] + (1.0 - imgs[..., -1:])
    else:
        imgs = imgs[..., :3]

    if half_res:
        H = H // 2
        W = W // 2
        focal = focal / 2.0
        imgs_half = np.zeros((imgs.shape[0], H, W, 3), dtype=np.float32)
        for i, img in enumerate(imgs):
            imgs_half[i] = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            imgs_half[i] = F.interpolate(imgs_half[i], size=(H, W), mode="area").squeeze(0).permute(1, 2, 0).numpy()
        imgs = imgs_half

    near = 2.0
    far = 6.0
    return imgs, poses, [H, W, focal], i_split, near, far


def _minify_llff(basedir, factors=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, f"images_{r}")
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    from PIL import Image
    imgdir = os.path.join(basedir, "images")
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.lower().endswith(("jpg", "jpeg", "png"))]
    for r in factors:
        outdir = os.path.join(basedir, f"images_{r}")
        os.makedirs(outdir, exist_ok=True)
        for f in imgs:
            img = Image.open(f)
            w, h = img.size
            img = img.resize((w // r, h // r), Image.LANCZOS)
            outpath = os.path.join(outdir, os.path.basename(f).rsplit(".", 1)[0] + ".png")
            img.save(outpath)


def load_llff_data(basedir, factor=4, recenter=True, bd_factor=0.75, llff_hold=8):
    poses_arr = np.load(os.path.join(basedir, "poses_bounds.npy"))
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).astype(np.float32)
    bds = poses_arr[:, -2:].astype(np.float32)

    _minify_llff(basedir, factors=[factor])
    imgdir = os.path.join(basedir, f"images_{factor}")
    if not os.path.exists(imgdir):
        raise FileNotFoundError(imgdir)
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.lower().endswith(("jpg", "jpeg", "png"))]
    imgs = [imageio.imread(f)[..., :3].astype(np.float32) / 255.0 for f in imgfiles]
    imgs = np.stack(imgs, 0)

    sh = imgs[0].shape
    poses[:, 0, 4] = sh[1]
    poses[:, 1, 4] = sh[0]
    poses[:, 2, 4] = poses[:, 2, 4] / factor

    poses_reordered = poses.copy()
    poses_reordered[:, 0, :4] = poses[:, 1, :4]
    poses_reordered[:, 1, :4] = -poses[:, 0, :4]
    poses_reordered[:, 2, :4] = poses[:, 2, :4]
    poses = poses_reordered

    sc = 1.0 / (bds.min() * bd_factor)
    poses[:, :3, 3] *= sc
    bds *= sc

    if recenter:
        poses = recenter_poses(poses)

    H, W, focal = poses[0, :, 4]
    H, W = int(H), int(W)
    i_test = np.arange(imgs.shape[0])[::llff_hold]
    i_val = i_test
    i_train = np.array([i for i in np.arange(imgs.shape[0]) if i not in i_test])

    train_bds = bds[i_train]
    near = np.percentile(train_bds[:, 0], 5)
    far = np.percentile(train_bds[:, 1], 95)

    return imgs, poses[:, :3, :4], [H, W, focal], [i_train, i_val, i_test], near, far


class RaySampler:
    def __init__(self, images, poses, H, W, focal, i_train, cfg: Config, near, far, device):
        self.images = torch.tensor(images, dtype=torch.float32, device=device)
        self.poses = torch.tensor(poses, dtype=torch.float32, device=device)
        self.H = H
        self.W = W
        self.focal = focal
        self.i_train = np.array(i_train)
        self.cfg = cfg
        self.near = near
        self.far = far
        self.device = device
        self._prepare_cache()

    def _prepare_cache(self):
        rays_o_all = []
        rays_d_all = []
        rgb_all = []

        for idx in self.i_train:
            img = self.images[idx]  # (H_img, W_img, 3)
            H_img, W_img = img.shape[:2]

            c2w = self.poses[idx]

            rays_o, rays_d = get_rays(H_img, W_img, self.focal, c2w, self.device)

            if self.cfg.use_ndc and self.cfg.dataset_type == "llff":
                rays_o, rays_d = ndc_rays(H_img, W_img, self.focal, 1.0, rays_o, rays_d)

            rays_o_all.append(rays_o)
            rays_d_all.append(rays_d)
            rgb_all.append(img)

        self.rays_o = torch.stack(rays_o_all, 0)
        self.rays_d = torch.stack(rays_d_all, 0)
        self.rgb = torch.stack(rgb_all, 0)

    def sample_random_rays(self, global_step):
        n_imgs = len(self.i_train)
        img_sel = torch.randint(0, n_imgs, (1,), device=self.device).item()

        H_img = self.rgb.shape[1]
        W_img = self.rgb.shape[2]

        if global_step < self.cfg.precrop_iters:
            dH = int(H_img // 2 * self.cfg.precrop_frac)
            dW = int(W_img // 2 * self.cfg.precrop_frac)

            coords_y, coords_x = torch.meshgrid(
                torch.arange(H_img // 2 - dH, H_img // 2 + dH, device=self.device),
                torch.arange(W_img // 2 - dW, W_img // 2 + dW, device=self.device),
                indexing="ij",
            )
        else:
            coords_y, coords_x = torch.meshgrid(
                torch.arange(H_img, device=self.device),
                torch.arange(W_img, device=self.device),
                indexing="ij",
            )

        coords = torch.stack([coords_y, coords_x], -1).reshape(-1, 2)
        select_inds = torch.randint(0, coords.shape[0], (self.cfg.rays_per_batch,), device=self.device)
        select_coords = coords[select_inds]

        y = select_coords[:, 0]
        x = select_coords[:, 1]

        rays_o = self.rays_o[img_sel, y, x]
        rays_d = self.rays_d[img_sel, y, x]
        target_rgb = self.rgb[img_sel, y, x]

        if self.cfg.use_viewdirs:
            viewdirs = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.near * torch.ones_like(rays_d[:, :1]),
                    self.far * torch.ones_like(rays_d[:, :1]),
                    viewdirs,
                ],
                -1,
            )
        else:
            rays = torch.cat(
                [
                    rays_o,
                    rays_d,
                    self.near * torch.ones_like(rays_d[:, :1]),
                    self.far * torch.ones_like(rays_d[:, :1]),
                ],
                -1,
            )

        return rays, target_rgb

def create_nerf(cfg: Config):
    embed_fn = Embedder(input_dims=3, max_freq_log2=cfg.multires - 1, N_freqs=cfg.multires)
    embeddirs_fn = Embedder(input_dims=3, max_freq_log2=cfg.multires_views - 1, N_freqs=cfg.multires_views)

    input_ch = embed_fn.out_dim
    input_ch_views = embeddirs_fn.out_dim if cfg.use_viewdirs else 0

    model = NeRF(
        D=cfg.netdepth,
        W=cfg.netwidth,
        input_ch=input_ch,
        input_ch_views=input_ch_views,
        skips=cfg.skips,
        use_viewdirs=cfg.use_viewdirs,
        activation_name=cfg.activation_name,
    ).to(cfg.device)

    model_fine = None
    if cfg.N_importance > 0:
        model_fine = NeRF(
            D=cfg.netdepth_fine,
            W=cfg.netwidth_fine,
            input_ch=input_ch,
            input_ch_views=input_ch_views,
            skips=cfg.skips,
            use_viewdirs=cfg.use_viewdirs,
            activation_name=cfg.activation_name,
        ).to(cfg.device)

    grad_vars = list(model.parameters())
    if model_fine is not None:
        grad_vars += list(model_fine.parameters())

    optimizer = torch.optim.Adam(grad_vars, lr=cfg.lr, betas=(0.9, 0.999))

    scheduler = None
    if cfg.lr_scheduler == "exponential":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_gamma)
    elif cfg.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.step_lr_size, gamma=cfg.lr_gamma)
    elif cfg.lr_scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.N_iters, eta_min=cfg.cosine_min_lr)

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and cfg.device.startswith("cuda")))
    return model, model_fine, embed_fn, embeddirs_fn, optimizer, scheduler, scaler


def get_two_stage_lr(base_lr, step, cfg: Config):
    if step < cfg.lr_decay_start:
        return base_lr
    if step < cfg.lr_stage2_start:
        return base_lr * (cfg.lr_gamma ** (step - cfg.lr_decay_start + 1))
    stage1_lr = base_lr * (cfg.lr_gamma ** max(0, cfg.lr_stage2_start - cfg.lr_decay_start))
    return stage1_lr * (cfg.lr_gamma_stage2 ** (step - cfg.lr_stage2_start + 1))


def set_optimizer_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


@torch.no_grad()
def evaluate_split(images, poses, hwf, idxs, cfg, near, far, model, model_fine, embed_fn, embeddirs_fn, white_bkgd):
    device = cfg.device
    psnrs = []

    for i in idxs:
        target = torch.tensor(images[i], dtype=torch.float32, device=device)
        pose = torch.tensor(poses[i], dtype=torch.float32, device=device)

        H_img, W_img = target.shape[:2]

        rgb, _, _ = render_image(
            H=H_img,
            W=W_img,
            focal=hwf[2],
            c2w=pose,
            chunk=1024,
            cfg=cfg,
            near=near,
            far=far,
            network_fn=model,
            network_fine=model_fine,
            embed_fn=embed_fn,
            embeddirs_fn=embeddirs_fn,
            white_bkgd=white_bkgd,
        )

        if rgb.shape != target.shape:
            rgb = rgb.permute(2, 0, 1).unsqueeze(0)
            rgb = F.interpolate(rgb, size=(H_img, W_img), mode="area")
            rgb = rgb.squeeze(0).permute(1, 2, 0)

        loss = img2mse(rgb, target)
        psnr = mse2psnr(loss).item()
        psnrs.append(psnr)

    return float(np.mean(psnrs)) if len(psnrs) > 0 else 0.0


def save_ckpt(path, step, model, model_fine, optimizer, cfg):
    ckpt = {
        "step": step,
        "network_fn_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": asdict(cfg),
    }
    if model_fine is not None:
        ckpt["network_fine_state_dict"] = model_fine.state_dict()
    torch.save(ckpt, path)


def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.output_dir, exist_ok=True)
    expdir = os.path.join(cfg.output_dir, cfg.exp_name)
    os.makedirs(expdir, exist_ok=True)

    if cfg.dataset_type == "blender":
        images, poses, hwf, i_split, near, far = load_blender_data(
            cfg.data_dir,
            half_res=cfg.blender_half_res,
            testskip=cfg.blender_testskip,
            white_bkgd=cfg.blender_white_bkgd,
        )
        white_bkgd = cfg.blender_white_bkgd
    elif cfg.dataset_type == "llff":
        images, poses, hwf, i_split, near, far = load_llff_data(
            cfg.data_dir,
            factor=cfg.llff_factor,
            recenter=cfg.llff_recenter,
            bd_factor=cfg.llff_bd_factor,
            llff_hold=cfg.llff_hold,
        )
        white_bkgd = False
    else:
        raise ValueError("dataset_type must be blender or llff")

    H, W, focal = hwf
    cfg.H = H
    cfg.W = W
    cfg.focal = focal
    cfg.near = near
    cfg.far = far

    i_train, i_val, i_test = i_split

    model, model_fine, embed_fn, embeddirs_fn, optimizer, scheduler, scaler = create_nerf(cfg)

    start = 0
    if cfg.ckpt_path is not None and os.path.exists(cfg.ckpt_path):
        ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["network_fn_state_dict"])
        if model_fine is not None and "network_fine_state_dict" in ckpt:
            model_fine.load_state_dict(ckpt["network_fine_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start = ckpt["step"] + 1

    sampler = RaySampler(images, poses, H, W, focal, i_train, cfg, near, far, cfg.device)

    for i in range(start, cfg.N_iters):
        t0 = time.time()

        if cfg.lr_scheduler == "two_stage":
            current_lr = get_two_stage_lr(cfg.lr, i, cfg)
            set_optimizer_lr(optimizer, current_lr)
        elif cfg.lr_scheduler == "none":
            current_lr = optimizer.param_groups[0]["lr"]
        else:
            current_lr = optimizer.param_groups[0]["lr"]

        rays, target_s = sampler.sample_random_rays(i)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(cfg.amp and cfg.device.startswith("cuda"))):
            ret = render_rays(
                rays,
                network_fn=model,
                network_fine=model_fine,
                embed_fn=embed_fn,
                embeddirs_fn=embeddirs_fn,
                N_samples=cfg.N_samples,
                lindisp=cfg.lindisp,
                perturb=cfg.perturb,
                N_importance=cfg.N_importance,
                raw_noise_std=cfg.raw_noise_std,
                white_bkgd=white_bkgd,
            )

            img_loss = img2mse(ret["rgb_map"], target_s)
            loss = img_loss

            if "rgb0" in ret:
                img_loss0 = img2mse(ret["rgb0"], target_s)
                loss = loss + img_loss0
            psnr = mse2psnr(img_loss)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        if cfg.grad_clip is not None and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + (list(model_fine.parameters()) if model_fine is not None else []), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        dt = time.time() - t0

        if (i + 1) % cfg.i_print == 0:
            print(f"Iter {i + 1}/{cfg.N_iters} | Loss {loss.item():.6f} | PSNR {psnr.item():.4f} | LR {optimizer.param_groups[0]['lr']:.8f} | Time {dt:.3f}s")

        if (i + 1) % cfg.i_weights == 0:
            ckpt_path = os.path.join(expdir, f"{i + 1:06d}.tar")
            save_ckpt(ckpt_path, i, model, model_fine, optimizer, cfg)

        if (i + 1) % cfg.i_testset == 0:
            val_psnr = evaluate_split(images, poses, hwf, i_val, cfg, near, far, model, model_fine, embed_fn, embeddirs_fn, white_bkgd)
            test_psnr = evaluate_split(images, poses, hwf, i_test, cfg, near, far, model, model_fine, embed_fn, embeddirs_fn, white_bkgd)
            print(f"Validation PSNR: {val_psnr:.4f} | Test PSNR: {test_psnr:.4f}")

    final_ckpt = os.path.join(expdir, "final.tar")
    save_ckpt(final_ckpt, cfg.N_iters - 1, model, model_fine, optimizer, cfg)

    print("\nFinal Evaluation Started...\n")

    val_psnr = evaluate_split(
        images, poses, hwf, i_val, cfg, near, far,
        model, model_fine, embed_fn, embeddirs_fn, white_bkgd
    )

    test_psnr = evaluate_split(
        images, poses, hwf, i_test, cfg, near, far,
        model, model_fine, embed_fn, embeddirs_fn, white_bkgd
    )

    print(f"\nFinal Validation PSNR: {val_psnr:.4f}")
    print(f"Final Test PSNR: {test_psnr:.4f}\n")


if __name__ == "__main__":
    cfg = Config(
        dataset_type="llff",
        data_dir="./llff/fern",
        exp_name="llff_fern_relu",
        device="cuda" if torch.cuda.is_available() else "cpu",
        llff_hold=8,
        llff_factor=4,
        llff_recenter=True,
        llff_bd_factor=0.75,
        use_ndc=True,
        N_iters=100,
        rays_per_batch=1024,
        N_samples=64,
        N_importance=64,
        perturb=0.5,
        lindisp=False,
        raw_noise_std=0.0,
        multires=10,
        multires_views=4,
        use_viewdirs=True,
        netdepth=8,
        netwidth=256,
        netdepth_fine=8,
        netwidth_fine=256,
        skips=(4,),
        lr=3e-4,
        lr_scheduler="two_stage",
        lr_gamma=0.995,
        lr_gamma_stage2=0.98,
        lr_decay_start=20,
        lr_stage2_start=70,
        step_lr_size=20,
        cosine_min_lr=5e-5,
        grad_clip=1.0,
        amp=True,
        precrop_iters=10,
        precrop_frac=0.5,
        i_print=1,
        i_weights=10,
        i_testset=101,
        activation_name="relu",
        output_dir="./outputs",
    )
    train(cfg)

    
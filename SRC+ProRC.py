import math
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision.transforms import GaussianBlur

# SRC+ ProgRC : Ref https://arxiv.org/pdf/2512.01510 Semantic-aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation + Progressive Random Convolutions for Single Domain Generalization


# ----------  1.  Gaussian kernel for smoothing masks  ----------
def gaussian_kernel_2d(k: int, sigma: float, device: torch.device):
    x = torch.arange(k, device=device) - k // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g /= g.sum()
    return (g.view(1, 1, k, 1) @ g.view(1, 1, 1, k)).expand(1, 1, k, k)

# ----------  2.  Progressive Block (one step)  ----------
def progressive_convolution_block(x_in: torch.Tensor, w: torch.Tensor, b: torch.Tensor,
                                  gamma: torch.Tensor, beta: torch.Tensor):
    """
    Plain 2-D version (no deformable conv).
    w: (C, C, k, k)   b: (C)   gamma/beta: (C, 1, 1)
    """
    # 1. random conv  (texture diversification)
    x = F.conv2d(x_in, w, b, padding=w.shape[-1]//2)

    # 2. standardisation  (contrast diversification)
    mu  = x.mean(dim=(2, 3), keepdim=True)
    std = x.std(dim=(2, 3), keepdim=True) + 1e-5
    x   = gamma * ((x - mu) / std) + beta

    # 3. non-linearity
    return torch.tanh(x)

# ----------  3.  Semantic-Progressive Augmenter  ----------
class SemanticProgressiveAug(nn.Module):
    """
    2-D implementation of the paper pipeline (Steps 1-5).
    Works for single-channel images.
    """
    def __init__(self, num_classes: int = 2, l_max: int = 3, kernel_size: int = 3,
                 sigma_w: float = 0.05, sigma_b: float = 0.02,
                 sigma_gamma: float = 0.1, sigma_beta: float = 0.05,
                 smooth_kernel: int = 5, smooth_sigma: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.l_max       = l_max
        self.ks          = kernel_size
        self.pad         = kernel_size // 2
        self.sigma_w     = sigma_w
        self.sigma_b     = sigma_b
        self.sigma_g     = sigma_gamma
        self.sigma_beta  = sigma_beta
        self.register_buffer('gauss', gaussian_kernel_2d(smooth_kernel, smooth_sigma,
                                                        torch.device('cpu')).float())
        self.pad_gauss   = smooth_kernel // 2

    @torch.no_grad()
    def get_smooth_masks(self, y: torch.Tensor):
        """y: (B,1,H,W) int  -> list[ (B,1,H,W) float ]  Σ m_c = 1"""
        b, _, h, w = y.shape
        device = y.device
        gauss = self.gauss.to(device)
        masks = []
        for c in range(self.num_classes):
            bin_mask = (y == c).float()
            smooth   = F.conv2d(bin_mask, gauss, padding=self.pad_gauss)
            masks.append(smooth)
        masks = torch.stack(masks, dim=0)          # (C,B,1,H,W)
        masks = masks / (masks.sum(dim=0) + 1e-6)  # normalise
        return [masks[c] for c in range(self.num_classes)]

    @torch.no_grad()
    def sample_params(self, b: int, c: int, device: torch.device):
        """Return dict of random parameters for **one** semantic class."""
        k = self.ks
        return dict(
            w   = torch.randn(c, c, k, k, device=device) * self.sigma_w,
            b   = torch.randn(c,        device=device) * self.sigma_b,
            gamma = torch.randn(c, 1, 1, device=device) * self.sigma_g + 1.0,
            beta  = torch.randn(c, 1, 1, device=device) * self.sigma_beta,
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """
        x: image  (B, C, H, W)   [-1,1] advised
        y: label (B, 1, H, W)    int  0 … num_classes-1
        returns: x_aug  same shape / range as x
        """
        b, c, h, w = x.shape
        device = x.device
        masks = self.get_smooth_masks(y)   # list of (B,1,H,W)
        l = random.randint(1, self.l_max)
        alpha = torch.rand(b, 1, 1, 1, device=device)   # per-sample mixing
        x_aug = torch.zeros_like(x)

        for c_idx in range(self.num_classes):
            params = self.sample_params(b, c, device)
            # progressive stack
            feat = x
            for _ in range(l):
                feat = progressive_convolution_block(feat, **params)
            # mix with original
            x_mix = alpha * feat + (1 - alpha) * x
            # Frobenius rescale
            # Frobenius rescale
            norm_x   = torch.linalg.matrix_norm(x.reshape(b, c, -1), ord='fro', dim=(-2, -1))
            norm_mix = torch.linalg.matrix_norm(x_mix.reshape(b, c, -1), ord='fro', dim=(-2, -1))
            x_mix = x_mix * (norm_x / (norm_mix + 1e-6)).view(b, c, 1, 1)
            # semantic blending
            x_aug += masks[c_idx] * x_mix

        return x_aug

# ----------  4.  Quick visualisation  ----------
def show_pair(orig, aug, mask, title):
    import cv2
    mask_bin = (mask > 0.5).astype(np.uint8)
    edge = cv2.Canny(mask_bin, 0, 1) > 0

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # original
    ax[0].imshow(orig, cmap='gray')
    ax[0].imshow(np.ma.masked_where(~edge, edge), cmap='Greens', alpha=0.9)
    ax[0].set_title(f"{title}  original"); ax[0].axis('off')
    # augmented
    ax[1].imshow(aug, cmap='gray')
    ax[1].imshow(np.ma.masked_where(~edge, edge), cmap='Reds', alpha=0.9)
    ax[1].set_title(f"{title}  semantic-prog"); ax[1].axis('off')
    plt.tight_layout()
    plt.draw()          # non-blocking draw
    plt.pause(0.001)    # let GUI refresh
    # DO NOT call plt.show() here

# ----------  5.  Demo on a folder  ----------
def load_npy(path):
    data = np.load(path, allow_pickle=True).item()
    return data['image'].astype(np.float32), data['mask'].astype(np.int64)

def build_mask(lbl, organ_id):
    m = np.zeros_like(lbl, dtype=np.int64)
    m[lbl == organ_id] = 1
    m[(lbl != 0) & (lbl != organ_id)] = 2
    return m

# ====  CONFIG  ====
device = 'cuda' if torch.cuda.is_available() else 'cpu'
organ_id = 1                       # liver
file_list = [
    "/mnt/scratch1/sachindu/Data/Small/gb/exp2_MRI_CT_1to1/train/CT_s0031_slice_100.npy",
    "/mnt/scratch1/sachindu/Data/Small/gb/exp2_MRI_CT_1to1/train/MRI_s0170_slice_72.npy",
]
# ====  build ONCE outside the loop ====
aug_model = SemanticProgressiveAug(num_classes=2, l_max=3).to(device)

# ====  LOOP  ====
for path in tqdm(file_list, desc="Augmenting"):
    img_np, lbl_np = load_npy(path)
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(device) * 2 - 1   # [-1,1]
    lbl_tensor = torch.from_numpy(build_mask(lbl_np, organ_id)).unsqueeze(0).unsqueeze(0).to(device)

    if lbl_tensor.sum() == 0:
        print("no organ – skipped"); continue

    with torch.no_grad():
        aug_tensor = aug_model(img_tensor, lbl_tensor)   # unique name

    orig = img_tensor.squeeze().cpu().numpy()
    aug_img = aug_tensor.squeeze().cpu().numpy()         # unique numpy name
    mask = (lbl_tensor == 1).float().squeeze().cpu().numpy()
    show_pair(orig, aug_img, mask, os.path.basename(path).split('.')[0])

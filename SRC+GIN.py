import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

#Ref: https://arxiv.org/pdf/2512.01510 Semantic-aware Random Convolution and Source Matching for Domain Generalization in Medical Image Segmentation
# ==========================================
# 1. THE CORRECTED SRC CLASS (PyTorch Native)
# ==========================================
class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel=32, in_channel=1, scale_pool=[1, 3], layer_id=0, use_act=True, requires_grad=False, device='cpu'):
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.scale_pool = scale_pool
        self.use_act = use_act
        self.requires_grad = requires_grad
        self.device = device
        assert requires_grad is False

    def forward(self, x_in):
        nb, nc, nx, ny = x_in.shape
        if nc != self.in_channel:
            raise ValueError(f"Expected {self.in_channel} channels, but got {nc}")
        
        idx_k = torch.randint(high=len(self.scale_pool), size=(1,), dtype=torch.long, device=self.device)
        k = self.scale_pool[idx_k.item()]
        
        dtype = x_in.dtype
        ker = torch.randn([self.out_channel, self.in_channel, k, k], 
                          requires_grad=self.requires_grad, device=self.device, dtype=dtype)
        shift = torch.randn([self.out_channel, 1, 1], 
                            requires_grad=self.requires_grad, device=self.device, dtype=dtype)

        x_conv = F.conv2d(x_in, ker, stride=1, padding=k // 2, dilation=1, groups=1)
        x_conv = x_conv + shift
        
        if self.use_act:
            x_conv = F.leaky_relu(x_conv, negative_slope=0.1) # Corrected slope
            
        return x_conv

class SRC(nn.Module):
    def __init__(self, organ_name='liver', interm_channel=2, scale_pool=[1, 3], n_layer=4, 
                 out_norm='frob', gaussian_sigma=1.0, gaussian_size=5, device='cpu'):
        super(SRC, self).__init__()
        self.organ_name = organ_name
        self.interm_channel = interm_channel
        self.scale_pool = scale_pool
        self.n_layer = n_layer
        self.out_norm = out_norm
        self.device = device
        
        # Pre-calculate Gaussian kernel on GPU
        self.gaussian_kernel = self._create_gaussian_kernel(gaussian_sigma, gaussian_size, device)
        self.gaussian_padding = gaussian_size // 2

        self.organ_src = self._build_src_net(in_channel=1, interm_channel=interm_channel, out_channel=1)
        self.bg_src = self._build_src_net(in_channel=1, interm_channel=interm_channel, out_channel=1)

    def _create_gaussian_kernel(self, sigma, size, device):
        coords = torch.arange(size, dtype=torch.float32, device=device)
        coords -= size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        g_2d = torch.outer(g, g).unsqueeze(0).unsqueeze(0)
        return g_2d

    def _build_src_net(self, in_channel, interm_channel, out_channel):
        layers = nn.ModuleList()
        layers.append(GradlessGCReplayNonlinBlock(out_channel=interm_channel, in_channel=in_channel, scale_pool=self.scale_pool, layer_id=0, device=self.device))
        for ii in range(self.n_layer - 2):
            layers.append(GradlessGCReplayNonlinBlock(out_channel=interm_channel, in_channel=interm_channel, scale_pool=self.scale_pool, layer_id=ii + 1, device=self.device))
        layers.append(GradlessGCReplayNonlinBlock(out_channel=out_channel, in_channel=interm_channel, scale_pool=self.scale_pool, layer_id=self.n_layer - 1, use_act=False, device=self.device))
        return layers

    def _gaussian_smooth_mask(self, binary_mask_tensor):
        if binary_mask_tensor.dim() == 3:
            binary_mask_tensor = binary_mask_tensor.unsqueeze(1)
        
        organ_soft = F.conv2d(binary_mask_tensor.float(), self.gaussian_kernel, stride=1, padding=self.gaussian_padding)
        organ_soft = torch.clamp(organ_soft, 0.0, 1.0)
        bg_soft = 1.0 - organ_soft
        return organ_soft, bg_soft

    def forward(self, image, organ_mask):
        image = image.to(self.device)
        organ_mask = organ_mask.to(self.device)
        B, C, H, W = image.shape
        
        organ_soft, bg_soft = self._gaussian_smooth_mask(organ_mask)
        
        x_organ = image
        for layer in self.organ_src: x_organ = layer(x_organ)
        
        x_bg = image
        for layer in self.bg_src: x_bg = layer(x_bg)
            
        blended = organ_soft * x_organ + bg_soft * x_bg
        alphas = torch.rand(B, 1, 1, 1, device=self.device)
        mixed = torch.lerp(image, blended, alphas)
        
        if self.out_norm == 'frob':
            # flatten H,W -> 2-D tensor (B*C, H*W) for frobenius norm
            in_flat  = image.reshape(B*C, -1)
            mix_flat = mixed.reshape(B*C, -1)

            in_norm  = torch.norm(in_flat,  p='fro', dim=1).view(B, C, 1, 1)
            mix_norm = torch.norm(mix_flat, p='fro', dim=1).view(B, C, 1, 1)

            mixed = mixed * (in_norm / (mix_norm + 1e-8))
        
        return mixed

# ==========================================
# 2. DATA LOADING & VISUALIZATION UTILS
# ==========================================

def load_sample(npy_path):
    """
    Loads image and mask from .npy dictionary.
    Returns: image (np.array), mask (np.array)
    """
    if not os.path.exists(npy_path):
        print(f"Error: {npy_path} not found.")
        # Create dummy data for testing if file missing
        return np.random.randn(256, 256).astype(np.float32), np.zeros((256, 256)).astype(np.uint8)

    data = np.load(npy_path, allow_pickle=True)
    if data.ndim == 0:          # 0-D array wrapping a dict
        data = data.item()

    img = data['image'].astype(np.float32)
    lbl = data['mask'].astype(np.uint8)   # <-- key changed from 'label' to 'mask'
    return img, lbl

def visualize_augmentation(original, augmented, mask, organ_id, title_prefix=""):
    """
    Plots Original vs Augmented with Mask Overlay
    """
    # Normalize images for display (0-1 range)
    def normalize(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    orig_disp = normalize(original.squeeze().cpu().numpy())
    aug_disp = normalize(augmented.squeeze().detach().cpu().numpy())
    mask_disp = mask.squeeze().cpu().numpy()

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # 1. Original + Mask (Green overlay)
    axes[0].imshow(orig_disp, cmap='gray')
    # Create a masked array where 0 is transparent
    mask_overlay = np.ma.masked_where(mask_disp != 1, mask_disp)
    axes[0].imshow(mask_overlay, cmap='Greens', alpha=0.5, vmin=0, vmax=1)
    axes[0].set_title(f"{title_prefix} Original\nOrgan ID: {organ_id}")
    axes[0].axis('off')

    # 2. Augmented + Mask (Red overlay)
    axes[1].imshow(aug_disp, cmap='gray')
    axes[1].imshow(mask_overlay, cmap='Reds', alpha=0.5, vmin=0, vmax=1)
    axes[1].set_title(f"{title_prefix} SRC Augmented\nOrgan ID: {organ_id}")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Running on: {device}")

    # Paths provided by user
    path_ct = "/mnt/scratch1/sachindu/Data/Small/gb/exp2_MRI_CT_1to1/train/CT_s0031_slice_100.npy"
    path_mri = "/mnt/scratch1/sachindu/Data/Small/gb/exp2_MRI_CT_1to1/train/MRI_s0170_slice_72.npy"

    # Initialize SRC Model
    src_model = SRC(device=device).to(device)

    # --- CONFIGURATION ---
    # Change this ID to select different organs dynamically!
    # Common IDs: 0=Bg, 1=Liver, 2=RK, 3=LK, 4=Spleen (Depends on your dataset)
    TARGET_ORGAN_ID = 1 

    # --- PROCESS CT ---
    print(f"Processing CT file: {os.path.basename(path_ct)}")
    img_np, lbl_np = load_sample(path_ct)
    
    # Prepare Tensors [B, C, H, W]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 256, 256]
    
    # Create Binary Mask for the Specific Organ
    # We create a mask where values == TARGET_ORGAN_ID becomes 1, else 0
    mask_binary_np = (lbl_np == TARGET_ORGAN_ID).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_binary_np).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, 256, 256]

    # Run SRC
    if mask_tensor.sum() > 0:
        aug_tensor = src_model(img_tensor, mask_tensor)
        visualize_augmentation(img_tensor, aug_tensor, mask_tensor, TARGET_ORGAN_ID, title_prefix="CT")
    else:
        print(f"Warning: Organ ID {TARGET_ORGAN_ID} not found in this CT slice.")

    # --- PROCESS MRI ---
    print(f"Processing MRI file: {os.path.basename(path_mri)}")
    img_np, lbl_np = load_sample(path_mri)
    
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).unsqueeze(0).to(device)
    mask_binary_np = (lbl_np == TARGET_ORGAN_ID).astype(np.float32)
    mask_tensor = torch.from_numpy(mask_binary_np).unsqueeze(0).unsqueeze(0).to(device)

    # Run SRC
    if mask_tensor.sum() > 0:
        aug_tensor = src_model(img_tensor, mask_tensor)
        visualize_augmentation(img_tensor, aug_tensor, mask_tensor, TARGET_ORGAN_ID, title_prefix="MRI")
    else:
        print(f"Warning: Organ ID {TARGET_ORGAN_ID} not found in this MRI slice.")

import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

# ====================== GIN 1: Random Kernel GIN (Your Version) ====================== #
class RandomKernelGIN(nn.Module):
    def __init__(self, num_layers=3, channels=1, kernel_size=3):
        super().__init__()
        self.num_layers = num_layers
        self.channels = channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):
        out = x
        for _ in range(self.num_layers):
            weight = torch.randn(self.channels, self.channels, self.kernel_size, self.kernel_size, device=x.device)
            out = F.conv2d(out, weight, padding=self.padding)
            out = F.leaky_relu(out, negative_slope=0.2)
        return out

def apply_random_gin(image_tensor, num_layers=3):
    gin = RandomKernelGIN(num_layers=num_layers)
    g_net_x = gin(image_tensor)

    alpha = torch.rand(1, device=image_tensor.device)
    interpolated = alpha * g_net_x + (1 - alpha) * image_tensor

    orig_norm = torch.norm(image_tensor, p='fro')
    aug_norm = torch.norm(interpolated, p='fro') + 1e-6
    return interpolated * (orig_norm / aug_norm)


# ====================== GIN 2: Learnable GIN (Controlled Augmentation) ====================== #
class LearnableGIN(nn.Module):
    def __init__(self, in_channels=1, num_layers=3, num_channels=16):
        super(LearnableGIN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(self._make_layer(in_channels, num_channels))
        for _ in range(num_layers - 2):
            self.layers.append(self._make_layer(num_channels, num_channels))
        self.layers.append(self._make_layer(num_channels, in_channels, final_layer=True))

    def _make_layer(self, in_ch, out_ch, final_layer=False):
        kernel_size = 1 if np.random.rand() < 0.8 else 3
        padding = 0 if kernel_size == 1 else 1
        conv = nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        if final_layer:
            return nn.Sequential(conv, nn.Tanh())
        else:
            return nn.Sequential(conv, nn.BatchNorm2d(out_ch), nn.LeakyReLU(0.2))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

def apply_learnable_gin(image_tensor, num_layers=3):
    gin = LearnableGIN(in_channels=1, num_layers=num_layers).to(image_tensor.device)
    with torch.no_grad():
        augmented_tensor = gin(image_tensor)
    return augmented_tensor

# ====================== Hybrid GIN Wrapper ====================== #
def apply_hybrid_gin(image_tensor, prob_random=0.5):
    """Randomly applies either Random Kernel GIN or Learnable GIN"""
    if np.random.rand() < prob_random:
        return apply_random_gin(image_tensor)  # Apply Random Kernel GIN (Your Version)
    else:
        return apply_learnable_gin(image_tensor)  # Apply Learnable GIN

# Load the NIfTI image
nii_image_path = "nifty image path"
nii_image = nib.load(nii_image_path)
image_data = nii_image.get_fdata()  # Get the image array

# Select a middle slice for visualization (assuming axial view)
mid_slice = image_data.shape[2] // 2
image_slice = image_data[:, :, mid_slice]

# Normalize image for numerical stability
image_slice = (image_slice - np.min(image_slice)) / (np.max(image_slice) - np.min(image_slice) + 1e-5)

# Ensure the image is in the correct format (C, H, W) with 1 channel
image_slice = image_slice[np.newaxis, :, :]  # Convert to (1, H, W)

# Convert to torch tensor and add batch dimension
image_tensor = torch.tensor(image_slice, dtype=torch.float32).unsqueeze(0)  # Shape (1, 1, H, W)

# Move to appropriate device (CPU/GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_tensor = image_tensor.to(device)

# Apply Hybrid GIN augmentation
augmented_image = apply_hybrid_gin(image_tensor)

# Convert back to NumPy for visualization
original_image_np = image_tensor.squeeze().cpu().numpy()
augmented_image_np = augmented_image.squeeze().cpu().detach().numpy()

# Plot the original and augmented images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(original_image_np, cmap="gray")
axes[0].set_title("Original CT Slice")
axes[0].axis("off")

axes[1].imshow(augmented_image_np, cmap="gray")
axes[1].set_title("Augmented CT Slice (Hybrid GIN)")
axes[1].axis("off")

plt.show()


import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

# Adopted from https://github.com/cheng-01037/Causality-Medical-Image-Domain-Generalization/

class GradlessGCReplayNonlinBlock(nn.Module):
    def __init__(self, out_channel = 32, in_channel = 3, scale_pool = [1, 3], layer_id = 0, use_act = True, requires_grad = False, **kwargs):
        """
        Conv-leaky relu layer. Efficient implementation by using group convolutions
        """
        super(GradlessGCReplayNonlinBlock, self).__init__()
        self.in_channel     = in_channel
        self.out_channel    = out_channel
        self.scale_pool     = scale_pool
        self.layer_id       = layer_id
        self.use_act        = use_act
        self.requires_grad  = requires_grad
        assert requires_grad == False

    def forward(self, x_in, requires_grad = False):
        """
        Args:
            x_in: [ nb (original), nc (original), nx, ny ]
        """
        # random size of kernel
        idx_k = torch.randint(high = len(self.scale_pool), size = (1,))
        k = self.scale_pool[idx_k[0]]

        nb, nc, nx, ny = x_in.shape

        ker = torch.randn([self.out_channel * nb, self.in_channel , k, k  ], requires_grad = self.requires_grad  ).cuda()
        shift = torch.randn( [self.out_channel * nb, 1, 1 ], requires_grad = self.requires_grad  ).cuda() * 1.0

        x_in = x_in.view(1, nb * nc, nx, ny)
        x_conv = F.conv2d(x_in, ker, stride =1, padding = k //2, dilation = 1, groups = nb )
        x_conv = x_conv + shift
        if self.use_act:
            x_conv = F.leaky_relu(x_conv)

        x_conv = x_conv.view(nb, self.out_channel, nx, ny)
        return x_conv


class GINGroupConv(nn.Module):
    def __init__(self, out_channel=3, in_channel=3, interm_channel=2, scale_pool=[1, 3], n_layer=4, 
                 out_norm='frob', use_custom_block=True, device='cpu'):
        """
        GIN with configurable convolutional layers.
        - `use_custom_block`: If True, uses `GradlessGCReplayNonlinBlock`; otherwise, `nn.Conv2d`.
        - `device`: Specifies the device to move the model to.
        """
        super(GINGroupConv, self).__init__()
        self.scale_pool = scale_pool  
        self.n_layer = n_layer
        self.out_norm = out_norm
        self.out_channel = out_channel
        self.device = device
        self.use_custom_block = use_custom_block  # Toggle between custom block or standard Conv2d

        # Choose block type
        self.block_type = GradlessGCReplayNonlinBlock if use_custom_block else self._conv_block
        
        self.layers = nn.ModuleList()
        self.layers.append(
            self.block_type(out_channel=interm_channel, in_channel=in_channel, scale_pool=scale_pool, layer_id=0).to(device)
        )
        for ii in range(n_layer - 2):
            self.layers.append(
                self.block_type(out_channel=interm_channel, in_channel=interm_channel, scale_pool=scale_pool, layer_id=ii + 1).to(device)
            )
        self.layers.append(
            self.block_type(out_channel=out_channel, in_channel=interm_channel, scale_pool=scale_pool, layer_id=n_layer - 1, use_act=False).to(device)
        )

    def _conv_block(self, out_channel, in_channel, scale_pool, layer_id, use_act=True):
        """ Standard Conv2D Block (Alternative to GradlessGCReplayNonlinBlock) """
        kernel_size = scale_pool[layer_id % len(scale_pool)]  # Choose from scale_pool
        padding = kernel_size // 2
        conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=padding)
        if use_act:
            return nn.Sequential(conv, nn.BatchNorm2d(out_channel), nn.LeakyReLU(0.2))
        else:
            return conv  # No activation in the final layer

    def forward(self, x_in):
        if isinstance(x_in, list):
            x_in = torch.cat(x_in, dim=0)

        nb, nc, nx, ny = x_in.shape
        alphas = torch.rand(nb, device=self.device).view(-1, 1, 1, 1)  # Shape: (nb, 1, 1, 1)

        x = self.layers[0](x_in)
        for blk in self.layers[1:]:
            x = blk(x)

        # Optimized alpha interpolation using torch.lerp()
        mixed = torch.lerp(x_in, x, alphas)

        # Frobenius norm-based normalization
        if self.out_norm == 'frob':
            _in_frob = torch.norm(x_in.view(nb, nc, -1), dim=(-1, -2), p='fro', keepdim=False).view(nb, 1, 1, 1)
            _self_frob = torch.norm(mixed.view(nb, self.out_channel, -1), dim=(-1, -2), p='fro', keepdim=False).view(nb, 1, 1, 1)
            mixed = mixed * (1.0 / (_self_frob + 1e-5)) * _in_frob

        return mixed


# ====================== GIN Wrapper Function ====================== #
def apply_gin(image_tensor, out_channel=3, in_channel=3, interm_channel=2, scale_pool=[1, 3], n_layer=4, 
              out_norm='frob', use_custom_block=True, device='cpu'):
    """ Apply the optimized GIN augmentation. """
    gin = GINGroupConv(out_channel, in_channel, interm_channel, scale_pool, n_layer, 
                       out_norm, use_custom_block, device).to(device)
    return gin(image_tensor)




# Load the NIfTI image
nii_image_path = "Nifty image path"
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
augmented_image = apply_gin(image_tensor)

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

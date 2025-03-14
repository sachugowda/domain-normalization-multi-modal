import torch
import torch.nn as nn
import numpy as np
import random
import cv2
import abc
import PIL
from torchvision.transforms import transforms

# Adpoted code from https://github.com/liamheng/Non-IID_Medical_Image_Segmentation/


# ---------------------- High-Frequency Component (HFC) Filters ---------------------- #
class HFCFilter(nn.Module, abc.ABC):
    def __init__(self, do_median_padding=True, normalization_percentile_threshold=3, sub_mask=True):
        super(HFCFilter, self).__init__()
        self.do_median_padding = do_median_padding
        self.normalization_percentile_threshold = normalization_percentile_threshold
        self.sub_mask = sub_mask

    @staticmethod
    
    def median_padding(x, mask):
        m_list = []
        batch_size = x.shape[0]
        for i in range(x.shape[1]):
            m_list.append(x[:, i].reshape(batch_size, -1).median(dim=1).values.reshape(batch_size, -1) + 0.2)
        median_tensor = torch.cat(m_list, dim=1).unsqueeze(2).unsqueeze(2)
        mask_x = mask * x
        padding = (1 - mask) * median_tensor
        return padding + mask_x


    @abc.abstractmethod
    def get_hfc(self, x, mask):
        pass

    def forward(self, x, mask):
        if self.do_median_padding:
            x = self.median_padding(x, mask)
        res = self.get_hfc(x, mask)
        # Normalize the result
        for n in range(res.shape[0]):
            for c in range(res.shape[1]):
                temp_res = (res * 256).int().float() / 256
                res_min, res_max = np.percentile(temp_res[n, c].detach().cpu().numpy(),
                                                 self.normalization_percentile_threshold), np.percentile(
                    temp_res[n, c].detach().cpu().numpy(), 100 - self.normalization_percentile_threshold)
                res[n, c] = (res[n, c] - res_min) / (res_max - res_min)
        if self.sub_mask:
            res = res * mask
        return res


# ---------------------- Gaussian High-Pass Filter ---------------------- #
class GaussianKernel(nn.Module):
    def __init__(self, kernel_len, nsig=20):
        super(GaussianKernel, self).__init__()
        self.kernel_len = kernel_len
        kernel = cv2.getGaussianKernel(kernel_len, nsig) * cv2.getGaussianKernel(kernel_len, nsig).T
        kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.padding = torch.nn.ReplicationPad2d(int(self.kernel_len / 2))

    def forward(self, x):
        x = self.padding(x)
        res = [F.conv2d(x[:, i:i + 1], self.weight) for i in range(x.shape[1])]
        return torch.cat(res, dim=1)


class GaussianHFCFilter(HFCFilter):
    def __init__(self, filter_width=23, nsig=9, ratio=4, sub_low_ratio=1, do_median_padding=True,
                 normalization_percentile_threshold=3, sub_mask=True):
        super(GaussianHFCFilter, self).__init__(do_median_padding, normalization_percentile_threshold, sub_mask)
        self.gaussian_filter = GaussianKernel(filter_width, nsig=nsig)
        self.ratio = ratio
        self.sub_low_ratio = sub_low_ratio

    def get_hfc(self, x, mask):
        gaussian_output = self.gaussian_filter(x)
        return self.ratio * (x - self.sub_low_ratio * gaussian_output)


# ---------------------- Fourier Butterworth High-Pass Filter ---------------------- #

class FourierButterworthHFCFilter(HFCFilter):
    def __init__(self, image_size=(512, 512), butterworth_d0_ratio=0.05, butterworth_n=1,
                 do_median_padding=True, normalization_percentile_threshold=3, sub_mask=True):
        super(FourierButterworthHFCFilter, self).__init__(do_median_padding, normalization_percentile_threshold, sub_mask)
        self.image_size = image_size

        d0 = int(max((butterworth_d0_ratio * min(*image_size)) // 2, 1))
        corners = ((0, 0), (0, image_size[1] - 1), (image_size[0] - 1, image_size[1] - 1), (image_size[0] - 1, 0))
        self.filter_map = np.zeros(image_size, dtype=np.float32)

        for i in range(image_size[0]):
            for j in range(image_size[1]):
                d = min([np.sqrt((i - x) ** 2 + (j - y) ** 2) for x, y in corners])
                self.filter_map[j, i] = 1 / (1 + (d0 / (d + 1)) ** (2 * butterworth_n))

        # Convert filter map to PyTorch tensor
        self.filter_map = nn.Parameter(torch.FloatTensor(self.filter_map), requires_grad=False)

    def get_hfc(self, x, mask=None):
        """
        Applies high-frequency filtering using Fourier Butterworth.
        Ensures filter map matches input image size.
        """
        # Compute Fourier transform
        x_fft = torch.fft.fft2(x)

        # Ensure self.filter_map matches input image size
        filter_map_resized = F.interpolate(
            self.filter_map.unsqueeze(0).unsqueeze(0),  # Add batch & channel dim
            size=(x.shape[-2], x.shape[-1]),  # Resize to input image size
            mode="bilinear",
            align_corners=False
        ).squeeze(0).squeeze(0)  # Remove batch & channel dim

        # Apply filter
        x_fft_temp = x_fft * filter_map_resized.to(x.device)

        # Inverse FFT to get the filtered image
        return torch.abs(torch.fft.ifft2(x_fft_temp))




# ---------------------- FMAug (Final Augmentation Class) ---------------------- #

class FMAug:
    def __init__(self, hfc_type='butterworth_all_channels', device='cuda'):
        """
        Frequency-based augmentation for domain generalization.
        Applies high-frequency filtering without requiring a mask.
        """
        self.device = torch.device(device)
        self.hfc_type = hfc_type
        self.hfc_list = nn.ModuleList()

        if hfc_type in ['butterworth_all_channels', 'butterworth_per_channel']:
            d0_list = np.linspace(0.01, 0.1, 10)  # Different frequency cutoffs
            for d0 in d0_list:
                self.hfc_list.append(
                    FourierButterworthHFCFilter(
                        image_size=(512, 512),
                        butterworth_d0_ratio=d0.item(),
                        butterworth_n=2
                    ).to(self.device)
                )

    def do_hfc(self, img, mask=None):
        """
        Applies a randomly chosen high-frequency filter to the image.
        If mask is None, it creates a dummy mask (all ones).
        """
        if mask is None:
            mask = torch.ones_like(img).to(self.device)  # Create a dummy mask
        
        return random.choice(self.hfc_list)(img, mask)

    def __call__(self, image):
        """
        Applies frequency-based augmentation only to the image.
        
        :param image: Input image tensor (B, C, H, W)
        :return: Augmented image
        """
        image = image.to(self.device)

        # Apply high-frequency augmentation
        augmented_image = self.do_hfc(image)

        return augmented_image


# ---------------------- Load NIfTI Image ---------------------- #
def load_nifti_image(nifti_path):
    """
    Load a NIfTI (.nii.gz) image and return it as a PyTorch tensor.
    """
    nii_img = nib.load(nifti_path)
    img_data = nii_img.get_fdata()

    # Normalize image intensity to [0, 1]
    img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))

    # Convert to PyTorch tensor with shape (1, 1, H, W)
    img_tensor = torch.FloatTensor(img_data).unsqueeze(0).unsqueeze(0).cuda()

    return img_tensor, nii_img.affine, img_data  # Return original numpy array too

# ---------------------- Apply Augmentation ---------------------- #
def apply_fmaug(image_tensor):
    """
    Apply FMAug to the image only.
    """
    # Initialize FMAug
    fmaug = FMAug(hfc_type='butterworth_all_channels', device='cuda')

    # Apply augmentation (No mask needed)
    augmented_image = fmaug(image_tensor)  # FIXED: Only pass image

    return augmented_image

# ---------------------- Display Images ---------------------- #
def display_images(original, augmented):
    """
    Display the original and augmented images side by side.
    Selects the middle slice from the 3D CT volume for visualization.
    """
    middle_slice = original.shape[-1] // 2  # Choose the middle slice along depth (D)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original[:, :, middle_slice], cmap='gray')  # Show middle slice
    axes[0].set_title("Original CT Image")
    axes[0].axis("off")

    axes[1].imshow(augmented[:, :, middle_slice], cmap='gray')  # Show same slice after augmentation
    axes[1].set_title("Augmented CT Image (FMAug Applied)")
    axes[1].axis("off")

    plt.show()


# ---------------------- Run Augmentation ---------------------- #
if __name__ == "__main__":
    # Define input path
    nifti_path = "/nifty image path/"

    # Load CT image
    image_tensor, affine, original_numpy = load_nifti_image(nifti_path)

    # Apply augmentation
    augmented_tensor = apply_fmaug(image_tensor)

    # Convert augmented tensor to numpy for display
    augmented_numpy = augmented_tensor.squeeze().cpu().numpy()

    # Display before and after images
    display_images(original_numpy, augmented_numpy)


import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100, padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.

    This is CRITICAL for neural data as it reduces high-frequency noise while preserving signal.

    Args:
        inputs (tensor: B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
        smooth_kernel_std (float): Standard deviation of the Gaussian smoothing kernel.
        smooth_kernel_size (int): Size of the smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.

    Returns:
        smoothed (tensor: B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]


def apply_data_augmentations(features, n_time_steps, mode='train', transform_args=None, device='cuda'):
    """
    Apply various augmentations and smoothing to neural data.
    Performing augmentations on GPU is much faster than CPU.

    Args:
        features (tensor): [batch, time, channels] - Neural activity data
        n_time_steps (tensor): [batch] - Actual length of each trial (before padding)
        mode (str): 'train' or 'val' - Augmentations only applied during training
        transform_args (dict): Dictionary of augmentation parameters
        device (str): Device to perform operations on

    Returns:
        features (tensor): Augmented/smoothed features
        n_time_steps (tensor): Updated time steps (may change due to random_cut)
    """
    if transform_args is None:
        # Default augmentation parameters from baseline
        transform_args = {
            'white_noise_std': 1.0,
            'constant_offset_std': 0.2,
            'random_walk_std': 0.0,
            'random_walk_axis': -1,
            'static_gain_std': 0.0,
            'random_cut': 3,
            'smooth_data': True,
            'smooth_kernel_std': 2,
            'smooth_kernel_size': 100,
        }

    data_shape = features.shape
    batch_size = data_shape[0]
    channels = data_shape[-1]

    # We only apply these augmentations in training
    if mode == 'train':
        # 1. Add static gain noise (multiplicative noise per channel)
        if transform_args.get('static_gain_std', 0) > 0:
            warp_mat = torch.tile(torch.unsqueeze(torch.eye(channels), dim=0), (batch_size, 1, 1))
            warp_mat = warp_mat.to(device)
            warp_mat += torch.randn_like(warp_mat, device=device) * transform_args['static_gain_std']
            features = torch.matmul(features, warp_mat)

        # 2. Add white noise (independent Gaussian noise per timestep/channel)
        if transform_args.get('white_noise_std', 0) > 0:
            features += torch.randn(data_shape, device=device) * transform_args['white_noise_std']

        # 3. Add constant offset noise (constant per trial, varies across channels)
        if transform_args.get('constant_offset_std', 0) > 0:
            features += torch.randn((batch_size, 1, channels), device=device) * transform_args['constant_offset_std']

        # 4. Add random walk noise (cumulative noise over time)
        if transform_args.get('random_walk_std', 0) > 0:
            features += torch.cumsum(
                torch.randn(data_shape, device=device) * transform_args['random_walk_std'],
                dim=transform_args['random_walk_axis']
            )

        # 5. Randomly cutoff part of the data timecourse (removes initial timesteps)
        if transform_args.get('random_cut', 0) > 0:
            cut = np.random.randint(0, transform_args['random_cut'])
            if cut > 0:
                features = features[:, cut:, :]
                n_time_steps = n_time_steps - cut

    # 6. Apply Gaussian smoothing to data
    # This is done in both training and validation (CRITICAL for neural data)
    if transform_args.get('smooth_data', True):
        features = gauss_smooth(
            inputs=features,
            device=device,
            smooth_kernel_std=transform_args.get('smooth_kernel_std', 2),
            smooth_kernel_size=transform_args.get('smooth_kernel_size', 100),
        )

    return features, n_time_steps
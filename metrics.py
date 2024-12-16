import torch
from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(img_gt, img_pred):
    img_gt = img_gt.astype(np.float64)
    img_pred = img_pred.astype(np.float64)

    # Compute SSIM per channel
    ssim_r, _ = ssim(img_gt[:,:,0], img_pred[:,:,0], data_range=img_pred[:,:,0].max() - img_pred[:,:,0].min(), full=True)
    ssim_g, _ = ssim(img_gt[:,:,1], img_pred[:,:,1], data_range=img_pred[:,:,1].max() - img_pred[:,:,1].min(), full=True)
    ssim_b, _ = ssim(img_gt[:,:,2], img_pred[:,:,2], data_range=img_pred[:,:,2].max() - img_pred[:,:,2].max(), full=True)

    # Average across channels
    avg_ssim = (ssim_r + ssim_g + ssim_b) / 3.0
    return avg_ssim


if __name__ == '__main__':
    gt = torch.rand(3, 256, 256).detach().cpu().numpy()
    pred = torch.rand(3, 256, 256).detach().cpu().numpy()
    print(compute_ssim(gt, pred))
import os
import piq
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

root_dir = "/data_western_digital/development/computer_tomography/model/MCG_diffusion/results/SV-CT/m6.0"

result = []

image_idx = sorted(os.listdir(root_dir))

for idx in tqdm(image_idx):
    if int(idx) < 268 or int(idx) > 325 or ("metrics_results" in idx): pass
    else:
        recon_dir = os.path.join(root_dir, idx, "MCG", "recon", f"0{idx}_clip.png")
        gt_dir = os.path.join(root_dir, idx, "MCG", "label", f"0{idx}_clip.png")

        recon = cv2.imread(recon_dir, cv2.IMREAD_GRAYSCALE)
        gt = cv2.imread(gt_dir, cv2.IMREAD_GRAYSCALE)

        recon = recon.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        ssim_val = ssim(gt, recon, data_range=1.0)
        psnr_val = psnr(gt, recon, data_range=1.0)

        recon = torch.tensor(recon).unsqueeze(0).unsqueeze(0)
        gt = torch.tensor(gt).unsqueeze(0).unsqueeze(0)

        fsim_val = piq.fsim(gt, recon, data_range=1.0, chromatic=False).item()

        result.append(
            {
                'idx': idx,
                'ssim': ssim_val,
                'psnr': psnr_val,
                'fsim': fsim_val
            }
        )

df = pd.DataFrame(result)
csv_path = os.path.join(root_dir, "metrics_results_MCG.csv")
df.to_csv(csv_path, index=False)
print(f"The result has been saved to {csv_path}")

summary = {}
for metric in ['ssim', 'psnr', 'fsim']:
    summary[metric + '_mean'] = df[metric].mean()
    summary[metric + '_max'] = df[metric].max()
    summary[metric + '_max_file'] = df.loc[df[metric].idxmax(), 'idx']
    summary[metric + '_min'] = df[metric].min()
    summary[metric + '_min_file'] = df.loc[df[metric].idxmin(), 'idx']

for k, v in summary.items():
    print(f"{k}: {v}")
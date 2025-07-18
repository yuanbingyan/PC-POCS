import torch
from models.ema import ExponentialMovingAverage

import numpy as np
import controllable_generation

from utils import restore_checkpoint, clear, \
    lambda_schedule_const, lambda_schedule_linear
from pathlib import Path
from models import utils as mutils
from sde_lib import VESDE
from sampling import (ReverseDiffusionPredictor,
                      LangevinCorrector)
import datasets

# for radon
from physics.ct import CT
import matplotlib.pyplot as plt

from models import ncsnpp
import os
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

# Configurations
solver = 'MCG'
# solver = 'song'
# solver = 'POCS'
config_name = 'AAPM_256_ncsnpp_continuous'
sde = 'VESDE'
num_scales = 2000
ckpt_num = 48
N = num_scales

root = './samples/test'

# Parameters for the inverse problem
sparsity = 12
num_proj = 180 // sparsity  # 180 / 6 = 30
det_spacing = 1.0
size = 256
det_count = int((size * (2*torch.ones(1)).sqrt()).ceil()) # ceil(size * \sqrt{2})

schedule = 'linear'
start_lamb = 1.0
end_lamb = 0.6

num_posterior_sample = 1

if schedule == 'const':
    lamb_schedule = lambda_schedule_const(lamb=start_lamb)
elif schedule == 'linear':
    lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)
else:
    NotImplementedError(f"Given schedule {schedule} not implemented yet!")

freq = 1

if sde.lower() == 'vesde':
    from configs.ve import AAPM_256_ncsnpp_continuous as configs
    ckpt_filename = f"./checkpoints/{config_name}/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(sigma_min=config.model.sigma_min, sigma_max=config.model.sigma_max, N=config.model.num_scales)
    sde.N = N
    sampling_eps = 1e-5

batch_size = 4
config.training.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1

batch_size = 4
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config).to('cuda:0')
score_model = torch.nn.DataParallel(score_model)
score_model = score_model.to(config.device)

# optimizer = get_optimizer(config, score_model.parameters())
ema = ExponentialMovingAverage(score_model.module.parameters(),
                               decay=config.model.ema_rate)
# state = dict(step=0, optimizer=optimizer,
#              model=score_model, ema=ema)
state = dict(step=0, model=score_model.module, ema=ema)

state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.module.parameters())

for batch_start in range(301, 326, batch_size):
    batch_indices = list(range(batch_start, min(batch_start + batch_size, 326)))
    imgs, sinograms, masks, save_paths = [], [], [], []

    for idx in batch_indices:
        filename = Path(root) / (str(idx).zfill(4) + '.npy')
        # Specify save directory for saving generated samples
        save_root = Path(f'./results/SV-CT/m{180/sparsity}/{idx}/{solver}')
        save_root.mkdir(parents=True, exist_ok=True)

        irl_types = ['input', 'recon', 'label']
        for t in irl_types:
            save_root_f = save_root / t
            save_root_f.mkdir(parents=True, exist_ok=True)

        # Read data
        img = torch.from_numpy(np.load(filename))
        h, w = img.shape
        img = img.view(1, 1, h, w)
        img = img.to(config.device)
        imgs.append(img)
        save_paths.append(save_root)


        # full
        angles = np.linspace(0, np.pi, 180, endpoint=False)
        radon = CT(img_width=size, radon_view=num_proj, circle=False, device=config.device)
        radon_all = CT(img_width=size, radon_view=180, circle=False, device=config.device)

        mask = torch.zeros([batch_size, 1, det_count, 180]).to(config.device)
        mask[..., ::sparsity] = 1

        sinogram = radon.A(img)
        sinograms.append(sinogram)
        masks.append(mask.clone())

        plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}.png', clear(img), cmap='gray')
        plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}_clip.png', np.clip(clear(img), 0.1, 1.0), cmap='gray')

        # Dimension Reducing (DR)
        sinogram = radon.A(img)

        # Dimension Preserving (DP)
        sinogram_full = radon_all.A(img) * mask

        # FBP
        fbp = radon.A_dagger(sinogram)
        plt.imsave(str(save_root / 'input' / f'FBP.png'), clear(fbp), cmap='gray')
        imgs_batch = torch.cat(imgs, dim=0)
        sinograms_batch = torch.cat(sinograms, dim=0)
        masks_batch = torch.cat(masks, dim=0)

        if solver == 'MCG':
            pc_MCG = controllable_generation.get_pc_radon_MCG(sde,
                                                      predictor, corrector,
                                                      inverse_scaler,
                                                      snr=snr,
                                                      n_steps=n_steps,
                                                      probability_flow=probability_flow,
                                                      continuous=config.training.continuous,
                                                      denoise=True,
                                                      radon=radon,
                                                      radon_all=radon_all,
                                                      weight=0.1,
                                                      save_progress=False,
                                                      save_root=save_root,
                                                      lamb_schedule=lamb_schedule,
                                                      mask=mask)
            x_batch = pc_MCG(score_model, scaler(imgs_batch), measurement=sinograms_batch)
        elif solver == 'song':
            pc_song = controllable_generation.get_pc_radon_song(sde,
                                                        predictor, corrector,
                                                        inverse_scaler,
                                                        snr=snr,
                                                        n_steps=n_steps,
                                                        probability_flow=probability_flow,
                                                        continuous=config.training.continuous,
                                                        save_progress=False,
                                                        save_root=save_root,
                                                        denoise=True,
                                                        radon=radon_all,
                                                        lamb=0.7)
            x = pc_song(score_model, scaler(img), mask, measurement=sinogram_full)
        elif solver == 'POCS':
            pc_POCS = controllable_generation.get_pc_radon_POCS(
            sde,
            predictor,
            corrector,
            inverse_scaler,
            snr=snr,
            n_steps=n_steps,
            probability_flow=probability_flow,
            continuous=config.training.continuous,
            denoise=True,
            eps=1e-5,
            radon=radon,
            save_progress=False,
            save_root=save_root,
            lamb_schedule=lamb_schedule,
            measurement_noise=False,
            final_consistency=False
            )
            x = pc_POCS(score_model, scaler(img), measurement=sinogram)

    # Recon
    for i, idx in enumerate(batch_indices):
        x = x_batch[i: i + 1]
        save_root = save_paths[i]
        plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}.png'), clear(x), cmap='gray')
        plt.imsave(str(save_root / 'recon' / f'{str(idx).zfill(4)}_clip.png'), np.clip(clear(x), 0.1, 1.0), cmap='gray')
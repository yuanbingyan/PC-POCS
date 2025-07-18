import os
import torch
import datasets
import numpy as np
import matplotlib.pyplot as plt

import controllable_generation

from physics.ct import CT
from pathlib import Path
from sde_lib import VESDE
from models import utils as mutils
from models.ema import ExponentialMovingAverage
from configs.ve import AAPM_256_ncsnpp_continuous as configs
from sampling import ReverseDiffusionPredictor, LangevinCorrector
from utils import restore_checkpoint, clear, lambda_schedule_const, lambda_schedule_linear

os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"

# Configuration

# diffusion parameters
solver = "MCG"
config_name = "AAPM_256_ncsnpp_continuous"
sde = "VESDE"
num_scales = int(2e3)
ckpt_num = 48
N = num_scales
root = './samples/test'
size = 256
if sde.lower() == "vesde":
    ckpt_filename = f"./checkpoints/{config_name}/checkpoint_{ckpt_num}.pth"
    config = configs.get_config()
    config.model.num_scales = N
    sde = VESDE(
        sigma_min = config.model.sigma_min,
        sigma_max = config.model.sigma_max,
        N = config.model.num_scales
    )
    sde.N = N
    sampling_eps = 1e-5

batch_size = 1
config.training.batch_size = batch_size
config.eval.batch_size = batch_size
predictor = ReverseDiffusionPredictor
corrector = LangevinCorrector
probability_flow = False
snr = 0.16
n_steps = 1
random_seed = 0

sigmas = mutils.get_sigmas(config)
scaler = datasets.get_data_scaler(config)
inverse_scaler = datasets.get_data_inverse_scaler(config)
score_model = mutils.create_model(config)
ema = ExponentialMovingAverage(score_model.parameters(), decay=config.model.ema_rate)
state = dict(step=0, model=score_model, ema=ema)
state = restore_checkpoint(ckpt_filename, state, config.device, skip_sigma=True, skip_optimizer=True)
ema.copy_to(score_model.parameters())

# inverse problem parameters
sparsity = 12
num_projections = 180 // sparsity
det_spacing = 1.0
det_count = int((size * (2 * torch.ones(1)).sqrt()).ceil())

schedule = 'linear'
start_lamb = 1.0
end_lamb = 0.6

num_posterior_sample = 1

if schedule == 'const':
    lamb_schedule = lambda_schedule_const(lamb=start_lamb)
elif schedule == 'linear':
    lamb_schedule = lambda_schedule_linear(start_lamb=start_lamb, end_lamb=end_lamb)

freq = 1

for idx in range(268, 326, 1):
    filename = Path(root) / (str(idx).zfill(4) + '.npy')
    save_root = Path(f'./results/SV-CT/m{180/sparsity}/{idx}/{solver}')
    save_root.mkdir(parents=True, exist_ok=True)

    irl_types = ['input', 'recon', 'label']
    for t in irl_types:
        save_root_f = save_root / t
        save_root_f.mkdir(parents=True, exist_ok=True)

    img = torch.from_numpy(np.load(filename))
    h, w = img.shape
    img = img.view(1, 1, h, w)
    img = img.to(config.device)

    plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}.png', clear(img), cmap='gray')
    plt.imsave(save_root / 'label' / f'{str(idx).zfill(4)}_clip.png', np.clip(clear(img), 0.1, 1.0), cmap='gray')

    angles = np.linspace(0, np.pi, 180, endpoint=False)
    radon = CT(img_width=size, radon_view=num_projections, circle=False, device=config.device)
    radon_all = CT(img_width=size, radon_view=180, circle=False, device=config.device)

    mask = torch.zeros([batch_size, 1, det_count, 180]).to(config.device)
    mask[..., ::sparsity] = 1

    sinogram = radon.A(img)
    sinogram_full = radon_all.A(img) * mask
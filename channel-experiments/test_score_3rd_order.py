#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch, sys, os, itertools, copy, argparse
sys.path.append('./')

from tqdm import tqdm as tqdm
from ncsnv2.models.ncsnv2 import NCSNv2Deepest
from data.loaders              import Channels
from torch.utils.data     import DataLoader
from matplotlib import pyplot as plt

# Args
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--train', type=str, default='CDL-C')
parser.add_argument('--test', type=str, default='CDL-C')
parser.add_argument('--save_channels', type=int, default=0)
parser.add_argument('--spacing', nargs='+', type=float, default=[0.5])
parser.add_argument('--pilot_alpha', nargs='+', type=float, default=[0.6])
parser.add_argument('--spacing_classes', nargs='+', type=float, default=40)
parser.add_argument('-d', '--discretization_scheme', type=str , default='BCOABC')
args = parser.parse_args()

# Disable TF32 due to potential precision issues
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32       = False
torch.backends.cudnn.benchmark        = True
# GPU
os.environ["CUDA_DEVICE_ORDER"]    = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Target file
target_dir  = './models/score/%s' % args.train
target_file = os.path.join(target_dir, 'final_model.pt')
contents    = torch.load(target_file)
config      = contents['config']

# Default hyper-parameters for pilot_alpha = 0.6, all SNR points
if args.train == 'CDL-A':
    # !!! Not to be confused with 'pilot_alpha' that denotes fraction of pilots
    alpha_step = 2.2e-10 # 'alpha' in paper Algorithm 1
    beta_noise = 0.01  # 'beta' in paper Algorithm 1
elif args.train == 'CDL-B':
    alpha_step = 2.2e-10
    beta_noise = 0.01
elif args.train == 'CDL-C':
    alpha_step = 2.2e-10
    beta_noise = 0.01
elif args.train == 'CDL-D':
    alpha_step = 2.2e-10
    beta_noise = 0.01
elif args.train == 'Mixed':
    alpha_step = 2.2e-10
    beta_noise = 0.01

# Instantiate model
diffuser = NCSNv2Deepest(config)
diffuser = diffuser.cuda()
# Load weights
diffuser.load_state_dict(contents['model_state']) 
diffuser.eval()

# Train and validation seeds
train_seed, val_seed = 1234, 4321
# Get training dataset for normalization
config.data.channel = args.train
dataset = Channels(train_seed, config, norm=config.data.norm_channels)

# Range of SNR, test channels and hyper-parameters
config.sampling.steps_each = 3
snr_range          = np.arange(-10, 30, 5)
spacing_range      = np.asarray(args.spacing) # From a pre-defined index
pilot_alpha_range  = np.asarray(args.pilot_alpha)
noise_range        = 10 ** (-snr_range / 10.) * config.data.image_size[1]
# Number of validation channels
num_channels = 100
    

# Global results
nmse_log = np.zeros((len(spacing_range), len(pilot_alpha_range),
                     len(snr_range), int(round(config.model.num_classes / args.spacing_classes) * \
                   config.sampling.steps_each), num_channels))


result_dir = './results/score/train-%s_test-%s' % (
    args.train, args.test)
os.makedirs(result_dir, exist_ok=True)

# Wrap sparsity, steps and spacings
meta_params = itertools.product(spacing_range, pilot_alpha_range)

# Hyperparameters of 2nd-order Langevin
gamma = 1
M_inv_list = [3.5, 3, 2, 2, 2, 2, 2, 2]
temp = 0.01
tau = [1]#5 levels of noise
lambd = [2]#5 levels of noise


# For each hyper-combo
for meta_idx, (spacing, pilot_alpha) in tqdm(enumerate(meta_params)):
    # Unwrap indices
    spacing_idx, pilot_alpha_idx = np.unravel_index(
        meta_idx, (len(spacing_range), len(pilot_alpha_range)))
    
    # Get validation dataset
    val_config = copy.deepcopy(config)
    val_config.data.channel      = args.test
    val_config.data.spacing_list = [spacing]
    val_config.data.num_pilots   = int(np.floor(config.data.image_size[1] * pilot_alpha))
    val_dataset = Channels(val_seed, val_config, norm=[dataset.mean, dataset.std])
    val_loader  = DataLoader(val_dataset, batch_size=num_channels,
        shuffle=False, num_workers=0, drop_last=True)
    val_iter = iter(val_loader)
    print('There are %d validation channels' % len(val_dataset))
        
    # Get all validation data explicitly
    val_sample = next(val_iter)
    _, val_P, _ = \
        val_sample['H'].cuda(), val_sample['P'].cuda(), val_sample['Y'].cuda()
    # Transposed pilots
    val_P = torch.conj(torch.transpose(val_P, -1, -2))
    val_H_herm = val_sample['H_herm'].cuda()
    val_H = val_H_herm[:, 0] + 1j * val_H_herm[:, 1]
    # Initial estimates
    init_val_H = torch.randn_like(val_H)
    init_val_v = torch.randn_like(val_H)
    init_val_sigma_prony = torch.randn_like(val_H)

    # For each SNR value
    for snr_idx, local_noise in tqdm(enumerate(noise_range)):
        val_Y     = torch.matmul(val_P, val_H)
        val_Y     = val_Y + np.sqrt(local_noise) * torch.randn_like(val_Y) 
        current   = init_val_H.clone()
        y         = val_Y
        forward   = val_P
        forward_h = torch.conj(torch.transpose(val_P, -1, -2))
        norm      = [0., 1.]
        oracle    = val_H # Ground truth channels
        vT        = init_val_v.clone()   
        sigma_prony = init_val_sigma_prony.clone()

        # Count every step
        trailing_idx = 0
        
        M_inv = M_inv_list[snr_idx]
        MM = 1/M_inv

        for step_idx in tqdm(range(0, val_config.model.num_classes, args.spacing_classes)):
            # Compute current step size and noise power
            current_sigma = diffuser.sigmas[step_idx].item()
            # Labels for diffusion model
            labels = torch.ones(init_val_H.shape[0]).cuda() * step_idx
            labels = labels.long()
            
            # Compute annealed step size
            alpha = alpha_step * (current_sigma / val_config.model.sigma_end) ** 2
            
           
            # Compute score using real view of data
            current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
            with torch.no_grad():
                score = diffuser(current_real, labels)
            # View as complex
            score = torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
            
            # Compute gradient for measurements in un-normalized space
            meas_grad = torch.matmul(forward_h, torch.matmul(forward, current) - y) / \
                        (local_noise/2. + current_sigma ** 2)
   
            
            # For each step spent at that noise level
            for inner_idx in range(val_config.sampling.steps_each):
                if args.discretization_scheme == 'BACOCAB':
                    theta = np.exp(-alpha * lambd[0] / tau[0]) 
                    alpha_prony = np.sqrt(1-theta**2)
                    # Apply update of momentum
                    vT = vT + alpha / 2  * (score - meas_grad)

                    #Apply update of position
                    current = current + M_inv * (alpha / 2) * vT

                    vT = vT + alpha / 2  * sigma_prony
                    
                    #Apply update of sigma_prony
                    sigma_prony = theta * sigma_prony - \
                            (tau[0] / lambd[0]) * (1 - theta) * vT + \
                            np.sqrt(MM * beta_noise * temp * (lambd[0] / tau[0])) * alpha_prony * torch.randn_like(current)
                    
                    vT = vT + alpha / 2  * sigma_prony
                    
                    #Apply update of position
                    current = current + M_inv * (alpha / 2) * vT

                    # Compute score using real view of data
                    current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                    with torch.no_grad():
                        score = diffuser(current_real, labels)
                    # View as complex
                    score = torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                    
                    # Compute gradient for measurements in un-normalized space
                    meas_grad = torch.matmul(forward_h, torch.matmul(forward, current) - y) / \
                                    (local_noise/2. + current_sigma ** 2)
                   
                    ## 
                    vT = vT + \
                        alpha / 2  * (score - meas_grad)
                            
                    
                elif args.discretization_scheme == 'BAOAB':
                    theta = np.exp(-alpha * lambd[0] / tau[0]) 
                    alpha_prony = np.sqrt(1-theta**2)
                    # Apply update of momentum
                    vT = vT + alpha / 2  * (score - meas_grad) + alpha / 2 * sigma_prony

                    #Apply update of position
                    current = current + M_inv * alpha * vT
                    
                    #Apply update of sigma_prony
                    sigma_prony = theta * sigma_prony - \
                            (tau[0] / lambd[0]) * (1 - theta) * vT + \
                            np.sqrt(MM * beta_noise * temp * (lambd[0] / tau[0])) * alpha_prony * torch.randn_like(current)
                    
                                        # Compute score using real view of data
                    current_real = torch.view_as_real(current).permute(0, 3, 1, 2)
                    with torch.no_grad():
                        score = diffuser(current_real, labels)
                    # View as complex
                    score = \
                        torch.view_as_complex(score.permute(0, 2, 3, 1).contiguous())
                    
                    # Compute gradient for measurements in un-normalized space
                    meas_grad = torch.matmul(forward_h, torch.matmul(forward, current) - y) / \
                                    (local_noise/2. + current_sigma ** 2)
    
                    
                    # Apply update of momentum in two steps
                    vT = vT + alpha / 2  * (score - meas_grad) + alpha / 2 * sigma_prony
                    
                else:
                    raise NotImplementedError('Discretization scheme not implemented')
                        
                # Store loss
                nmse_log[spacing_idx, pilot_alpha_idx, snr_idx, trailing_idx] = \
                    (torch.sum(torch.square(torch.abs(current - oracle)), dim=(-1, -2))/\
                    torch.sum(torch.square(torch.abs(oracle)), dim=(-1, -2))).cpu().numpy()
                trailing_idx = trailing_idx + 1

                
# Use average estimation error to select best number of steps
avg_nmse  = np.mean(nmse_log, axis=-1)
best_nmse = np.min(avg_nmse, axis=-1)
               
# Save results to file based on noise
save_dict = {'nmse_log': nmse_log,
             'avg_nmse': avg_nmse,
             'best_nmse': best_nmse,
             'spacing_range': spacing_range,
             'pilot_alpha_range': pilot_alpha_range,
             'snr_range': snr_range,
             'val_config': val_config,
            }

if args.discretization_scheme == 'BACOCAB':
    torch.save(save_dict, os.path.join(result_dir, 'results_lang_3rdorder_BACOCAB.pt'))
elif args.discretization_scheme == 'BAOAB':
    torch.save(save_dict, os.path.join(result_dir, 'results_lang_3rdorder_BAOAB.pt'))
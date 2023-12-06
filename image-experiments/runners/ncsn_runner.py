import numpy as np
import sys
import os

import torch
import torch.nn.functional as F
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
import torchvision.transforms as T
import lpips
from skimage.metrics import structural_similarity as ssim

from models.ncsnv2 import NCSNv2Deeper, NCSNv2, NCSNv2Deepest
from datasets import get_dataset, data_transform, inverse_data_transform
from models import general_anneal_Langevin_dynamics, general_anneal_2ndorder_Langevin_dynamics_BAOAB, general_anneal_3rdorder_Langevin_dynamics_BACOCAB
from models import get_sigmas
from models.ema import EMAHelper

__all__ = ['NCSNRunner']

def rgb2gray(rgb):

    r, g, b = rgb[0,:,:], rgb[1,:,:], rgb[2,:,:]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray

def get_model(config):
    if config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)
    
def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))

def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img)
    return img


class NCSNRunner():
    def __init__(self, args, config):
        self.args = args
        self.config = config
        args.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(args.log_sample_path, exist_ok=True)

    def sample_general(self, score, samples, init_samples, sigma_0, sigmas, num_variations = 8, deg = 'sr4'):
        # Stochastic variations: Per batch size, we have the clean, noisy measurement, MMSE and std,
        #  and the num of variations of samples
        stochastic_variations = torch.zeros((4 + num_variations) * self.config.sampling.batch_size, 
                                            self.config.data.channels, self.config.data.image_size,
                                            self.config.data.image_size)

        # Load LPIPS
        loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores

        # Create the clean samples and add to stochastic variation array
        clean = samples.view(samples.shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size)
        sample = inverse_data_transform(self.config, clean)
        
        stochastic_variations[0 : self.config.sampling.batch_size,:,:,:] = sample
        
        img_dim = self.config.data.image_size ** 2
        # Get degradation matrix H
        H = 0
        if deg[:2] == 'cs':
            ## random with set singular values ##
            compress_by = int(deg[2:])
            Vt = torch.rand(img_dim, img_dim).to(self.config.device)
            Vt, _ = torch.qr(Vt, some=False)
            U = torch.rand(img_dim // compress_by, img_dim // compress_by).to(self.config.device)
            U, _ = torch.qr(U, some=False)
            S = torch.hstack((torch.eye(img_dim // compress_by), torch.zeros(img_dim // compress_by, (compress_by-1) * img_dim // compress_by))).to(self.config.device)
            H = torch.matmul(U, torch.matmul(S, Vt))
        elif deg == 'inp':
            #Inp 
            from functions.svd_replacement import Inpainting
            if deg == 'inp_lolcat':
                loaded = np.load("inp_masks/lolcat_extra.npy")
                mask = torch.from_numpy(loaded).to(self.config.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            elif deg == 'inp_lorem':
                loaded = np.load("inp_masks/lorem3.npy")
                mask = torch.from_numpy(loaded).to(self.config.device).reshape(-1)
                missing_r = torch.nonzero(mask == 0).long().reshape(-1) * 3
            else:
                # Random inpainting
                missing_r = torch.randperm(self.config.data.image_size**2)[:int(self.config.data.image_size**2 * 0.5)].to(self.config.device).long() * 3
            missing_g = missing_r + 1
            missing_b = missing_g + 1
            missing = torch.cat([missing_r, missing_g, missing_b], dim=0)
            H_funcs = Inpainting(self.config.data.channels, self.config.data.image_size, missing, self.config.device)
        elif deg[:10] == 'sr_bicubic':
            factor = int(deg[10:])
            from functions.svd_replacement import SRConv
            def bicubic_kernel(x, a=-0.5):
                if abs(x) <= 1:
                    return (a + 2)*abs(x)**3 - (a + 3)*abs(x)**2 + 1
                elif 1 < abs(x) and abs(x) < 2:
                    return a*abs(x)**3 - 5*a*abs(x)**2 + 8*a*abs(x) - 4*a
                else:
                    return 0
            k = np.zeros((factor * 4))
            for i in range(factor * 4):
                x = (1/factor)*(i- np.floor(factor*4/2) +0.5)
                k[i] = bicubic_kernel(x)
            k = k / np.sum(k)
            kernel = torch.from_numpy(k).float().to(self.config.device)
            H_funcs = SRConv(kernel / kernel.sum(), \
                             self.config.data.channels, self.config.data.image_size, self.config.device, stride = factor)
        elif deg == 'deblur_uni':
            # Deblur uniform
            from functions.svd_replacement import Deblurring
            H_funcs = Deblurring(torch.Tensor([1/9] * 9).to(self.config.device), self.config.data.channels, self.config.data.image_size, self.config.device)

        elif deg == 'deblur_gauss':
            # Deblur gauss isotropic
            from functions.svd_replacement import Deblurring
            sigma = 3.
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel = torch.Tensor([pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2)]).to(self.config.device)
            H_funcs = Deblurring(kernel / kernel.sum(), self.config.data.channels, self.config.data.image_size, self.config.device)

        elif deg == 'deblur_aniso':
            # Deblur gauss isotropic
            from functions.svd_replacement import Deblurring2D
            sigma = 20.
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel2 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.config.device)

            sigma = 1.
            pdf = lambda x: torch.exp(torch.Tensor([-0.5 * (x/sigma)**2]))
            kernel1 = torch.Tensor([pdf(-4), pdf(-3), pdf(-2), pdf(-1), pdf(0), pdf(1), pdf(2), pdf(3), pdf(4)]).to(self.config.device)

            H_funcs = Deblurring2D(kernel1 / kernel1.sum(), kernel2 / kernel2.sum(), self.config.data.channels, self.config.data.image_size, self.config.device)
        elif deg[:2] == 'sr':
            ## downscale - super resolution ##
            blur_by = int(deg[2:])
            from functions.svd_replacement import SuperResolution
            H_funcs = SuperResolution(self.config.data.channels, self.config.data.image_size, blur_by, self.config.device)

        else:
            print("ERROR: degradation type not supported")
            quit()

        # Generate y_0
        y_0 = H_funcs.H(samples)
        y_0 = y_0 + sigma_0 * torch.randn_like(y_0)
        
        torch.save(y_0, os.path.join(self.args.image_folder, "y_0.pt"))

        pinv_y_0 = H_funcs.H_pinv(y_0).view(y_0.shape[0], self.config.data.channels, self.config.data.image_size, self.config.data.image_size)

        if deg == 'deblur_uni' or deg == 'deblur_gauss' or deg == 'deblur_aniso' or deg == 'deblur_aniso_diff': pinv_y_0 = y_0
        elif deg[:3] == 'inp': pinv_y_0 += H_funcs.H_pinv(H_funcs.H(torch.ones_like(pinv_y_0))).reshape(*pinv_y_0.shape) - 1

        sample = inverse_data_transform(self.config, pinv_y_0.view(samples.shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size)) 
        
        # Add noisy measurment to array
        stochastic_variations[1 * self.config.sampling.batch_size : 2 * self.config.sampling.batch_size,:,:,:] = sample

        clean = stochastic_variations[0 * self.config.sampling.batch_size : 1 * self.config.sampling.batch_size,:,:,:]

        # Run Langevin dynamics
        for i in range(num_variations):
            if self.args.order_lang == '1st':
                all_samples = general_anneal_Langevin_dynamics(H_funcs, y_0, init_samples, score, sigmas,
                                               self.config.sampling.n_steps_each,
                                               self.config.sampling.step_lr, verbose=True,
                                               final_only=self.config.sampling.final_only,
                                               denoise=self.config.sampling.denoise, c_begin=0, 
                                               sigma_0 = sigma_0, spacing_classes = self.args.spacing_classes)
            elif self.args.order_lang == '2nd':
                all_samples = general_anneal_2ndorder_Langevin_dynamics_BAOAB(H_funcs, y_0, init_samples, score, sigmas,
                                            self.config.sampling.n_steps_each,
                                            self.config.sampling.step_lr, verbose=True,
                                            final_only=self.config.sampling.final_only,
                                            denoise=self.config.sampling.denoise, c_begin=0,
                                            sigma_0 = sigma_0, spacing_classes = self.args.spacing_classes)
            elif self.args.order_lang == '3rd':
                all_samples = general_anneal_3rdorder_Langevin_dynamics_BACOCAB(H_funcs, y_0, init_samples, score, sigmas,
                                            self.config.sampling.n_steps_each,
                                            self.config.sampling.step_lr, verbose=True,
                                            final_only=self.config.sampling.final_only,
                                            denoise=self.config.sampling.denoise, c_begin=0, 
                                            sigma_0 = sigma_0, spacing_classes = self.args.spacing_classes)
            
            else:
                print("No method selected. Program halted.")
                sys.exit(0)

            sample = all_samples[-1].view(all_samples[-1].shape[0], self.config.data.channels,
                                      self.config.data.image_size,
                                      self.config.data.image_size).to(self.config.device)
            stochastic_variations[(self.config.sampling.batch_size) * (i+2) : (self.config.sampling.batch_size) * (i+3),:,:,:] = inverse_data_transform(self.config, sample)

        # Calculate mean and std 
        runs = stochastic_variations[(self.config.sampling.batch_size) * (2) : (self.config.sampling.batch_size) * (2+num_variations),:,:,:]
        runs = runs.view(-1, self.config.sampling.batch_size, self.config.data.channels,
                          self.config.data.image_size,
                          self.config.data.image_size)

        stochastic_variations[(self.config.sampling.batch_size) * (-2) : (self.config.sampling.batch_size) * (-1),:,:,:] = torch.mean(runs, dim=0)
        stochastic_variations[(self.config.sampling.batch_size) * (-1) : ,:,:,:] = torch.std(runs, dim=0)

        torch.save(stochastic_variations, os.path.join(self.args.image_folder, "results.pt"))

        image_grid = make_grid(stochastic_variations, self.config.sampling.batch_size)
        save_image(image_grid, os.path.join(self.args.image_folder, 'stochastic_variation.png'))
        
        # COMMENTED: To save images separated (clean. noisy measurement and estimation)
        # for bs in range(12):
        #     clean_input = stochastic_variations[0 + bs,:,:,:]
        #     noisy_input = stochastic_variations[12 + bs,:,:,:]
        #     general = stochastic_variations[24 + bs,:,:,:]
        #     save_image(clean_input, os.path.join("exp/label/", f"0000{bs}.png"w))
        #     save_image(noisy_input, os.path.join("exp/input/", f"noisy_input_{bs}.png"))
        #     save_image(general, os.path.join("exp/recon/", f"recon_{bs}_2ndorder_noisy.png"))

        # print(stochastic_variations.shape)
        for i in range(num_variations):
            general = stochastic_variations[(2+i) * self.config.sampling.batch_size : (3+i) * self.config.sampling.batch_size,:,:,:]
            mse = torch.mean((general - clean) ** 2)
            instance_mse = ((general - clean) ** 2).view(general.shape[0], -1).mean(1)
            psnr = torch.mean(10 * torch.log10(1/instance_mse))
            LPIPS = loss_fn_alex(general, clean)
            print("MSE/PSNR of the general #%d: %f, %f" % (i, mse, psnr))
            print("LPIS #%d: %f" % (i, torch.mean(LPIPS).detach().numpy()))

        mean = stochastic_variations[(2+num_variations) * self.config.sampling.batch_size : (3+num_variations) * self.config.sampling.batch_size,:,:,:]
        mse = torch.mean((mean - clean) ** 2)
        instance_mse = ((mean - clean) ** 2).view(mean.shape[0], -1).mean(1)
        psnr = torch.mean(10 * torch.log10(1/instance_mse))
        print("MSE/PSNR of the mean: %f, %f" % (mse, psnr))

        ssim_val_arr = np.zeros((self.config.sampling.batch_size))
        for bs in range(self.config.sampling.batch_size):
            ssim_val = ssim(rgb2gray(np.array(general[bs,:,:,:].detach().cpu().numpy())), 
                            rgb2gray(np.array(clean[bs ,:,:,:].detach().cpu().numpy())),
                            data_range=1)
            ssim_val_arr[bs] = ssim_val
        
        print("SSIM - mean: %f" % (np.mean(ssim_val)))

    def sample(self):
        score, states = 0, 0
        if self.config.sampling.ckpt_id is None:
            states = torch.load(os.path.join(self.args.log_path, 'checkpoint.pth'), map_location=self.config.device)
        else:
            states = torch.load(os.path.join(self.args.log_path, f'checkpoint_{self.config.sampling.ckpt_id}.pth'),
                                map_location=self.config.device)

        score = get_model(self.config)
        score = torch.nn.DataParallel(score, device_ids = [0,1,2])

        score.load_state_dict(states[0], strict=True)

        if self.config.model.ema:
            ema_helper = EMAHelper(mu=self.config.model.ema_rate)
            ema_helper.register(score)
            ema_helper.load_state_dict(states[-1])
            ema_helper.ema(score)

        sigmas_th = get_sigmas(self.config)
        sigmas = sigmas_th.cpu().numpy()

        sigma_0 = self.args.sigma_0

        dataset, test_dataset = get_dataset(self.args, self.config)

        ########## 
        # Same as DDRM
        def seed_worker(worker_id):
            worker_seed = self.args.seed % 2**32
            np.random.seed(worker_seed)
            
        g = torch.Generator()
        g.manual_seed(self.args.seed)
        dataloader = DataLoader(test_dataset, batch_size=self.config.sampling.batch_size,
                                shuffle=True, num_workers=self.config.data.num_workers,
                                worker_init_fn=seed_worker, generator=g)
        self.args.sigma_0 = 2 * self.args.sigma_0 #to account for scaling to [-1,1]
        sigma_0 = self.args.sigma_0
        ########## 

        score.eval()

        data_iter = iter(dataloader)
        samples, _ = next(data_iter)
        samples = samples.to(self.config.device)
        samples = data_transform(self.config, samples)
        init_samples = torch.rand_like(samples)
        
        self.sample_general(score, samples, init_samples, sigma_0, sigmas, num_variations=self.args.num_variations, deg=self.args.degradation)

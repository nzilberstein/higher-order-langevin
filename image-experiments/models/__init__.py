import torch
import numpy as np
import tqdm

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
                np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                                config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

def invert_diag(M):
    M_inv = torch.zeros_like(M)
    M_inv[M != 0] = 1 / M[M != 0]
    return M_inv
        
@torch.no_grad()
def general_anneal_Langevin_dynamics(H, y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, c_begin = 0, sigma_0 = 1, spacing_classes = 120):
    
    ''' 1st order Langevin with Euler-Maruyama scheme discretization. Spacing classes is the number of steps between two consecutives
     levels of noise.  '''

    ZERO = 1e-3

    # print(x_mod.shape)
    img_size = x_mod.shape[-1]
    singulars = H.singulars()

    temp = 1 #0.1 for LPIPS
    singulars[singulars < ZERO] = 0

    Sigma = torch.zeros(x_mod.shape[1]*x_mod.shape[2]*x_mod.shape[3], device=x_mod.device)
    Sigma[:singulars.shape[0]] = singulars

    S_S_t = singulars ** 2
    s0_2_I = ((sigma_0 ** 2) * torch.ones(singulars.shape[0])).to(x_mod.device)

    V_t_x = H.Vt(x_mod)
    U_t_y = H.Ut(y_0)

    img_dim = x_mod.shape[2]

    images = []

    with torch.no_grad():
        for step_idx in tqdm.tqdm(range(0, len(sigmas), spacing_classes), desc='general annealed Langevin sampling'):

            sigma = sigmas[step_idx]
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (step_idx + c_begin)
            labels = labels.long()
            step_size = step_lr * ((1 / sigmas[-1]) ** 2)

            falses = torch.zeros(V_t_x.shape[1] - singulars.shape[0], dtype=torch.bool, device=x_mod.device)
            cond_before_lite = singulars * sigma > sigma_0
            cond_after_lite = singulars * sigma < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            step_vector = torch.zeros_like(V_t_x)
            step_vector[:, :] = step_size * (sigma**2)
            step_vector[:, cond_before] = step_size * ((sigma**2) - (sigma_0 / singulars[cond_before_lite])**2)
            step_vector[:, cond_after] = step_size * (sigma**2) * (1 - (singulars[cond_after_lite] * sigma / sigma_0)**2)

            for s in range(n_steps_each):
                grad = torch.zeros_like(V_t_x)
                score = scorenet(x_mod, labels)
                score = H.Vt(score)

                diag_mat = S_S_t * (sigma ** 2) - s0_2_I
                diag_mat[cond_after_lite] = diag_mat[cond_after_lite] * (-1)

                first_vec = U_t_y - (V_t_x * Sigma)[:, :U_t_y.shape[1]]

                cond_grad = first_vec * invert_diag(diag_mat)
                
                cond_grad = Sigma[:cond_grad.shape[1]] * cond_grad
                grad = torch.zeros_like(cond_grad)
                grad = score
                grad[:, cond_before] = cond_grad[:, cond_before_lite]
                grad[:, cond_after] = cond_grad[:, cond_after_lite] + score[:, cond_after]

                noise = torch.randn_like(V_t_x)
                V_t_x = V_t_x + step_vector * grad + noise * torch.sqrt(step_vector * 2 * temp)
                x_mod = H.V(V_t_x).view(-1, 3, img_size,img_size)

                if not final_only:
                    images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images
        
@torch.no_grad()
def general_anneal_2ndorder_Langevin_dynamics_BAOAB(H, y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, c_begin = 0, sigma_0 = 1, spacing_classes = 120):
    ZERO = 1e-3

    ''' 2nd order Langevin with BAOAB scheme discretization. Spacing classes is the number of steps between two consecutives
     levels of noise.  '''

    # Hyperparameters
    gamma = 1.5 #4
    M_inv = 1 #0.8
    MM = 1 / M_inv
    temp = 1 #0.08

    img_size = x_mod.shape[-1]
    singulars = H.singulars()

    temp = 1 #0.1 for LPIPS
    singulars[singulars < ZERO] = 0

    # Get spectral decomposition of H
    Sigma = torch.zeros(x_mod.shape[1]*x_mod.shape[2]*x_mod.shape[3], device=x_mod.device)
    Sigma[:singulars.shape[0]] = singulars

    S_S_t = singulars ** 2
    s0_2_I = ((sigma_0 ** 2) * torch.ones(singulars.shape[0])).to(x_mod.device)

    V_t_x = H.Vt(x_mod)
    U_t_y = H.Ut(y_0)

    img_dim = x_mod.shape[2]

    images = []

    #Initialization of momentum variable
    p_t =  torch.randn_like(V_t_x)

    with torch.no_grad():
        for step_idx in tqdm.tqdm(range(0, len(sigmas), spacing_classes), desc='general annealed 2nd order Langevin sampling BAOAB'):

            sigma = sigmas[step_idx]
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (step_idx + c_begin)
            labels = labels.long()
            step_size = step_lr * ((1 / sigmas[-1]) ** 2)

            falses = torch.zeros(V_t_x.shape[1] - singulars.shape[0], dtype=torch.bool, device=x_mod.device)
            cond_before_lite = singulars * sigma > sigma_0
            cond_after_lite = singulars * sigma < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            step_vector = torch.zeros_like(V_t_x)
            step_vector[:, :] = step_size * (sigma**2)
            step_vector[:, cond_before] = step_size * ((sigma**2) - (sigma_0 / singulars[cond_before_lite])**2)
            step_vector[:, cond_after] = step_size * (sigma**2) * (1 - (singulars[cond_after_lite] * sigma / sigma_0)**2)
            
            grad = torch.zeros_like(V_t_x)
            score = scorenet(x_mod, labels)
            score = H.Vt(score)

            diag_mat = S_S_t * (sigma ** 2) - s0_2_I
            diag_mat[cond_after_lite] = diag_mat[cond_after_lite] * (-1)

            first_vec = U_t_y - (V_t_x * Sigma)[:, :U_t_y.shape[1]]

            cond_grad = first_vec * invert_diag(diag_mat)
            cond_grad = Sigma[:cond_grad.shape[1]] * cond_grad
            grad = torch.zeros_like(cond_grad)
            grad = score
            grad[:, cond_before] = cond_grad[:, cond_before_lite]
            grad[:, cond_after] = cond_grad[:, cond_after_lite] + score[:, cond_after]

            for s in range(n_steps_each):
                p_t = p_t + step_vector / 2 * grad

                V_t_x = V_t_x + M_inv * step_size / 2 * p_t

                noise = torch.randn_like(V_t_x)
                p_t = np.exp(- gamma * step_size) * p_t + np.sqrt( MM * temp * (1 - np.exp(-2 * gamma * step_size))) * torch.sqrt(step_vector / step_size) * noise

                x_mod = H.V(V_t_x).view(-1, 3, img_size,img_size)   

                grad = torch.zeros_like(V_t_x)
                score = scorenet(x_mod, labels)
                score = H.Vt(score)

                diag_mat = S_S_t * (sigma ** 2) - s0_2_I
                diag_mat[cond_after_lite] = diag_mat[cond_after_lite] * (-1)

                first_vec = U_t_y - (V_t_x * Sigma)[:, :U_t_y.shape[1]]

                cond_grad = first_vec * invert_diag(diag_mat)
                cond_grad = Sigma[:cond_grad.shape[1]] * cond_grad
                grad = torch.zeros_like(cond_grad)
                grad = score
                grad[:, cond_before] = cond_grad[:, cond_before_lite]
                grad[:, cond_after] = cond_grad[:, cond_after_lite] + score[:, cond_after]

                ## Update momentum variable
                V_t_x = V_t_x + M_inv * step_size / 2 * p_t
                p_t = p_t + step_vector / 2 * grad

                x_mod = H.V(V_t_x).view(-1, 3, img_size,img_size) 

                # print(x_mod)
                if not final_only:
                    images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def general_anneal_3rdorder_Langevin_dynamics_BACOCAB(H, y_0, x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, c_begin = 0, sigma_0 = 1, spacing_classes = 120):
    ZERO = 1e-3
    ''' 3rd order Langevin with BACOCAB scheme discretization. Spacing classes is the number of steps between two consecutives
     levels of noise.  '''

    # Hyperparameters
    M_inv = 0.8 #
    MM = 1 / M_inv
    temp = 1 #0.3 is best LPIPS
    tau = [1.1] #1
    lambd = [1.7] # 1.5 para spacing 25

    img_size = x_mod.shape[-1]
    singulars = H.singulars()

    singulars[singulars < ZERO] = 0


    # Get spectral decomposition of H
    Sigma = torch.zeros(x_mod.shape[1]*x_mod.shape[2]*x_mod.shape[3], device=x_mod.device)
    Sigma[:singulars.shape[0]] = singulars

    S_S_t = singulars ** 2
    s0_2_I = ((sigma_0 ** 2) * torch.ones(singulars.shape[0])).to(x_mod.device)

    V_t_x = H.Vt(x_mod)
    U_t_y = H.Ut(y_0)

    #Initialization of momentum variable
    p_t =  torch.randn_like(V_t_x)
    sigma_prony_t =  torch.randn_like(V_t_x)

    images = []

    with torch.no_grad():
        for step_idx in tqdm.tqdm(range(0, len(sigmas), spacing_classes), desc='general annealed 3rd order Langevin sampling BACOCAB'):

            sigma = sigmas[step_idx]
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * (step_idx + c_begin)
            labels = labels.long()
            step_size = step_lr * ((1 / sigmas[-1]) ** 2)

            # Gradient computation
            falses = torch.zeros(V_t_x.shape[1] - singulars.shape[0], dtype=torch.bool, device=x_mod.device)
            cond_before_lite = singulars * sigma > sigma_0
            cond_after_lite = singulars * sigma < sigma_0
            cond_before = torch.hstack((cond_before_lite, falses))
            cond_after = torch.hstack((cond_after_lite, falses))

            step_vector = torch.zeros_like(V_t_x)
            step_vector[:, :] = step_size * (sigma**2)
            step_vector[:, cond_before] = step_size * ((sigma**2) - (sigma_0 / singulars[cond_before_lite])**2)
            step_vector[:, cond_after] = step_size * (sigma**2) * (1 - (singulars[cond_after_lite] * sigma / sigma_0)**2)

            # Hyperparameters
            theta = np.exp(-step_size * lambd[0]) 
            alpha_prony = np.sqrt( (2 * (1-theta)**2) / ( step_size * lambd[0]) )

            
            grad = torch.zeros_like(V_t_x)
            score = scorenet(x_mod, labels)
            score = H.Vt(score)

            diag_mat = S_S_t * (sigma ** 2) - s0_2_I
            diag_mat[cond_after_lite] = diag_mat[cond_after_lite] * (-1)

            first_vec = U_t_y - (V_t_x * Sigma)[:, :U_t_y.shape[1]]

            cond_grad = first_vec * invert_diag(diag_mat)
            cond_grad = Sigma[:cond_grad.shape[1]] * cond_grad
            grad = torch.zeros_like(cond_grad)
            grad = score
            grad[:, cond_before] = cond_grad[:, cond_before_lite]
            grad[:, cond_after] = cond_grad[:, cond_after_lite] + score[:, cond_after]

            for s in range(n_steps_each):
                # Apply update of momentum                
                p_t = p_t + step_vector / 2 * grad

                # Update of the image
                V_t_x = V_t_x + M_inv * step_size / 2 * p_t

                #Update of the momentum
                p_t = p_t + step_size / 2 * tau[0] * sigma_prony_t

                #Update of sigma prony
                sigma_prony_t = theta * sigma_prony_t - (tau[0] / lambd[0]) * (1 - theta) * p_t + np.sqrt(MM * temp) * alpha_prony * torch.sqrt(step_vector / step_size) * torch.randn_like(V_t_x)

                #Update of the momentum
                p_t = p_t + step_size / 2 * tau[0] * sigma_prony_t

                V_t_x = V_t_x + M_inv * step_size / 2 * p_t

                x_mod = H.V(V_t_x).view(-1, 3, img_size,img_size)

                grad = torch.zeros_like(V_t_x)
                score = scorenet(x_mod, labels)
                score = H.Vt(score)

                diag_mat = S_S_t * (sigma ** 2) - s0_2_I
                diag_mat[cond_after_lite] = diag_mat[cond_after_lite] * (-1)

                first_vec = U_t_y - (V_t_x * Sigma)[:, :U_t_y.shape[1]]

                cond_grad = first_vec * invert_diag(diag_mat)
                cond_grad = Sigma[:cond_grad.shape[1]] * cond_grad
                grad = torch.zeros_like(cond_grad)
                grad = score
                grad[:, cond_before] = cond_grad[:, cond_before_lite]
                grad[:, cond_after] = cond_grad[:, cond_after_lite] + score[:, cond_after]

                ## Update momentum variable
                p_t = p_t + step_vector / 2 * grad

                if not final_only:
                    images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images
        


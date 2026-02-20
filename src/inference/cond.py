import numpy as np
import torch as th
from tqdm import tqdm

def infer_grad(cfm_model : th.nn.Module,
                samples_per_batch, total_samples, dims_of_img, 
                num_of_steps, grad_cost_func, meas_func, conditioning, conditioning_scale,
                device, refine, sample_noise, use_heavy_noise, rf_start, start_provided=False, start_point=None, **kwargs):
    """Grad-based inference algorithm"""
    
    start = 5e-3 if rf_start else 0 
    ts = th.linspace(start, 1, num_of_steps, device=device)
    dt = ts[1] - ts[0]
    
    samples_per_batch = 1
    
    samples = []
    
    for i in range(total_samples//samples_per_batch):
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        if kwargs["swag"]:
            kwargs["model"].sample()

        x = sample_noise(samples_size, dims_of_img, use_heavy_noise, device, nu=kwargs["nu"]) if not start_provided else start_point
        conditioning_per_batch = conditioning
        pbar = tqdm(ts[:-1])        
        for t in pbar:            
            x_fixed = x.clone().detach()
                           
            for _ in range(refine): ##Picard Iteration
                x = x.requires_grad_()
                v = cfm_model(t, x)
                
                scaled_grad, loss = grad_cost_func(meas_func, x, conditioning_per_batch, 
                                                is_grad_free=False, grad={"t" : t, "v" : v},
                                                **kwargs)
                scaled_grad *= th.linalg.norm(v.flatten())
                pbar.set_postfix({'distance': loss}, refresh=False)

                v = v - conditioning_scale*scaled_grad
                x = x_fixed + v*dt 
                
                x = x.detach()
        
        samples.append(x.cpu().numpy())
        
    return np.concatenate(samples)
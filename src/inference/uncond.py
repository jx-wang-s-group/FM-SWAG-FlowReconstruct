import numpy as np
import torch as th
from tqdm import tqdm
from torchdyn.core import NeuralODE
from torchdiffeq import odeint
from functools import partial

def infer(dims_of_img, total_samples, samples_per_batch,
          use_odeint, cfm_model, t_start, t_end,
          scale, device, m=None, std=None, t_steps=2, use_heavy_noise=False, 
          y = None, y0_provided = False, y0= None, all_traj=False, **kwargs):
    
    y0_ = y0.clone().detach() if y0_provided else None
    cfm_model_ = lambda t, x : cfm_model(t, x, y=y)
        
    samples_list = []

    use_heavy_noise = False
    
    if use_odeint:
        ode_solver_ = partial(odeint, func=cfm_model_, t=th.linspace(t_start, t_end, t_steps, device=device), 
                            atol=1e-5, rtol=1e-5, 
                            method=kwargs["method"] if "method" in kwargs.keys() else None,
                            options=kwargs["options"] if "options" in kwargs.keys() else None)
        ode_solver = lambda x : ode_solver_(y0=x)
    else:
        ode = NeuralODE(cfm_model_, kwargs["method"], sensitivity="adjoint", atol=1e-5, rtol=1e-5)
        ode_solver_ = partial(ode.trajectory, t_span=th.linspace(t_start, t_end, t_steps, device=device))
        ode_solver = lambda x: ode_solver_(x=x)
    
    for i in tqdm(range(total_samples//samples_per_batch)):
        
        samples_size = samples_per_batch
        if i == total_samples//samples_per_batch - 1:
            samples_size = samples_per_batch + total_samples%samples_per_batch
        samples_size = (samples_size,) + dims_of_img 
        
        with th.no_grad():
            if not use_heavy_noise and not y0_provided:
                y0 = th.randn(samples_size, device=device)
            else:
                y0 = (y0_[i*samples_size[0] : (i+1)*samples_size[0]]).clone().detach()
        
            traj = ode_solver(y0)
            
        out = traj.detach().cpu().numpy() if all_traj else traj[-1].detach().cpu().numpy() 
        
        if scale:
            assert m is not None and std is not None, "Provide output scaling for generated samples"
            out *= std
            out += m
            
        samples_list.append(out)

    if len(samples_list) == 1:
        return samples_list[0]
    else:
        if not all_traj:
            return np.concatenate(samples_list)
        else:
            return np.concatenate(samples_list, axis=1)
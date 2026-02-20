import torch as th
from torch.nn.functional import interpolate

def identity(x_hat, **kwargs):
    return x_hat

def inpainting(x_hat, **kwargs):
    mask = kwargs["mask"]
    return x_hat * mask

def inpainting2(x_hat, **kwargs):
    slice_x, slice_y = slice(kwargs["sx"],kwargs["ex"]), slice(kwargs["sy"],kwargs["ey"]),
    return x_hat[..., slice_x, slice_y]

def partial_wall_pres_forward(x_hat, **kwargs):
    det_model = kwargs["model"]
    x_pred = det_model(x_hat)
    if "mask" not in kwargs.keys(): 
        if "full" in kwargs.keys():
            return x_pred
        slice_c = slice(kwargs["sc"], kwargs["ec"])
        slice_x, slice_y = slice(kwargs["sx"],kwargs["ex"]), slice(kwargs["sy"],kwargs["ey"])
        return x_pred[..., slice_c, slice_x, slice_y]
    else :
        mask = kwargs["mask"]
        return x_pred * mask
    
def coarse_wall_pres_forward(x_hat, **kwargs):
    det_model = kwargs["model"]
    size = kwargs["size"]
    mode = kwargs["mode"] if "mode" in kwargs.keys() else "nearest"
    x_pred = det_model(x_hat)
    return interpolate(x_pred, size=size, mode=mode)

MEAS_MODELS = {"inpainting": inpainting2, "partial_wall_pres_forward": partial_wall_pres_forward,
               "coarse_wall_pres_forward": coarse_wall_pres_forward, "inpainting_2": inpainting, "identity": identity}

#------------------------------------------------------------------------------------------#

def cost_func(meas_func, x_hat, measurement, **kwargs):
    pred_measurement = meas_func(x_hat, **kwargs)
    diff = pred_measurement - measurement
    return (diff**2).mean()

def grad_cost_func(meas_func, x, measurement, cost_func=cost_func, **kwargs):
    
    if kwargs["is_grad_free"]:
        
        if kwargs["use_fd"]:
            assert "x_prev" in kwargs["grad_free"].keys() and "dt" in kwargs["grad_free"].keys(), "previous step is not cached!"
            x_prev, dt, t = kwargs["grad_free"]["x_prev"], kwargs["grad_free"]["dt"],  kwargs["grad_free"]["t"]
            v_fd = (x - x_prev)/dt
            x_hat = x + (1 - t)*v_fd
        else:
            a_t, b_t, x_gauss = kwargs["grad_free"]["a_t"], kwargs["grad_free"]["b_t"], kwargs["grad_free"]["x_gauss"]
            x_hat = 1/a_t * (x - b_t * x_gauss)
    else:
        t, v = kwargs["grad"]["t"], kwargs["grad"]["v"]
        x_hat = x + (1 - t)*v
            
    diff_norm =  cost_func(meas_func, x_hat, measurement, **kwargs) if "cost_func" not in kwargs.keys() else kwargs["cost_func"](meas_func, x_hat, measurement, **kwargs)
    
    grad = th.autograd.grad(diff_norm, x)[0]
    unit_grad = grad / th.linalg.norm(grad)
    return unit_grad, diff_norm.item()
    
#------------------------------------------------------------------------------------------#    

def sample_noise(samples_size, dims_of_img, use_heavy_noise, device, **kwargs):
    return th.randn(samples_size, device=device)
import numpy as np
from torch.utils.data import DataLoader

def get_loaders_vf_fm(vf_paths, batch_size, dataset_, jump=None, x_spatial_cutoff=None, z_spatial_cutoff=None, time_cutoff=None):
    
    def norm(d, m, s):
        return (d-m)/s

    def process_data(data, x_spatial_cutoff, z_spatial_cutoff):
        if data.ndim == 4:
            data = data[:, :, :x_spatial_cutoff]
            data = data[..., :z_spatial_cutoff]
            m, s = np.mean(data, axis=(0,2,3), keepdims=True), np.std(data, axis=(0,2,3), keepdims=True)
        elif data.ndim == 5:
            data = data[:, :, :, :x_spatial_cutoff]
            data = data[..., :z_spatial_cutoff]
            m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)
        else:
            raise ValueError("Data has to be either 4D or 5D")
        return data, m, s

    data = []

    # Data processing is specific to how we save the data internally, modify this if needed.
    
    if (len(vf_paths) == 3 and type(vf_paths[0]) == str):
        for path in vf_paths:
            d = np.load(path)
            data.append(d)
        data = np.concatenate(data, axis=1)
        data, m, s = process_data(data, x_spatial_cutoff, z_spatial_cutoff)
            
    elif (len(vf_paths) == 1 and type(vf_paths[0][0]) == str):
        vf_paths = vf_paths[0]
        for path in vf_paths:
            d = np.load(path)
            if d.ndim == 3:
                d = d[:, None]
            data.append(d)
        data = np.concatenate(data, axis=1)
        data, m, s = process_data(data, x_spatial_cutoff, z_spatial_cutoff)

    else:
        for uvw_path in vf_paths:
            uvw_data = []    
            for path in uvw_path:
                d = np.load(path)
                uvw_data.append(d)   
            data.append(uvw_data)
        data = [np.concatenate(uvw, axis=1) for uvw in data]
        data = np.stack(data, axis=1)
        data, m, s = process_data(data, x_spatial_cutoff, z_spatial_cutoff)
    
    data = norm(data, m, s)

    if time_cutoff is not None:
        train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump]), batch_size=batch_size, shuffle=True)
    else:
        train_dataloader = DataLoader(dataset_(data[::jump]), batch_size=batch_size, shuffle=True)
      
    return train_dataloader

def get_loaders_wmvf_patch(wm_paths, vf_paths, batch_size, time_cutoff, patch_dims, dataset_, x_spatial_cutoff=None, jump=None, scale_inputs=False):
    
    def norm(d, m, s):
        if not scale_inputs:
            return (d-m)/s
        else: # only scale the outputs
            d[:, 0] = (d[:, 0] - m[:, 0])/s[:, 0]
            return d

    # Data processing is specific to how we save the data internally, modify this if needed.

    wm_data = []
    for pxy_path in wm_paths:
        pxy_data = []    
        for path in pxy_path:
            d = np.load(path)
            pxy_data.append(d)   
        wm_data.append(pxy_data)
    if len(wm_data) == 1:    
        wm_data = np.concat(wm_data[0], axis=1)[:, None]
    else:
        wm_data = [np.concat(wm, axis=1) for wm in wm_data]
        wm_data = np.stack(wm_data, axis=1)
    
    vf_data = []  
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        vf_data.append(uvw_data)
    if len(vf_data) == 1:
        vf_data = np.concat(vf_data[0], axis=1)[:, None]
    else:
        vf_data = [np.concat(vf, axis=1) for vf in vf_data]
        vf_data = np.stack(vf_data, axis=1)
    
    data = np.concatenate([wm_data, vf_data], axis=1)
    m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)

    # scaling v and w velocity components with u velocity component (Guastoni et al. 2021)
    
    if scale_inputs:
        for i in range(1,3):
            data[:, 1, i] = data[:, 1, i] * s[:, 1, 0]/s[:, 1, i]
            s[:, 1, i] = s[:, 1, 0]
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], patch_dims, x_spatial_cutoff), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[time_cutoff::jump], patch_dims, x_spatial_cutoff), batch_size=batch_size, shuffle=False)

    return train_dataloader, test_dataloader

def get_loaders_wmvf_baseline(wm_paths, vf_paths, batch_size, time_cutoff, dataset_, jump=None, scale_inputs=False, wm_vf=True):
    
    def norm(d, m, s):
        if not scale_inputs:
            return (d-m)/s
        else: # only scale the outputs
            d[:, 0] = (d[:, 0] - m[:, 0])/s[:, 0]
            return d
        
    # Data processing is specific to how we save the data internally, modify this if needed.

    wm_data = []
    for pxy_path in wm_paths:
        pxy_data = []    
        for path in pxy_path:
            d = np.load(path)
            pxy_data.append(d)   
        wm_data.append(pxy_data)
    if len(wm_data) == 1:    
        wm_data = np.concat(wm_data[0], axis=1)[:, None]
    else:
        wm_data = [np.concat(wm, axis=1) for wm in wm_data]
        wm_data = np.stack(wm_data, axis=1)
    
    vf_data = []  
    for uvw_path in vf_paths:
        uvw_data = []    
        for path in uvw_path:
            d = np.load(path)
            uvw_data.append(d)   
        vf_data.append(uvw_data)
    if len(vf_data) == 1:
        vf_data = np.concat(vf_data[0], axis=1)[:, None]
    else:
        vf_data = [np.concat(vf, axis=1) for vf in vf_data]
        vf_data = np.stack(vf_data, axis=1)
    
    data = np.concatenate([wm_data, vf_data], axis=1)
    m, s = np.mean(data, axis=(0,3,4), keepdims=True), np.std(data, axis=(0,3,4), keepdims=True)

    # scaling v and w velocity components with u velocity component (Guastoni et al. 2021)
    
    if scale_inputs:
        for i in range(1,3):
            data[:, 1, i] = data[:, 1, i] * s[:, 1, 0]/s[:, 1, i]
            s[:, 1, i] = s[:, 1, 0]
    data = norm(data, m, s)

    train_dataloader = DataLoader(dataset_(data[:time_cutoff:jump], wm_vf), batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset_(data[time_cutoff::jump], wm_vf), batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader
import numpy as np
from torch.utils.data import Dataset

class VF_FM(Dataset):
    def __init__(self, data, all_vel=True) -> None:
        super().__init__()
        self.all_vel = all_vel
        self._preprocess(data, 'data')
        
        if self.data.ndim == 4:
            self.shape = self.data.shape[1:]
            self.one_yp = True
        elif self.data.ndim == 5:
            self.shape = self.data.shape[2:]
            self.num_yp = self.data.shape[1]
            self.one_yp = False
        else:
            raise ValueError("Check the members of the dataset!")
                
    def _preprocess(self, data, name):
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):
        if self.one_yp:  
            return self.data.shape[0]
        else:
            return self.data.shape[0]*self.data.shape[1]
    
    def __getitem__(self, index):
        if self.one_yp:
            return  np.empty(self.shape, dtype=np.float32), self.data[index]
        else:
            yp_ind = index % self.num_yp
            batch = index // self.num_yp
            return np.empty(self.shape, dtype=np.float32), self.data[batch, yp_ind], yp_ind
        
class WMVF_P(Dataset):
    def __init__(self, data, patch_dims,
                 cutoff=None) -> None:
        super().__init__()
        self._preprocess(data, cutoff,'data')
        self.patch_dims = patch_dims
        self.space_ress = self.data.shape[-2:]
        
    def _preprocess(self, data, cutoff, name):
        data = (data[:, :, : ,:cutoff])
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]
    
    def __getitem__(self, index):
        start_indices = [np.random.randint(0,space_res-patch_dim) if space_res > patch_dim else 0 for space_res, patch_dim in zip(self.space_ress, self.patch_dims)]
        patch_x, patch_y = (slice(start_index, start_index+patch_dim) for start_index, patch_dim in zip(start_indices,self.patch_dims))
        return self.data[index, 1, ..., patch_x, patch_y], self.data[index, 0, ..., patch_x, patch_y]

class WMVF_baseline(Dataset):
    def __init__(self, data, wm_vf=True) -> None:
        super().__init__()
        self._preprocess(data, 'data')
        self.wm_vf = wm_vf
        
    def _preprocess(self, data, name):
        setattr(self, name, (data).astype(np.float32))
               
    def __len__(self):  
        return self.data.shape[0]
    
    def __getitem__(self, index):
        if self.wm_vf:
            return self.data[index, 0], self.data[index, 1] 
        else:
            return self.data[index, 1], self.data[index, 0]

DATASETS = {"VF_FM":VF_FM, "WMVF_P":WMVF_P, "WMVF_baseline":WMVF_baseline}

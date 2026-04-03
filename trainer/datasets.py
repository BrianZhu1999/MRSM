import numpy as np
from torch.utils.data import Dataset
from einops import rearrange
import torch.nn.functional as F


def get_dataset(config, data, train=True):
    num_samples = config.num_samples_train if train else config.num_samples_val

    total_samples = len(data)
    ind_train = int(config.train_split * total_samples)


    
    if '1dtube' in config.data.lower():
        #frequencies = np.arange(150, 400 + 10, 10) 
        #runs_per_freq = 3
        frequencies = np.arange(800, 1200 + 10, 10) 
        runs_per_freq = 5
        full_labels = np.repeat(frequencies, runs_per_freq)
        
        if len(full_labels) != total_samples:
            print(f"Warning: Generated labels ({len(full_labels)}) != Data ({total_samples}). Truncating to match.")
            full_labels = full_labels[:total_samples]
        if train:
            data_slice = data[:ind_train]
            labels_slice = full_labels[:ind_train]
        else:
            data_slice = data[ind_train:]
            labels_slice = full_labels[ind_train:]
            
        is_condition = False if config.num_conditions == 0 else True
        #return Dataset1DTube(
        return  Dataset1DTube_Channel_Wise_Normalization(
            data_slice, 
            labels_slice, 
            num_samples, 
            config.num_frames, 
            config.num_interval, 
            is_scalar=config.is_scalar, 
            is_condition=is_condition
        )


    
    if len(data) >= 4:
        ind_train = int(config.train_split*len(data))
        data = data[:ind_train] if train else data[ind_train:]
    else:
        ind_train = int(config.train_split*len(data[0]))
        data = data[:, :ind_train] if train else data[:, ind_train:]
    is_condition = False if config.num_conditions == 0 else True
    if 'kse' in config.data.lower():
        return DatasetKSEVideo(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif '1dtube' in config.data.lower():
        #return Dataset1DTube(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
        return Dataset1DTube_Channel_Wise_Normalization(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'exp_p' in config.data.lower():
        return Dataset1DTube_p(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'line_array' in config.data.lower():
        return Datasetline_array(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif '1dsound' in config.data.lower():
        return Dataset1dsound(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'kol' in config.data.lower():
        return DatasetKolmogorov(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif '2dsound' in config.data.lower():
        #return Dataset2DSound(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
        #return DatasetERA5(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
        return Dataset2DSound_P_Vx_Vy(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif '2dgausspulse' in config.data.lower():
        return Dataset2DGaussPulse(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif '2dsound_small_data1' in config.data.lower():
        return Dataset2DSound_smalldata1(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'compns'  in config.data.lower():
        return DatasetCompNS(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'era5' in config.data.lower():
        return DatasetERA5(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'exp_48k' in config.data.lower():
        return DatasetExp_48k(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar)
    elif 'cylinder' in config.data.lower():
        return DatasetCylinder(data, num_samples, config.num_frames, config.num_interval, is_scalar=config.is_scalar, is_condition=is_condition)
    else:
        raise NotImplementedError('Unexpected type of data!')


class Datasetline_array(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*c (1500, 2000, 32, 4)"""
        self.length = length
        
        # general setting
        self.num_b = data.shape[0]
        self.num_t = data.shape[1]
        self.num_h = data.shape[2]  # Space: 32
        self.num_c = data.shape[3]  # Channel: 4
        
        self.num_frames = num_frames
        self.num_interval = num_interval

        self.mean = np.mean(data)
        self.std = np.std(data)
        
        if is_scalar:
            self.data = (data - self.mean) / self.std
        else:
            self.data = data
            
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        max_time = self.num_t - self.num_frames * self.num_interval
        i_t = np.random.choice(max_time, 1)[0]
        
        x = self.data[i_b, i_t : i_t + self.num_frames * self.num_interval : self.num_interval, :]
        
        x = x.transpose(0, 2, 1)
        
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        
        return x, frame_indices, obs_mask, latent_mask

    def __len__(self):
        return self.length



class Dataset1DTube_p(Dataset):

    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        self.length = length
        

        self.num_b = data.shape[0]
        self.num_t = data.shape[1]
        self.num_h = data.shape[2]
        self.num_c = data.shape[3]
        
        self.num_frames = num_frames 
        self.num_interval = num_interval 
        
        assert self.num_c == 1, "Dataset1DTube_p requires single channel (Pressure)."

        self.mean = np.mean(data)
        self.std = np.std(data)
        self.std = np.maximum(self.std, 1e-6)

        if is_scalar:
            self.data = (data - self.mean) / self.std
            print(f"Global Normalization: Mean={self.mean:.4e}, Std={self.std:.4e}")
        else:
            self.data = data
            print("No Normalization applied (is_scalar=False).")
            
        self.data = self.data.astype(dtype)
        
        print(f"1D Pressure Dataset initialized. Data shape: {self.data.shape}")

    def __getitem__(self, item):

        i_b = np.random.choice(self.num_b, 1)[0]
        
    
        max_start_time = self.num_t - (self.num_frames - 1) * self.num_interval
        i_t = np.random.choice(max_start_time, 1)[0]
        
        indices = i_t + np.arange(self.num_frames) * self.num_interval
        x = self.data[i_b, indices] # Shape: [Frames, Space(H), Channel(C)]
        
        x = x.transpose(0, 2, 1) 
        
        shift_x = np.random.choice(self.num_h, 1)[0]
        x = np.roll(x, shift_x, axis=2)
        
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask 

    def __len__(self):
        return self.length

    def get_data_scalar(self):
        return self.mean, self.std




class DataScalar:
    def __init__(self, mean, std):
        import torch
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()
        
        if self.mean.ndim == 4:
            self.mean = self.mean.permute(0, 1, 3, 2) 
            self.std = self.std.permute(0, 1, 3, 2)
            
        elif self.mean.ndim == 5:
            self.mean = self.mean.permute(0, 1, 4, 2, 3)
            self.std = self.std.permute(0, 1, 4, 2, 3)
        
    def __call__(self, x):
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return x * self.std + self.mean


class Dataset2DSound_P_Vx_Vy(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        self.length = length
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-3]
        self.num_w = data.shape[-2]
        self.num_c = data.shape[-1]
        
        self.num_frames = num_frames
        self.num_interval = num_interval
        
        _, self.y = np.meshgrid(
            np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
            np.linspace(0, 2 * np.pi, self.num_h, endpoint=False),
            indexing='ij'
        )

        if is_scalar:
            self.mean = np.mean(data, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.std(data, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.maximum(self.std, 1e-6)

            
            if self.num_c >= 3:
                p_std = self.std[0,0,0,0,0]
                vx_std = self.std[0,0,0,0,1]
                vy_std = self.std[0,0,0,0,2] 
                
                print(f"  Ch0 (P)  Std: {p_std:.4e}")
                print(f"  Ch1 (Vx) Std: {vx_std:.4e}")
                print(f"  Ch2 (Vy) Std: {vy_std:.4e}")
                
                if abs(vx_std - vy_std) > 1e-9:
                    print(">>> Note: Vx and Vy have different stats (Independent Normalization).")
            print("-" * 30)

            self.data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros((1, 1, 1, 1, self.num_c))
            self.std = np.ones((1, 1, 1, 1, self.num_c))
            self.data = data
            
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        max_time_idx = self.num_t - (self.num_frames - 1) * self.num_interval - 1
        i_t = np.random.choice(max_time_idx, 1)[0]
        i_w = np.random.choice(self.num_w - self.num_w + 1, 1)[0]
        i_h = np.random.choice(self.num_h - self.num_h + 1, 1)[0]
        time_indices = i_t + np.arange(self.num_frames) * self.num_interval
        
        x = self.data[i_b, time_indices, i_h:i_h+self.num_h, i_w:i_w+self.num_w, :]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        
        return x, frame_indices, obs_mask, latent_mask

    def __len__(self):
        return self.length

    def get_data_scalar(self):
        return DataScalar(self.mean, self.std)






def get_dataset_inverse(config, data, train=True):
    num_samples = config.num_samples_train if train else config.num_samples_val
    if len(data) >= 4:
        ind_train = int(config.train_split*len(data))
        data = data[:ind_train] if train else data[ind_train:]
    else:
        ind_train = int(config.train_split*len(data[0]))
        data = data[:, :ind_train] if train else data[:, ind_train:]
    is_condition = False if config.num_conditions == 0 else True
    if 'kse' in config.data.lower():
        return DatasetKSEInverse(data, num_samples, config.num_frames, degen_type=config.degen_type, scale=config.scale, is_scalar=config.is_scalar, is_condition=is_condition)
    elif 'kol' in config.data.lower():
        return DatasetKolInverse(data, num_samples, config.num_frames, degen_type=config.degen_type, scale=config.scale, is_scalar=config.is_scalar, is_condition=is_condition)
    else:
        raise NotImplementedError('Unexpected type of data!')


class DatasetKolmogorov(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for Kolmogorov flow
        u0 = 1.0
        rey = np.linspace(100, 1050, 20, endpoint=True)
        sigma = np.linspace(2, 8, 7, endpoint=True)
        r, s = np.meshgrid(rey, sigma, indexing='ij')
        self.rey = r.reshape(-1)
        self.num_rey = len(self.rey)

        self.vis = u0 / self.rey
        self.f = [1. * np.sin(k * self.y) for k in s.reshape(-1)]

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r = self.vis[i_b%self.num_rey]*100*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            f = self.f[i_b%self.num_rey][np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, f, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H W
        else:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetCompNS(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: list/tuple [dataset1, dataset2, ...], each of the dataset is of shape b*t*h*h*c"""
        data = data
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        mean, std = [], []
        for i in range(self.num_c):
            mean.append(data[..., i].mean())
            std.append(data[..., i].std())
        self.mean = np.array(mean)
        self.std = np.array(std)
        print(f'Statistics of Data (mean, std): {self.mean}, {self.std}')
        if is_scalar:
            self.data = (data-self.mean[np.newaxis, np.newaxis, np.newaxis, np.newaxis])/self.std[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetERA5(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length

class DatasetExp_48k(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length


class DatasetKSE(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for Kolmogorov flow

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length
    

class DatasetKSEVideo(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.y = np.transpose(np.conj(np.arange(1, self.num_h+1))) / self.num_h * 2*np.pi
        # the following is for Kolmogorov flow
        self.is_condition = is_condition

        vis_min, vis_max, n_vis = 1, 5, 20
        self.vis = vis_min + (vis_max-vis_min) * np.arange(0, n_vis+1)/n_vis
        self.vis_scale = (self.vis-(vis_min+vis_max)/2)/(vis_max-vis_min)
        self.num_vis = len(self.vis)

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval]
        x = x.transpose(0, 2, 1)
        shift_x = np.random.choice(self.num_h, 1)[0]   # data aug.
        x = np.roll(x, shift_x, axis=2)     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r = self.vis_scale[i_b%self.num_vis]*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H
        else:
            return x, frame_indices, obs_mask, latent_mask 

    def __len__(self):
        return self.length



class Dataset1dsound(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.y = np.transpose(np.conj(np.arange(1, self.num_h+1))) / self.num_h * 2*np.pi
        self.is_condition = False

        fre_min, fre_max, n_fre = 100, 300, 5
        self.fre = fre_min + (fre_max-fre_min) * np.arange(0, n_fre+1)/n_fre
        self.fre_scale = (self.fre-(fre_min+fre_max)/2)/(fre_max-fre_min)
        self.num_fre = len(self.fre)

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval]
        x = x.transpose(0, 2, 1)
        shift_x = np.random.choice(self.num_h, 1)[0]   # data aug.
        x = np.roll(x, shift_x, axis=2)     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r = self.fre_scale[i_b%self.num_fre]*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H
        else:
            return x, frame_indices, obs_mask, latent_mask 

    def __len__(self):
        return self.length


class Dataset1DTube(Dataset):
    def __init__(self, data, labels, length, num_frames, num_interval, 
                 is_scalar=True, dtype='float32', is_condition=True):
        
        super().__init__()
        self.length = length
        self.num_b = data.shape[0]
        self.num_t = data.shape[1]
        self.num_h = data.shape[2]
        self.num_c = data.shape[3]

        self.num_frames = num_frames   
        self.num_interval = num_interval 
        assert len(data) == len(labels), \
            f"Data ({len(data)}) and Labels ({len(labels)}) length mismatch!"

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data - self.mean) / self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)
        self.is_condition = is_condition
        self.labels = labels

        if self.is_condition:
            self.fre_min = 150.0
            self.fre_max = 400.0
            self.labels_norm = (self.labels - (self.fre_min + self.fre_max) / 2) / (self.fre_max - self.fre_min)
        
        print(f"1D Dataset initialized. Mode: {'Train' if self.length > 1000 else 'Val/Test'}") # 简单判断打印
        print(f"  Data shape: {self.data.shape}")
        print(f"  Labels shape: {self.labels.shape}")

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        
        max_start_time = self.num_t - (self.num_frames - 1) * self.num_interval - 1
        i_t = np.random.choice(max_start_time, 1)[0]
        
        indices = i_t + np.arange(self.num_frames) * self.num_interval
        x = self.data[i_b, indices] 
        
        x = x.transpose(0, 2, 1) 
        shift_x = np.random.choice(self.num_h, 1)[0]
        x = np.roll(x, shift_x, axis=2)
        
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        
        if self.is_condition:
            current_fre_scale = self.labels_norm[i_b]
            
            r = np.ones((self.num_frames, 1, self.num_h)) * current_fre_scale
            x_with_cond = np.concatenate([x, r], axis=1)
            
            return x_with_cond, frame_indices, obs_mask, latent_mask
        else:
            return x, frame_indices, obs_mask, latent_mask

    def __len__(self):
        return self.length

class Dataset1DTube_Channel_Wise_Normalization(Dataset):
    def __init__(self, data, labels, length, num_frames, num_interval, 
                 is_scalar=True, dtype='float32', is_condition=True):
        
        super().__init__()
        self.length = length
        # 假设 data shape: [Batch, Time, Space, Channel]
        self.num_b = data.shape[0]
        self.num_t = data.shape[1]
        self.num_h = data.shape[2]
        self.num_c = data.shape[3] 

        self.num_frames = num_frames    
        self.num_interval = num_interval 
        assert len(data) == len(labels), \
            f"Data ({len(data)}) and Labels ({len(labels)}) length mismatch!"

        
        if is_scalar:
            self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True) # shape: (1, 1, 1, C)
            self.std = np.std(data, axis=(0, 1, 2), keepdims=True)   # shape: (1, 1, 1, C)
            self.std = np.maximum(self.std, 1e-6)

            print(f"Channel-wise Stats:")
            for c in range(self.num_c):
                print(f"  Ch{c}: Mean={self.mean[0,0,0,c]:.4e}, Std={self.std[0,0,0,c]:.4e}")

            self.data = (data - self.mean) / self.std
        else:
            self.mean = np.zeros((1, 1, 1, self.num_c))
            self.std = np.ones((1, 1, 1, self.num_c))
            self.data = data


        print("-" * 30)
        print("DEBUG CHECK: Data Statistics")
        print(f"Mean shape: {self.mean.shape}")
        print(f"Std shape:  {self.std.shape}")

        p_std = self.std[0,0,0,0]
        v_std = self.std[0,0,0,1]

        print(f"Channel 0 (Pressure) Std: {p_std:.6f}")
        print(f"Channel 1 (Velocity) Std: {v_std:.6f}")
        
        self.data = self.data.astype(dtype)
        self.is_condition = is_condition
        self.labels = labels 

        if self.is_condition:
            self.fre_min = 150.0
            self.fre_max = 400.0
            self.labels_norm = (self.labels - (self.fre_min + self.fre_max) / 2) / (self.fre_max - self.fre_min)
        
        print(f"1D Dataset initialized. Mode: {'Train' if self.length > 1000 else 'Val/Test'}") 
        print(f"  Data shape: {self.data.shape}")

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        
        max_start_time = self.num_t - (self.num_frames - 1) * self.num_interval - 1
        i_t = np.random.choice(max_start_time, 1)[0]
        
        indices = i_t + np.arange(self.num_frames) * self.num_interval
        x = self.data[i_b, indices] # Shape: [Frames, Space, Channel]
    
        x = x.transpose(0, 2, 1) 
        
        shift_x = np.random.choice(self.num_h, 1)[0]
        x = np.roll(x, shift_x, axis=2)
        
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        
        if self.is_condition:
            current_fre_scale = self.labels_norm[i_b]
            r = np.ones((self.num_frames, 1, self.num_h)) * current_fre_scale
            x_with_cond = np.concatenate([x, r], axis=1)
            return x_with_cond, frame_indices, obs_mask, latent_mask
        else:
            return x, frame_indices, obs_mask, latent_mask

    def __len__(self):
        return self.length
    def get_data_scalar(self):
        return DataScalar(self.mean, self.std)


class DatasetCylinder(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: list/tuple [dataset1, dataset2, ...], each of the dataset is of shape b*t*h*h*c"""
        data = data
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        # the following is for cylinder flow
        self.vis = np.arange(0.003, 0., -0.0001).astype(dtype)
        self.u0 = 0.1
        # self.f = [np.zeros_like(y) for _ in range(len(self.vis))]
        self.rey = self.u0 / self.vis
        self.num_rey = len(self.rey)
        # the following is for Kolmogorov flow

        self.mean = np.mean(rearrange(data, 'b t h w c -> (b t h w) c'), axis=0)
        self.std = np.std(rearrange(data, 'b t h w c -> (b t h w) c'), axis=0)
        if is_scalar:
            self.data = (data-self.mean[np.newaxis, np.newaxis, np.newaxis, np.newaxis])/self.std[np.newaxis, np.newaxis, np.newaxis, np.newaxis]
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        if self.is_condition:
            r =  self.vis[i_b%self.num_rey]*1000*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, r], axis=1), frame_indices, obs_mask, latent_mask       # B T C H W
        else:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W

    def __len__(self):
        return self.length


class DatasetKSEInverse(Dataset):
    def __init__(self, data, length, num_frames=1, degen_type='fi', scale=64, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.y = np.transpose(np.conj(np.arange(1, self.num_h+1))) / self.num_h * 2*np.pi
        # the following is for Kolmogorov flow
        self.degen_type = degen_type
        self.scale = scale
        self.is_condition = is_condition
        self.num_frames = num_frames

        vis_min, vis_max, n_vis = 1, 5, 20
        self.vis = vis_min + (vis_max-vis_min) * np.arange(0, n_vis+1)/n_vis
        self.vis_scale = (self.vis-(vis_min+vis_max)/2)/(vis_max-vis_min)
        self.num_vis = len(self.vis)

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames]
        x = rearrange(x, 't h c -> (t c) h')
        shift_x = np.random.choice(self.num_h, 1)[0]   # data aug.
        x = np.roll(x, shift_x, axis=1)     # data aug.
        if 'fi' in self.degen_type:
            x_lr = x[:, ::self.scale]
        elif 'spec_real' in self.degen_type:
            x_lr = np.fft.fftshift(np.fft.fft(x[:, ::self.scale], axis=-1).real, axes=-1)
        r = self.vis_scale[i_b%self.num_vis]*np.ones_like(self.y)[np.newaxis]
        return x_lr, x, r, self.y[np.newaxis]       # B C H

    def __len__(self):
        return self.length


class DatasetKolInverse(Dataset):
    def __init__(self, data, length, num_frames=1, degen_type='fi', scale=8, is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # general setting
        self.num_b = len(data)
        self.num_t = len(data[0])
        self.num_w = data.shape[-3]
        self.num_h = data.shape[-2]
        self.num_c = data.shape[-1]
        self.is_condition = is_condition
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')
        x, y = [np.linspace(0, 2*np.pi, self.num_h, dtype=dtype),
                np.linspace(0, 2*np.pi, self.num_w, dtype=dtype)]
        self.X, self.Y = np.meshgrid(x, y)
        self.grid = np.stack([self.X, self.Y], axis=0)
        self.degen_type = degen_type
        self.scale = scale
        self.num_frames = num_frames
        # the following is for Kolmogorov flow
        u0 = 1.0
        rey = np.linspace(100, 1050, 20, endpoint=True)
        sigma = np.linspace(2, 8, 7, endpoint=True)
        r, s = np.meshgrid(rey, sigma, indexing='ij')
        self.rey = r.reshape(-1)
        self.num_rey = len(self.rey)

        self.vis = u0 / self.rey
        self.f = [1. * np.sin(k * self.y) for k in s.reshape(-1)]

        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data-self.mean)/self.std
        else:
            self.data = data
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]
        x = self.data[i_b, i_t:i_t+self.num_frames]
        x = rearrange(x, 't h w c -> (t c) h w')
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = np.roll(x, (shift_x, shift_y), axis=(1, 2))     # data aug.
        if 'fi' in self.degen_type:
            x_lr = x[:, ::self.scale, ::self.scale]
        elif 'spec_real' in self.degen_type:
            x_lr = np.real(np.fft.fft(x[:, ::self.scale, ::self.scale], axis=(-2, -1)))
        if self.is_condition:
            r = self.vis[i_b%self.num_rey]*100*np.ones_like(self.y)[np.newaxis]
            f = self.f[i_b%self.num_rey][np.newaxis]
            return x_lr, x, np.concatenate([f, r], axis=0), self.grid      # B T C H W
        else:
            return x_lr, x, self.grid       # B T C H W

    def __len__(self):
        return self.length

class Dataset2DSound(Dataset):
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32', is_condition=False):
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        self.num_b = len(data)          
        self.num_t = len(data[0])      
        self.num_w = data.shape[-3]     
        self.num_h = data.shape[-2]     
        self.num_c = data.shape[-1]     
        self.num_frames = num_frames    
        self.num_interval = num_interval        
        self.is_condition = False        
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        self.mean = np.mean(data)                                         
        self.std = np.std(data)                                          
        if is_scalar:
            self.data = (data-self.mean)/self.std                          
        else:
            self.data = data
        self.data = self.data.astype(dtype)                               

    def __getitem__(self, item):
        i_b = np.random.choice(self.num_b, 1)[0]
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]

        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        x = x.transpose(0, 3, 1, 2)

        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames) 
        if self.is_condition:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W
        else:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W
    def __len__(self):
        return self.length

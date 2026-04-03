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

        # 简洁复刻：直接计算全局均值方差 (Global Normalization)
        self.mean = np.mean(data)
        self.std = np.std(data)
        
        if is_scalar:
            self.data = (data - self.mean) / self.std
        else:
            self.data = data
            
        self.data = self.data.astype(dtype)

    def __getitem__(self, item):
        # 随机采样 Batch 和 Time
        i_b = np.random.choice(self.num_b, 1)[0]
        # 预留足够的长度给 num_frames * num_interval
        max_time = self.num_t - self.num_frames * self.num_interval
        i_t = np.random.choice(max_time, 1)[0]
        
        # 核心切片逻辑 (Slicing)，替代 np.arange
        # Shape: [Frames, Space, Channel]
        x = self.data[i_b, i_t : i_t + self.num_frames * self.num_interval : self.num_interval, :]
        
        # Transpose: [Frames, Space, Channel] -> [Frames, Channel, Space]
        x = x.transpose(0, 2, 1)
        
        # Masks (适配 1D 空间数据 shape)
        latent_mask = np.ones([self.num_frames, 1, 1]).astype(bool)
        obs_mask = np.zeros([self.num_frames, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)
        
        return x, frame_indices, obs_mask, latent_mask

    def __len__(self):
        return self.length



class Dataset1DTube_p(Dataset):
    """
    基于 DatasetKSE 简洁风格的 1D 声压数据集类。
    实现了统一归一化 (Global Normalization)。
    
    预期 data shape: [Batch, Time, Space, 1]
    """
    # 移除 labels 参数
    def __init__(self, data, length, num_frames, num_interval, is_scalar=True, dtype='float32'):
        super().__init__()
        self.length = length
        
        # --- 1. 通用设置 ---
        # 假设 data shape: [Batch, Time, Space, Channel]
        self.num_b = data.shape[0]
        self.num_t = data.shape[1]
        self.num_h = data.shape[2] # 空间维度 H=64
        self.num_c = data.shape[3] # 通道维度 C=1
        
        self.num_frames = num_frames 
        self.num_interval = num_interval 
        
        assert self.num_c == 1, "Dataset1DTube_p requires single channel (Pressure)."
        
        # --- 2. 统一归一化 (Global Normalization) ---
        # 对整个数据集求均值和标准差
        self.mean = np.mean(data)
        self.std = np.std(data)
        self.std = np.maximum(self.std, 1e-6) # 防止除以 0

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

    # 兼容原有的获取归一化参数的方法，返回 mean 和 std
    def get_data_scalar(self):
        """返回 (mean, std) 的标量值（此处为统一归一化后的标量）。"""
        # 由于是统一归一化，直接返回标量
        return self.mean, self.std




class DataScalar:
    """
    通用版：支持 1D (Tube) 和 2D (Sound) 的反归一化
    """
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
        
        """data: ndarray with shape b*t*h*w*c"""
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

        # --- 通道独立归一化 (Channel-wise Independent) ---
        if is_scalar:
            print("Calculating Independent Channel-wise Statistics...")
            
            # axis=(0, 1, 2, 3) 表示聚合 Batch, Time, H, W，保留 Channel 维度独立
            # shape: (1, 1, 1, 1, 3)
            self.mean = np.mean(data, axis=(0, 1, 2, 3), keepdims=True)
            self.std = np.std(data, axis=(0, 1, 2, 3), keepdims=True)
            
            # 防止除以 0
            self.std = np.maximum(self.std, 1e-6)

            print("-" * 30)
            print(f"DEBUG: Dataset Statistics (Independent)")
            print(f"  Mean shape: {self.mean.shape}")
            
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
        
        # 计算最大允许的起始时间索引
        max_time_idx = self.num_t - (self.num_frames - 1) * self.num_interval - 1
        i_t = np.random.choice(max_time_idx, 1)[0]
        
        # 你的数据 H=64, W=64，若全图训练则起始点为 0
        i_w = np.random.choice(self.num_w - self.num_w + 1, 1)[0]
        i_h = np.random.choice(self.num_h - self.num_h + 1, 1)[0]

        # 构造时间索引
        time_indices = i_t + np.arange(self.num_frames) * self.num_interval
        
        # 取数据 (Frames, H, W, C)
        x = self.data[i_b, time_indices, i_h:i_h+self.num_h, i_w:i_w+self.num_w, :]
        
        # 转置为 PyTorch 格式: (T, H, W, C) -> (T, C, H, W)
        x = x.transpose(0, 3, 1, 2)

        # 构造 Masks
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
    """
    直接接收 labels 数组。
    """
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
        self.labels = labels # (N_samples,) 存储真实的频率值 (例如 150, 150, ..., 400)

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
    """
    修改版：支持按通道独立归一化 (Channel-wise Normalization)
    """
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

        # --- 修改重点开始 ---
        # 需要在除了 Channel 以外的所有维度上求均值和标准差
        # data shape: (B, T, H, C) -> 我们要保留 C 维度
        # axis=(0, 1, 2) 表示对 Batch, Time, Space 求聚合，只留下 Channel 的统计量
        
        if is_scalar:
            self.mean = np.mean(data, axis=(0, 1, 2), keepdims=True) # shape: (1, 1, 1, C)
            self.std = np.std(data, axis=(0, 1, 2), keepdims=True)   # shape: (1, 1, 1, C)
            # 防止除以 0 
            self.std = np.maximum(self.std, 1e-6)

            print(f"Channel-wise Stats:")
            for c in range(self.num_c):
                print(f"  Ch{c}: Mean={self.mean[0,0,0,c]:.4e}, Std={self.std[0,0,0,c]:.4e}")

            self.data = (data - self.mean) / self.std
        else:
            # 如果不是 scalar 模式，通常也建议保留统计量以便反归一化
            self.mean = np.zeros((1, 1, 1, self.num_c))
            self.std = np.ones((1, 1, 1, self.num_c))
            self.data = data
        # --- 修改重点结束 ---


        print("-" * 30)
        print("DEBUG CHECK: Data Statistics")
        print(f"Mean shape: {self.mean.shape}") # 应该看到 Channel 维度，例如 (1,1,1,2)
        print(f"Std shape:  {self.std.shape}")

        # 打印具体数值
        p_std = self.std[0,0,0,0] # 假设通道0是声压
        v_std = self.std[0,0,0,1] # 假设通道1是振速

        print(f"Channel 0 (Pressure) Std: {p_std:.6f}")
        print(f"Channel 1 (Velocity) Std: {v_std:.6f}")

        if abs(p_std - v_std) < 1e-4:
            print("!!! 警告：两个通道的标准差几乎一样，你可能还在用统一归一化！")
        else:
            print(">>> 成功：检测到通道独立归一化，振速 Std 应该很小。")
        print("-" * 30)


        
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

'''
class DataScalar:
    """
    用于在计算物理 Loss 时将数据反归一化回真实单位
    """
    def __init__(self, mean, std):
        import torch
        # 确保转为 Tensor 并在正确的设备上 (在 forward 中会自动处理 device)
        self.mean = torch.tensor(mean).float()
        self.std = torch.tensor(std).float()
        
        # 调整维度以适配 [Batch, Time, Channel, Space]
        # Dataset 中 mean 是 [1, 1, 1, C] -> 对应 data [B, T, H, C]
        # 但是网络输出通常是 [Batch, Time, Channel, Space]
        # 所以我们需要把 mean/std 变为 [1, 1, C, 1]
        self.mean = self.mean.permute(0, 1, 3, 2) 
        self.std = self.std.permute(0, 1, 3, 2)

    def __call__(self, x):
        # x: Normalized data [Batch, Time, Channel, Space]
        # returns: Real unit data
        if x.device != self.mean.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        return x * self.std + self.mean
'''






        

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
        # data	ndarray	输入数据，形状为 (batch, time, height, width, channels)
        # length	int	数据集总样本数（通常为 num_samples 参数传入）
        # num_frames	int	每个样本包含的时间步数（输入序列长度）
        # num_interval	int	时间步采样间隔（如 num_interval=2 表示每隔2步采样一次）
        # is_scalar	bool	是否对数据进行标准化（均值和方差归一化）
        # dtype	str	输出数据类型（如 float32）
        # is_condition	bool	是否添加条件参数到输入中
        super().__init__()
        """data: ndarray with shape b*t*h*w*c"""
        self.length = length
        # 数据形状解析
        self.num_b = len(data)          # Batch维度
        self.num_t = len(data[0])       # 时间步数
        self.num_w = data.shape[-3]     # 空间宽度（x方向）
        self.num_h = data.shape[-2]     # 空间高度（y方向）
        self.num_c = data.shape[-1]     # 通道数（如速度u/v分量）
        self.num_frames = num_frames    # 每个训练样本中需要提取的时间步数（即输入模型的时间序列长度）
        self.num_interval = num_interval        # list [num_interval for dataset1, num_interval for dataset2, ...]
        self.is_condition = False        # 是否拼接条件参数
        # 生成二维网格, _ 表示忽略生成的 x_grid（不需要保存 x 坐标）。
        _, self.y = np.meshgrid(np.linspace(0, 2 * np.pi, self.num_w, endpoint=False),
                           np.linspace(0, 2 * np.pi, self.num_h, endpoint=False), indexing='ij')

        self.mean = np.mean(data)                                          # 全局均值
        self.std = np.std(data)                                            # 全局标准差
        if is_scalar:
            self.data = (data-self.mean)/self.std                          # 标准化
        else:
            self.data = data
        self.data = self.data.astype(dtype)                                # 类型转换（如float32）

    # 定义如何动态生成单个数据样本
    def __getitem__(self, item):
        # 随机选择批次、起始时间、空间窗口
        i_b = np.random.choice(self.num_b, 1)[0]
        # 随机选择一个起始时间点，用于提取连续的时间序列片段。
        # 最大允许起始位置：num_t - num_frames * num_interval（如 700 - 10 = 690）。
        i_t = np.random.choice(self.num_t-self.num_frames*self.num_interval, 1)[0]
        # 随机选择一个空间起始位置（x 方向），用于提取局部空间区域。
        # 最大允许起始位置：num_w - num_h + 1（如 64 - 64 + 1 = 1）。
        # 确保：从 i_w 开始，提取 num_h 个连续的 x 方向网格时，不超过数据范围。
        i_w = np.random.choice(self.num_w-self.num_h+1, 1)[0]

        # 提取时空数据 [T, H, W, C] → [T, C, H, W]
        # 时间维度：若 num_frames=16, num_interval=2：总覆盖时间步：16 * 2 = 32 实际提取索引：i_t, i_t+2, i_t+4, ..., i_t+30（共16步）
        # 空间维度：若 num_w=128, num_h=64：提取 x 方向第 i_w 到 i_w+64 的网格区域
        #切片后的数据形状:
        #输入形状：(batch, time, height, width, channels)
        #切片后形状：(num_frames, height, num_h, channels)
        x = self.data[i_b, i_t:i_t+self.num_frames*self.num_interval:self.num_interval, i_w:i_w+self.num_h]
        # 维度转置操作:
        # 原始维度顺序：(num_frames, height, num_h, channels) → 索引顺序 (0, 1, 2, 3)
        # 目标维度顺序：(num_frames, channels, height, num_h) → 新索引顺序 (0, 3, 1, 2)
        # 转置后的数据形状
        # 输出形状：(num_frames, channels, height, num_h)
        x = x.transpose(0, 3, 1, 2)

        # 数据增强：随机平移（利用周期性边界）
        shift_x, shift_y = np.random.choice(self.num_h, 1)[0], np.random.choice(self.num_w, 1)[0]   # data aug.
        # 沿高度（axis=2）和宽度（axis=3）方向循环平移数据。
        x = np.roll(x, (shift_x, shift_y), axis=(2, 3))     # data aug.
        # 隐变量掩码生成
        # 创建一个全为 True 的掩码，标记潜在变量需处理的位置
        # 形状：(num_frames, 1, 1, 1)，与输入数据的 (num_frames, channels, height, width) 兼容
        # 用途：在模型中标记需要参与潜在空间计算的区域（如所有时间步和位置）
        # np.astype(bool):数据类型转换为布尔类型（bool）
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        # 观测掩码生成：创建一个全为 False 的掩码，标记观测数据的位置
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        frame_indices = np.arange(self.num_frames)    # 生成从 0 到 num_frames-1 的连续整数数组。标识每个时间步的索引，用于模型的时间步跟踪或条件输入

        # 条件参数拼接（雷诺数和外力项）
        if self.is_condition:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W
        else:
            return x, frame_indices, obs_mask, latent_mask       # B T C H W
        # 为 DataLoader 提供数据集的总样本数
    def __len__(self):
        return self.length
 
class Dataset2DSound_smalldata1(Dataset):
    def __init__(self, data, length, num_frames, num_interval, 
                 is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()
        """
        data: 声学数据张量 (batch_size, time_steps, height, width, channels)
        """
        self.length = length
        self.num_frames = num_frames
        self.num_interval = num_interval
        self.is_condition = is_condition
        self.dtype = dtype
        
        # 数据形状信息
        self.num_b = data.shape[0]  # 实际批次大小
        self.num_t = data.shape[1]
        self.num_h = data.shape[2]
        self.num_w = data.shape[3]
        self.num_c = data.shape[4]
        
        # 创建条件参数（只包含实际存在的批次）
        # 点声源频率和边界加速度激励频率都来自 [50, 100, 150, 200, 250, 300]
        self.source_freq = np.zeros(self.num_b)
        self.boundary_freq = np.zeros(self.num_b)
        
        # 完整参数空间对应的频率
        full_frequencies = np.array([50, 100, 150, 200, 250, 300])
        
        # 训练集排除的测试点（源频，边界频）
        test_points = [
            (150, 150),  # 索引14
            (150, 200),  # 索引15
            (150, 250),  # 索引16
            (200, 150),  # 索引20
            (200, 200)   # 索引21
        ]
        
        # 遍历所有可能的参数组合
        available_idx = 0
        for source in full_frequencies:
            for boundary in full_frequencies:
                # 如果是测试点则跳过
                if (source, boundary) in test_points:
                    continue
                
                # 确保不超过实际批次数量
                if available_idx >= self.num_b:
                    break
                
                # 记录当前参数组合
                self.source_freq[available_idx] = source
                self.boundary_freq[available_idx] = boundary
                available_idx += 1
        
        # 数据标准化
        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data - self.mean) / self.std
        else:
            self.data = data
        
        # 转换数据类型
        if dtype == 'float32':
            self.data = self.data.astype(np.float32)
        elif dtype == 'float64':
            self.data = self.data.astype(np.float64)
    
    def __getitem__(self, item):
        # 随机选择批次和时间起点
        i_b = np.random.choice(self.num_b)
        i_t = np.random.choice(self.num_t - self.num_frames * self.num_interval)
        
        # 提取时间序列数据
        x = self.data[i_b, i_t:i_t + self.num_frames * self.num_interval:self.num_interval]
        # 转换为 PyTorch 格式:时间 × 通道 × 高度 × 宽度
        x = x.transpose(0, 3, 1, 2)
        
        # 获取当前批次的声源条件
        source_freq = self.source_freq[i_b]
        boundary_freq = self.boundary_freq[i_b]
        
        # 创建条件向量
        frame_indices = np.arange(self.num_frames)
        obs_mask = np.zeros([self.num_frames, 1, 1, 1]).astype(bool)
        latent_mask = np.ones([self.num_frames, 1, 1, 1]).astype(bool)
        
        if self.is_condition:
            # 返回数据+条件
            return (
                torch.tensor(x),  # 声场数据
                torch.tensor(frame_indices),  # 时间索引
                torch.tensor(obs_mask),  # 观测掩码
                torch.tensor(latent_mask),  # 潜在变量掩码
                torch.tensor([source_freq, boundary_freq])  # 声源条件
            )
        else:
            # 只返回数据
            return (
                torch.tensor(x),
                torch.tensor(frame_indices),
                torch.tensor(obs_mask),
                torch.tensor(latent_mask)
            )
    
    def __len__(self):
        return self.length



class Dataset2DGaussPulse(Dataset):
    def __init__(self, data, length, num_frames, num_interval, 
                 is_scalar=True, dtype='float32', is_condition=True):
        super().__init__()

        self.length = length
        self.num_b = len(data) # Total number of simulation batches (125 in your case)
        self.num_t = len(data[0]) if self.num_b > 0 else 0
        self.num_h = data.shape[-3]
        self.num_w = data.shape[-2]
        self.num_c = data.shape[-1]
        self.num_frames = num_frames
        self.num_interval = num_interval
        self.is_condition = is_condition

        _, self.y = np.meshgrid(np.linspace(0, 63, self.num_w, endpoint=False),
                           np.linspace(0, 63, self.num_h, endpoint=False), indexing='ij')

        ma_values = [0.3, 0.4, 0.5, 0.6, 0.7]
        v_beta_values = [0.02, 0.03, 0.04, 0.05, 0.06]
        ma_grid, v_beta_grid = np.meshgrid(ma_values, v_beta_values, indexing='ij')

        self.ma = ma_grid.reshape(-1) # Shape (25,)
        self.num_ma = len(self.ma)
        
        self.v = v_beta_grid.reshape(-1) # Shape (25,)

        # Data normalization
        self.mean = np.mean(data)
        self.std = np.std(data)
        if is_scalar:
            self.data = (data - self.mean) / self.std
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
            ma_condition = self.ma[i_b%self.num_ma]*1*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            v_beta_condition = self.v[i_b%self.num_ma]*1*np.ones_like(self.y)[np.newaxis, np.newaxis].repeat(self.num_frames, 0)
            return np.concatenate([x, ma_condition, v_beta_condition], axis=1), frame_indices, obs_mask, latent_mask
        else:
            return x, frame_indices, obs_mask, latent_mask

    def __len__(self):
        return self.length

import os
import sys
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from timeit import default_timer
import time

# Enable cuDNN algorithm selection for fixed input shapes.
torch.backends.cudnn.benchmark = True

# --- 导入 FNO 模型 (假设 fno.py 在 models 目录下) ---
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.fno import FNO2d
from scripts.baselines._cli import run

# --- 配置参数类 (通用配置) ---
class Config:
    # Paths and the channel count are selected by --case at launch.
    TRAIN_PATH = None
    TEST_PATH = None
    SAVE_DIR = None

    # 2. 训练超参数 (Batch Size 现在由 main 函数动态设置)
    LEARNING_RATE = 0.001
    EPOCHS = 200
    STEP_SIZE = 100
    GAMMA = 0.5

    # 3. FNO model parameters
    MODES = 16
    WIDTH = 64
    SEQUENCE_LENGTH = 10

    #kol
    #NUM_CHANNELS = 2
    #2dsound
    NUM_CHANNELS = 3

    OUTPUT_RES = 64

    NUM_WORKERS = 8

# --- 辅助函数：相对 L2 损失 ---
class LpLoss(object):
    """ 计算相对 L_p 损失 """
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def __call__(self, x, y):
        num_examples = x.size(0)
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

# --- 辅助函数：生成网格 ---
def get_grid(shape, device):
    """ 生成 (x, y) 坐标网格作为 FNO 的输入特征 """
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x, dtype=np.float32)).reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y, dtype=np.float32)).reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)

def load_data_file(path, config, label="Data"):
    """
    加载数据并将时间维度堆叠到通道维度。
    """
    print(f"Loading {label} from {path}...")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} file not found at {path}")

    # 原始数据: (Simulations, Time, H, W, C)
    data = np.load(path)
    if data.ndim != 5 or data.shape[-1] != config.NUM_CHANNELS:
        raise ValueError(f"expected [sample,time,x,y,{config.NUM_CHANNELS}] data, got {data.shape}")
    if data.shape[2:4] != (config.OUTPUT_RES, config.OUTPUT_RES):
        raise ValueError(f"expected spatial shape {(config.OUTPUT_RES, config.OUTPUT_RES)}, got {data.shape[2:4]}")
    if data.shape[1] < config.SEQUENCE_LENGTH:
        raise ValueError("the data must contain at least one complete time window")
    data = torch.tensor(data, dtype=torch.float)

    N, T_total, H, W, C = data.shape
    seq_len = config.SEQUENCE_LENGTH

    num_segments = T_total // seq_len
    data = data[:, :num_segments*seq_len, ...]

    data = data.reshape(N, num_segments, seq_len, H, W, C)
    data = data.permute(0, 1, 3, 4, 2, 5)
    data = data.reshape(-1, H, W, seq_len * C)

    print(f"{label} processed shape: {data.shape} (Time stacked into Channels)")
    return data

def train_and_evaluate(input_res, batch_size, config, x_train, x_test, device_id):
    """
    训练和评估一个特定下采样率的模型。

    Args:
        input_res (int): 稀疏输入的下采样步长 (e.g., 2 for 2x downsampling).
        batch_size (int): 当前任务使用的批次大小。
        config (Config): 全局配置对象.
        x_train (torch.Tensor): 训练数据 (CPU Tensor).
        x_test (torch.Tensor): 测试数据 (CPU Tensor).
        device_id (int): 要使用的 GPU 索引。
    """
    if device_id >= 0 and torch.cuda.is_available():
        device = f'cuda:{device_id}'
        # 在多进程环境中，必须在每个进程内部设置设备
        torch.cuda.set_device(device_id)
    else:
        device = 'cpu'

    # Use the downsampling factor in checkpoint filenames.
    downsample_factor_label = input_res

    print(f"\n--- Starting Experiment: {downsample_factor_label}x Downsampling (INPUT_RES={input_res}) [Batch={batch_size}] on DEVICE: {device} ---")

    # 1. 数据加载器 (使用动态 Batch Size)
    train_loader = DataLoader(
        TensorDataset(x_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device != 'cpu' else False
    )
    test_loader = DataLoader(
        TensorDataset(x_test),
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True if device != 'cpu' else False
    )

    # 2. 初始化模型
    model = FNO2d(
        num_channels=config.NUM_CHANNELS,
        out_channels=config.NUM_CHANNELS * config.SEQUENCE_LENGTH,
        modes1=config.MODES,
        modes2=config.MODES,
        width=config.WIDTH,
        initial_step=config.SEQUENCE_LENGTH
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.STEP_SIZE, gamma=config.GAMMA)
    myloss = LpLoss(size_average=False)

    # 3. 训练循环
    print(f"Training on {len(x_train)} samples for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        model.train()
        t1 = default_timer()
        train_l2 = 0

        for y_batch, in train_loader:
            y_batch = y_batch.to(device, non_blocking=True)

            # --- 构造稀疏输入 ---
            B, H, W, _ = y_batch.shape
            y_reshaped = y_batch.view(B, H, W, config.SEQUENCE_LENGTH, config.NUM_CHANNELS)

            # 1. 下采样 (Spatial Downsampling)
            x_low_res = y_reshaped[:, ::input_res, ::input_res, :, :]

            # 2. 双线性插值回 64x64
            low_res_h = x_low_res.shape[1]
            low_res_w = x_low_res.shape[2]

            x_input = x_low_res.permute(0, 3, 4, 1, 2).reshape(-1, config.NUM_CHANNELS, low_res_h, low_res_w)
            x_input = F.interpolate(x_input, size=(config.OUTPUT_RES, config.OUTPUT_RES),
                                    mode='bilinear', align_corners=True)

            # 3. 还原回 FNO 输入格式: (B, 64, 64, 20)
            x_input = x_input.reshape(B, config.SEQUENCE_LENGTH, config.NUM_CHANNELS, H, W).permute(0, 3, 4, 1, 2)
            x_input = x_input.reshape(B, H, W, config.SEQUENCE_LENGTH * config.NUM_CHANNELS)

            # 4. 准备 FNO 输入
            grid = get_grid(x_input.shape, device)

            optimizer.zero_grad()
            out = model(x_input, grid)
            out = out.squeeze(-2)

            loss = myloss(out.view(B, -1), y_batch.view(B, -1))
            loss.backward()
            optimizer.step()
            train_l2 += loss.item()

        scheduler.step()
        train_l2 /= len(x_train)
        t2 = default_timer()

        # --- 验证阶段 ---
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for y_test_batch, in test_loader:
                y_test_batch = y_test_batch.to(device, non_blocking=True)

                # 同样的预处理逻辑 (使用当前 input_res)
                B, H, W, _ = y_test_batch.shape
                y_test_reshaped = y_test_batch.view(B, H, W, config.SEQUENCE_LENGTH, config.NUM_CHANNELS)
                x_test_low = y_test_reshaped[:, ::input_res, ::input_res, :, :]

                low_res_h = x_test_low.shape[1]
                low_res_w = x_test_low.shape[2]

                x_test_in = x_test_low.permute(0, 3, 4, 1, 2).reshape(-1, config.NUM_CHANNELS, low_res_h, low_res_w)
                x_test_in = F.interpolate(x_test_in, size=(config.OUTPUT_RES, config.OUTPUT_RES),
                                          mode='bilinear', align_corners=True)
                x_test_in = x_test_in.reshape(B, config.SEQUENCE_LENGTH, config.NUM_CHANNELS, H, W).permute(0, 3, 4, 1, 2)
                x_test_in = x_test_in.reshape(B, H, W, config.SEQUENCE_LENGTH * config.NUM_CHANNELS)

                grid_test = get_grid(x_test_in.shape, device)

                out_test = model(x_test_in, grid_test)
                out_test = out_test.squeeze(-2)

                test_l2 += myloss(out_test.view(B, -1), y_test_batch.view(B, -1)).item()

        test_l2 /= len(x_test)

        if (epoch + 1) % 10 == 0 or epoch == config.EPOCHS - 1:
            print(f"[GPU {device_id} | {downsample_factor_label}x | B={batch_size}] Epoch: {epoch + 1}/{config.EPOCHS} | Time: {t2-t1:.2f}s | Train Loss: {train_l2:.5f} | Test Loss: {test_l2:.5f}")

    # 4. 保存模型 (使用正确的 factor 标签)

    #kol
    #save_path = os.path.join(config.SAVE_DIR, f'fno_reconstruction_{downsample_factor_label}x.pth')

    #2dsound
    save_path = os.path.join(config.SAVE_DIR, f'{config.FILE_PREFIX}_reconstruction_{downsample_factor_label}x.pth')

    torch.save(model.state_dict(), save_path)
    print(f"[GPU {device_id} | {downsample_factor_label}x | B={batch_size}] Model saved to: {save_path}")

def main():
    run(Config, load_data_file, train_and_evaluate, model="fno",
        normalizer=None, batches={2: 350, 3: 768, 4: 768, 5: 768, 6: 350})


if __name__ == "__main__":
    main()

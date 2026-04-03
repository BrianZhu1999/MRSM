import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))


import numpy as np
import os
import torch
import torch.nn as nn
import sys
import logging
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from collections import namedtuple

from configs import config
from models import get_model, get_ema
from sampler import VESDE, setup_seed
from trainer import loss_fn_video, fpe_regularizer_loss
from trainer import restore_checkpoint, save_checkpoint
from trainer import get_dataset
from einops import rearrange

from torch.utils.data import DataLoader, Subset


def train(config):
    setup_seed(config.seed)

    log_path = config.results_path + '/log.txt'
    loss_log_path = config.results_path + '/loss_log.npy'
    checkpoint_path = config.results_path + '/checkpoint.pth'
    #checkpoint_path_800 = config.results_path + '/checkpoint_800.pth'
    f = open(log_path, 'w+', encoding='utf-8')

    net = get_model(config)
    net = nn.DataParallel(net)
    net.to(device)
    ema = get_ema(net.parameters(), decay=config.ema_rate)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
    state = dict(optimizer=optimizer, model=net, ema=ema, epoch=0, iteration=0, loss_train=[], loss_val=[])
    if config.continue_training:
        state = restore_checkpoint(checkpoint_path, state, device)
        #state = restore_checkpoint(checkpoint_path_800, state, device)
    initial_epoch = int(state['epoch'])

    #train
    x = np.load(config.data_location)
    data = get_dataset(config, x, train=True)
    # data_val = get_dataset(config, x, train=False)
    np.save(config.results_path + '/scalar.npy', [data.mean, data.std])
    dataloader = DataLoader(data, batch_size=config.batch_size, drop_last=False, shuffle=True)
    # dataloader_val = DataLoader(data_val, batch_size=config.batch_size, drop_last=False, shuffle=False)

    #finetune
    '''
    x = np.load(config.data_location)
    full_data = get_dataset(config, x, train=True)
    new_num_samples = 10000
    print(f"Original dataset size: {len(full_data)}. Using random subset of {new_num_samples} for fine-tuning.")
    indices = torch.randperm(len(full_data))[:new_num_samples]
    data = Subset(full_data, indices)
    np.save(config.results_path + '/scalar.npy', [full_data.mean, full_data.std])
    dataloader = DataLoader(data, batch_size=config.batch_size, drop_last=False, shuffle=True)
    '''

    sde = VESDE(config, sigma_min=config.beta_min, sigma_max=config.beta_max, N=config.num_scales)
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters())}")

    state['model'].train()
    loss_log = []
    loss_val_log = []
    loss_val_min = np.inf if len(state['loss_val'])<1 else state['loss_val'][-1]
    print(loss_val_min)
    f.write('starting training...\n')
    f.flush()
    for epoch in range(initial_epoch, config.epochs):
        loss_avg = 0
        i = 0
        
        #train
        '''
        loader = tqdm(enumerate(dataloader), desc=f'training epoch {epoch}...',
                  total=int(config.num_samples_train // config.batch_size)) if config.verbose else enumerate(dataloader)
        '''
        
        #finetune
        loader = tqdm(enumerate(dataloader), desc=f'training epoch {epoch}...',
                    total=len(dataloader)) if config.verbose else enumerate(dataloader)
        
        for i, (x_aug, frame_indices, obs_mask, latent_mask) in loader:
            state['optimizer'].zero_grad()
            x_aug = x_aug.to(device).float()
            # np.save(config.results_path + '/test.npy', x_aug.detach().cpu().numpy())
            frame_indices = frame_indices.to(device)
            obs_mask = obs_mask.to(device).float()
            latent_mask = latent_mask.to(device).float()
            kwargs = {
                'frame_indices': frame_indices,
                'obs_mask': obs_mask,
                'latent_mask': latent_mask,
            }
            dsm_loss = loss_fn_video(state['model'], sde, x_aug, **kwargs)
            fpe_loss = fpe_regularizer_loss(state['model'], sde, x_aug, alpha=0.0001, beta=0.00, m=1, **kwargs)
            #fpe_loss = fpe_regularizer_loss(state['model'], sde, x_aug, alpha=0.0001, beta=0.00, m=1, **kwargs)
            #fpe_loss = fpe_regularizer_loss(state['model'], sde, x_aug, alpha=0.00001, beta=0.00, m=1, **kwargs)
            loss = dsm_loss + fpe_loss

            loss.backward()
            state['optimizer'].step()
            loss_avg += loss.detach().cpu().numpy()
            state['ema'].update(state['model'].parameters())
        loss_avg /= i + 1
        loss_log.append(loss_avg)
        state['loss_train'] = loss_log

        if epoch % config.print_freq == 0:
            # with torch.no_grad():
            #     loss_val_avg = 0
            #     for j, (x_aug, frame_indices, obs_mask, latent_mask) in tqdm(enumerate(dataloader_val),
            #                             desc=f'training epoch {epoch}...',
            #                             total=int(config.num_samples_val // config.batch_size)):
            #         x_aug = x_aug.to(device).float()
            #         frame_indices = frame_indices.to(device)
            #         obs_mask = obs_mask.to(device).float()
            #         latent_mask = latent_mask.to(device).float()
            #         kwargs = {
            #             'frame_indices': frame_indices,
            #             'obs_mask': obs_mask,
            #             'latent_mask': latent_mask,
            #         }
            #         loss_val = loss_fn_video(state['model'], sde, x_aug, **kwargs)
            #         loss_val_avg += loss_val.detach().cpu().numpy()
            #     loss_val_avg /= j + 1
            #     loss_val_log.append(loss_val_avg)
            #     state['loss_val'] = loss_val_log
            # f.write(f'epoch: {epoch}\tloss: {loss_avg}\tloss_val: {loss_val_avg}\n')
            # f.flush()
            # if loss_val_avg < loss_val_min:
            #     loss_val_min = loss_val_avg
            #     save_checkpoint(checkpoint_path, state)
            np.save(loss_log_path, np.array(loss_log))
            save_checkpoint(checkpoint_path, state)
            f.write(f'Training loss at epoch {epoch}: {loss_avg:.5f}\n')
            f.flush()
            print(f'Training loss at epoch {epoch}: {loss_avg:.5f}\n')
        if epoch % 1 == 0:
            save_checkpoint(config.results_path + f'/checkpoint_{epoch}.pth', state)
        state['epoch'] += 1
    save_checkpoint(checkpoint_path, state)
    np.save(loss_log_path, np.array(loss_log))
    f.write('model trained!')
    f.flush()
    f.close()


if __name__ == "__main__":
    config.cuda = config.gpu is not None
    if config.cuda:
        # torch.cuda.set_device(config.gpu)
        device = 'cuda'
    else:
        device = 'cpu'
    config.device = device



    
    '''create results folder'''
    #path = config.results_path + '/' + config.data + '_' + config.version
    #config.results_path = path
    original_results_path = config.results_path + '/' + config.data + '_' + config.version
    finetune_results_path = original_results_path + '_finetune_FPE'
    #finetune_results_path = original_results_path + '_finetune_FPE2'
    config.results_path = finetune_results_path


    used_para = dict(
        epochs = config.epochs,
        batch_size=config.batch_size,
        data_location=config.data_location,
        num_channels = config.num_components + config.num_conditions,
        verbose=config.verbose
        )
    
    config.num_channels = config.num_components + config.num_conditions
    
    '''
    if not os.path.exists(path):
        os.mkdir(path)
    '''
    
    if not os.path.exists(finetune_results_path):
        os.makedirs(finetune_results_path)
        
    if not config.continue_training:
        with open(config.results_path + "/config.json", mode="w") as f:
            json.dump(config.__dict__, f, indent=4)
    else:
        '''load option file'''
        #opt_path = path + '/config.json'
        opt_path = original_results_path + '/config.json'
        with open(opt_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            config['continue_training'] = True
            for key in used_para.keys():
                config[key] = used_para[key]
        config['results_path'] = finetune_results_path
        OPT_class = namedtuple('OPT_class', config.keys())
        config = OPT_class(**config)

    train(config)

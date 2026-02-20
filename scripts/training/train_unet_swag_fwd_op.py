import sys
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np

from src import DATASETS, get_loaders_wmvf_patch, SWAG, DD_loss, train_model_swag as train, UNetModelWrapperWOTime as UNetModel

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy

def create_dir(path, config):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Directory '{path}' created.")
    else:
        assert config.restart != False, "Are you restarting?"
        print(f"Directory '{path}' already exists.")

def main(config_path):

    config = OmegaConf.load(config_path)
    
    th.manual_seed(config.th_seed)
    np.random.seed(config.np_seed)
    
    logpath = config.path + f"/exp_{config.exp_num}"
    savepath = logpath + "/saved_state"
    create_dir(logpath, config=config)
    create_dir(savepath, config=config)
    
    dev = th.device(config.device)
    
    writer = SummaryWriter(log_dir=logpath)
    
    train_dataloader, test_dataloader = get_loaders_wmvf_patch(wm_paths=config.dataloader.wm_paths,
                                        vf_paths=config.dataloader.vf_paths,
                                        batch_size=config.dataloader.batch_size,
                                        time_cutoff=config.dataloader.time_cutoff,
                                        patch_dims=config.dataloader.patch_dims,
                                        dataset_=DATASETS[config.dataloader.dataset],
                                        x_spatial_cutoff=config.dataloader.x_spatial_cutoff if hasattr(config.dataloader, 'x_spatial_cutoff') else None,
                                        jump=config.dataloader.jump if hasattr(config.dataloader, 'jump') else None,
                                        scale_inputs=config.dataloader.scale_inputs if hasattr(config.dataloader, 'scale_inputs') else False)
        
    model = UNetModel(dim=config.unet.dim,
                      channel_mult=config.unet.channel_mult,
                      out_dim=config.unet.out_dim,
                      num_channels=config.unet.num_channels,
                      num_res_blocks=config.unet.res_blocks,
                      num_head_channels=config.unet.head_chans,
                      attention_resolutions=config.unet.attn_res,
                      dropout=config.unet.dropout,
                      use_new_attention_order=config.unet.new_attn,
                      use_scale_shift_norm=config.unet.film
                      )
    
    swag_model = SWAG(base_model= deepcopy(model), device=dev, variance_clamp=config.swag.variance_clamp, max_rank=config.swag.max_rank)
    
    swag_model.to(dev)

    model.to(dev)

    optim = Adam(model.parameters(), lr=config.optimizer.lr)

    sched = None
        
    loss_fn = DD_loss
    
    train(model=model,
                swag_model=swag_model,
                swag_start=config.swag.swag_start,
                swag_epoch_int=config.swag.swag_epoch_int,
                swag_eval_rounds=config.swag.swag_eval_rounds,
                train_dataloader=train_dataloader,
                test_dataloader=test_dataloader,
                optimizer=optim,
                sched=sched,
                loss_fn=loss_fn,
                writer=writer,
                num_epochs=config.num_epochs,
                print_epoch_int=config.print_epoch_int,
                save_epoch_int=config.save_epoch_int,
                print_within_epoch_int=config.print_with_epoch_int,
                path=savepath,
                device=dev,
                restart=config.restart,
                restart_epoch=config.restart_epoch)

if __name__ == '__main__':
    main(sys.argv[1])
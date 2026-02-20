import sys
from omegaconf import OmegaConf
import os

import torch as th
import numpy as np

from src import DATASETS, get_loaders_wmvf_baseline, DD_loss, train_model_baseline as train, FCN

from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

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
    
    train_dataloader, test_dataloader = get_loaders_wmvf_baseline(wm_paths=config.dataloader.wm_paths,
                                        vf_paths=config.dataloader.vf_paths,
                                        batch_size=config.dataloader.batch_size,
                                        time_cutoff=config.dataloader.time_cutoff,
                                        dataset_=DATASETS[config.dataloader.dataset],
                                        jump=config.dataloader.jump,
                                        scale_inputs=config.dataloader.scale_inputs,
                                        wm_vf=config.dataloader.wm_vf)
        
    model = FCN(config.model.pad_size)

    model.to(dev)
    
    optim = Adam(model.parameters(), lr=config.optimizer.lr)
    
    sched =  None
    
    loss_fn = DD_loss
    
    train(model=model,
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
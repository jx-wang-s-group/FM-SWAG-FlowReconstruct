import sys
import numpy as np
import torch as th
from omegaconf import OmegaConf
from src import UNetModelWrapper as UNetModel
from src import infer

def main(config_path):
    config = OmegaConf.load(config_path)
    
    # Set device
    device = th.device(config.device)
    
    # Data config
    y_val = config.data.y
    y_m_path = config.data.y_m_path
    y_std_path = config.data.y_std_path
    
    # Load scaling stats
    m, std = np.load(y_m_path), np.load(y_std_path)
    
    # Wall normal mapping
    wall_norm = {5: 0, 20: 1, 40: 2} 
    if y_val not in wall_norm.keys():
        print(f"Warning: y={y_val} not in known wall_norm keys {wall_norm.keys()}. Assuming class 0.")
    
    y_class = wall_norm[y_val] if y_val in wall_norm else 0

    # Load FM model
    cfm_model = UNetModel(dim=tuple(config.fm.dim),
                      channel_mult=config.fm.channel_mult,
                      num_channels=config.fm.num_channels,
                      num_res_blocks=config.fm.res_blocks,
                      num_head_channels=config.fm.head_chans,
                      attention_resolutions=config.fm.attn_res,
                      dropout=config.fm.dropout,
                      use_new_attention_order=config.fm.new_attn,
                      use_scale_shift_norm=config.fm.film,
                      class_cond=config.fm.class_cond,
                      num_classes=config.fm.num_classes
                      )
    
    state = th.load(config.fm.checkpoint_path, map_location=device, weights_only=True)
    cfm_model.load_state_dict(state["model_state_dict"])
    cfm_model.to(device)
    cfm_model.eval()
    
    # Prepare arguments
    ode_kwargs = OmegaConf.to_container(config.ode)
    
    y_tensor = y_class * th.ones(config.gen.batch_size, device=device, dtype=th.int)
    
    samples = infer(
        dims_of_img=tuple(config.fm.dim),
        total_samples=config.gen.n_samples,
        samples_per_batch=config.gen.batch_size,
        use_odeint=True,
        cfm_model=cfm_model,
        t_start=0.0,
        t_end=1.0,
        scale=True,
        device=device,
        m=m,
        std=std,
        t_steps=config.ode.num_steps,
        y=y_tensor,
        y0_provided=False,
        all_traj=False,
        **ode_kwargs
    )
    
    np.save(config.gen.save_path, samples)
    print(f"Samples saved to {config.gen.save_path}")

if __name__ == "__main__":
    OmegaConf.register_new_resolver("as_tuple", lambda *args: tuple(args))
    main(sys.argv[1])

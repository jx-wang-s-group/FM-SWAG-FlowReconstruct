import sys
import numpy as np
import torch as th
from src import UNetModelWrapperWOTime as UNetModel, SWAG
from omegaconf import OmegaConf
from tqdm import tqdm

def main(config_path):
    config = OmegaConf.load(config_path)

    # load info from config file
    y = config.data.y
    wm_m_path = config.data.wm_m_path
    wm_std_path = config.data.wm_std_path

    need_noisy_meas = config.meas.need_noisy_meas
    noise_scale = config.meas.noise_scale

    epoch = config.model.epoch
    batch_size = config.model.batch_size
    swag_count = config.model.swag_count
    var_clamp = config.model.var_clamp
    rank = config.model.rank

    arch_dim = list(config.arch.dim)
    arch_num_channels = config.arch.num_channels
    arch_out_dim = config.arch.out_dim
    arch_num_res_blocks = config.arch.num_res_blocks
    arch_num_head_channels = config.arch.num_head_channels
    arch_attention_resolutions = config.arch.attention_resolutions
    arch_dropout = config.arch.dropout
    arch_use_new_attention_order = config.arch.use_new_attention_order
    arch_use_scale_shift_norm = config.arch.use_scale_shift_norm

    dev = config.device

    # For conditioning on y^+
    wall_norm = {5: 0, 20: 1, 40: 2}
    assert y in wall_norm.keys(), "Check the y (wall normal) value provided"
    
    # load wall-measurement statistics (used to denormalize model output)
    m = np.load(wm_m_path)
    std = np.load(wm_std_path)

    # set device
    device = th.device(dev)

    # load velocity-field normalization statistics (input)
    m_ = np.load(f"data/stats/input/{y}/m.npy")
    std_ = np.load(f"data/stats/input/{y}/std.npy")

    # load and normalize velocity-field measurements (input to the network)
    X = np.concatenate([np.load(f"data/input/channel_180_{vel}_y{y}_test.npy") for vel in ['u', 'v', 'w']], axis=1)
    X = (X[:5000:10] - m_[:, :3])/std_[:, :3]

    measurements = th.from_numpy(X).to(device).type(th.float32)
    if need_noisy_meas:
        measurements += noise_scale * th.rand_like(measurements, device=device)

    # load SWAG model
    model = UNetModel(dim=arch_dim,
                      num_channels=arch_num_channels,
                      out_dim=arch_out_dim,
                      num_res_blocks=arch_num_res_blocks,
                      num_head_channels=arch_num_head_channels,
                      attention_resolutions=arch_attention_resolutions,
                      dropout=arch_dropout,
                      use_new_attention_order=arch_use_new_attention_order,
                      use_scale_shift_norm=arch_use_scale_shift_norm)
    state = th.load(f"ckpt/checkpoint_{epoch}.pth", weights_only=True)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval();

    swag_model = SWAG(model, device, var_clamp, rank)
    swag_model.load_state_dict(state["swag_model_state_dict"])
    swag_model.to(device)
    swag_model.eval();

    # Generate samples (SWAG inference)
    minibatches = measurements.shape[0]//batch_size
    samples = []
    for _ in tqdm(range(swag_count)):
        swag_model.sample()
        samples_list = []
        for i in range(minibatches):
            with th.no_grad():
                out = swag_model(measurements[i * batch_size : (i + 1) * batch_size]).detach().cpu().numpy()
            samples_list.append(out)
        samples.append(np.concatenate(samples_list)*std + m)
    samples = np.stack(samples) if swag_count > 1 else samples[0]

    np.save(config.save_path, samples)

if __name__=="__main__":
    main(sys.argv[1])

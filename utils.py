import os, shutil, yaml, torch, pickle
from shutil import copyfile
from si_encoder.model import SingerIdEncoder
from collections import OrderedDict


def setup_sie(config):
    sie_checkpoint = torch.load(os.path.join(config.sie_path, 'saved_model.pt'))
    new_state_dict = OrderedDict()
    sie_num_feats_used = sie_checkpoint['model_state']['lstm.weight_ih_l0'].shape[1]
    sie_num_voices_used = sie_checkpoint['model_state']['class_layer.weight'].shape[0]
    for i, (key, val) in enumerate(sie_checkpoint['model_state'].items()):
        new_state_dict[key] = val 
    sie =  SingerIdEncoder(config.device, torch.device("cpu"), sie_num_voices_used, sie_num_feats_used)
    for param in sie.parameters():
        param.requires_grad = False
    sie_optimizer = torch.optim.Adam(sie.parameters(), 0.0001)
    sie.load_state_dict(new_state_dict)
    for state in sie_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(config.device)
    sie.to(config.device)
    sie.eval()
    return sie


def setup_gen(config, Generator, num_feats):
    G = Generator(config.dim_neck, config.dim_emb, config.dim_pre, config.freq, num_feats)
    g_optimizer = torch.optim.Adam(G.parameters(), config.adam_init)
    g_checkpoint = torch.load(config.autovc_ckpt, map_location='cpu')
    G.load_state_dict(g_checkpoint['model_state_dict'])
    g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
    # fixes tensors on different devices error
    # https://github.com/pytorch/pytorch/issues/2830
    for state in g_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(config.which_cuda)
    G.to(config.device)
    return G


# this function seems like a hack - find out the standard method for passing boolean values as parser args
def str2bool(v):
    return v.lower() in ('true')


# as it says on the tin
def overwrite_dir(directory):
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.makedirs(directory)


"finds the index for each new song in dataset"
def new_song_idx(dataset):
    new_song_idxs = []
    song_idxs = list(range(255))
    for song_idx in song_idxs:
        for ex_idx, ex in enumerate(dataset):
            if ex[1] == song_idx:
                new_song_idxs.append(ex_idx)
                break
    return new_song_idxs


"Setup and populate new directory for model"
def new_dir_setup(config):
    model_dir_path = os.path.join(config.model_dir, config.file_name)
    overwrite_dir(model_dir_path)
    os.makedirs(model_dir_path +'/ckpts')
    os.makedirs(model_dir_path +'/generated_wavs')
    os.makedirs(model_dir_path +'/image_comparison')
    # save config in multiple formats (choose suitable one later)
    config_dict = {attr: getattr(config, attr) for attr in dir(config) if not attr.startswith('_')}
    with open(model_dir_path +'/config.pkl', 'wb') as config_file:
        pickle.dump(config, config_file)
    with open(model_dir_path +'/config.yaml', 'w') as config_file:
        yaml.dump(config_dict, config_file, default_flow_style=False)
    # open(model_dir_path +'/config.txt', 'a').write(str(config))
    copyfile(config.feature_dir +'/feat_params.yaml', (model_dir_path +'/feat_params.py'))
    copyfile('./model_vc.py',(model_dir_path +'/this_model_vc.py'))
    copyfile('./sv_converter.py',(model_dir_path +'/sv_converter.py'))
    copyfile('./main.py',(model_dir_path +'/main.py'))


"Replace config values with those of previous config file"
def use_prev_config_vals(config):
    max_iters = config.max_iters
    file_name = config.file_name
    autovc_ckpt = config.autovc_ckpt
    sie_path = config.sie_path
    ckpt_weights = config.ckpt_weights
    ckpt_freq = config.ckpt_freq
    config = pickle.load(open(os.path.join(config.model_dir, config.ckpt_weights, 'config.pkl'), 'rb'))
    config.ckpt_weights = ckpt_weights
    config.max_iters = max_iters
    config.file_name = file_name
    config.autovc_ckpt = autovc_ckpt
    config.sie_path = sie_path
    config.ckpt_freq = ckpt_freq


"Process config object, reassigns values if necessary, raise exceptions"
def process_config(config):
    if (config.ckpt_weights != '') and (config.use_ckpt_config == True): # if using pretrained weights
        use_prev_config_vals(config)
    if config.file_name == config.ckpt_weights:
        raise Exception("Your file name and ckpt_weights name can't be the same")
    if not config.ckpt_freq%int(config.train_iter*0.2) == 0 or not config.ckpt_freq%int(config.train_iter*0.2) == 0:
        raise Exception(f"ckpt_freq {config.ckpt_freq} and spec_freq {config.spec_freq} need to be a multiple of val_iter {int(config.train_iter*0.2)}")
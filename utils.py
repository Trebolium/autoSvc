import os, shutil, yaml, torch, pickle
from shutil import copyfile
from model_sie import SingerIdEncoder
from collections import OrderedDict
from train_params import *
import numpy as np
import matplotlib.pyplot as plt

def setup_sie(config):
    sie_checkpoint = torch.load(os.path.join(SIE_path, 'saved_model.pt'))
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
    G = Generator(dim_neck, dim_emb, dim_pre, sample_freq, num_feats)
    g_optimizer = torch.optim.Adam(G.parameters(), config.adam_init)
    g_checkpoint = torch.load(config.autovc_ckpt, map_location='cpu')
    G.load_state_dict(g_checkpoint['model_state_dict'])
    g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
    # fixes tensors on different devices error
    # https://github.com/pytorch/pytorch/issues/2830
    for state in g_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(which_cuda)
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
def new_dir_setup():
    model_dir_path = os.path.join(svc_model_dir, svc_model_name)
    overwrite_dir(model_dir_path)
    os.makedirs(model_dir_path +'/ckpts')
    os.makedirs(model_dir_path +'/generated_wavs')
    os.makedirs(model_dir_path +'/image_comparison')
    os.makedirs(model_dir_path +'/input_tensor_plots')
    copyfile('./model_vc.py',(model_dir_path +'/this_model_vc.py'))
    copyfile('./sv_converter.py',(model_dir_path +'/sv_converter.py'))
    copyfile('./main.py',(model_dir_path +'/main.py'))
    copyfile('./train_params.py',(model_dir_path +'/train_params.py'))


"Process config object, reassigns values if necessary, raise exceptions"
def process_config():
    if svc_model_name == svc_ckpt_path:
        raise Exception("Your file name and svc_ckpt_path name can't be the same")
    if not ckpt_freq%int(train_iter*0.2) == 0 or not ckpt_freq%int(train_iter*0.2) == 0:
        raise Exception(f"ckpt_freq {ckpt_freq} and spec_freq {spec_freq} need to be a multiple of val_iter {int(train_iter*0.2)}")


# check use_aper_feats boolean to produce total num feats being used for training
def determine_dim_size(SIE_params, SVC_params):

    if use_aper_feats:

        if 'world' in SIE_feat_dir: # requires no else a mel feature set means leave num_feats as is
            SIE_params['num_feats'] = SIE_params['num_harm_feats'] + SIE_params['num_aper_feats']
        if 'world' in SVC_feat_dir:
            SVC_params['num_feats'] = SVC_params['num_harm_feats'] + SVC_params['num_aper_feats']

    else:
        if 'world' in SIE_feat_dir:
            SIE_params['num_feats'] = SIE_params['num_harm_feats']
        if 'world' in SVC_feat_dir:
            SVC_params['num_feats'] = SVC_params['num_harm_feats']

    return SIE_params, SVC_params


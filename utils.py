import os, torch
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
import os, pickle, random, argparse, yaml, sys, pdb
from torch.backends import cudnn
sys.path.insert(1, '/homes/bdoc3/my_utils') # only need to do this once in the main script
from sv_converter import AutoSvc
from train_params import *
from data_objects.data_loaders import load_primary_dataloader, load_val_dataloaders
from utils import new_dir_setup, determine_dim_size


if __name__ == '__main__':
    new_dir_setup()

    cudnn.benchmark = True # For fast training.
    random.seed(1)
    # collects relevant feat params and create 'num_feats' parameter for each
    with open(os.path.join(SIE_feat_dir, 'feat_params.yaml')) as File:
        SIE_feats_params = yaml.load(File, Loader=yaml.FullLoader)
    if SVC_feat_dir != '':
        with open(os.path.join(SVC_feat_dir, 'feat_params.yaml')) as File:
            SVC_feats_params = yaml.load(File, Loader=yaml.FullLoader)
    else:
        SVC_feats_params = SIE_feats_params
    SIE_feats_params, SVC_feats_params = determine_dim_size(SIE_feats_params, SVC_feats_params)
    train_dataset, train_loader = load_primary_dataloader(SIE_feats_params, 'train', SVC_feats_params)

    # make eval loaders
    if eval_all:
        val_loaders = load_val_dataloaders(SIE_feats_params, SVC_feats_params)
    else:
        _, val_loader = load_primary_dataloader(SIE_feats_params, 'val', SVC_feats_params)
        val_loaders = [('damp', val_loader)]

    solver = AutoSvc(train_loader, SIE_feats_params, SVC_feats_params)
    current_iter = solver.get_current_iters()
    
    # training loop
    log_list = []
    while current_iter < max_iters:
        current_iter, log_list = solver.iterate('train', train_loader, current_iter, train_iter, log_list)
        for ds_label, val_loader in val_loaders:
            current_iter, log_list = solver.iterate(f'val_{ds_label}', val_loader, current_iter, int(train_iter*0.2), log_list)

    # Finish writing and save log
    solver.closeWriter()
    with open(os.path.join(svc_model_dir, svc_model_name, 'log_list.pkl'), 'wb') as File:
        pickle.dump(log_list, File)


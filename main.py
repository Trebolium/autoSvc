import random, argparse, yaml, pdb
from torch.backends import cudnn
import sys, os
if os.path.abspath('../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('../my_utils')) # only need to do this once in the main script

from vc_training import AutoSvc
from train_params import *
from data_loaders import load_primary_dataloader 
from utils import new_dir_setup, determine_dim_size, str2bool


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)   
    parser.add_argument("-a", "--ask", type=str2bool, default=True, help= "If False and the model name directory already exists, it will be overwritten without asking the user")
    config = parser.parse_args()

    new_dir_setup(config.ask, SVC_models_dir, SVC_model_name)

    cudnn.benchmark = True # For fast training.
    random.seed(1)

    # collects relevant feat params
    with open(os.path.join(SIE_feat_dir, 'feat_params.yaml')) as File:
        SIE_feats_params = yaml.load(File, Loader=yaml.FullLoader)
    if SVC_feat_dir != '':
        with open(os.path.join(SVC_feat_dir, 'feat_params.yaml')) as File:
            SVC_feats_params = yaml.load(File, Loader=yaml.FullLoader)
    else:
        SVC_feats_params = SIE_feats_params

    # creates 'num_feats' parameter for each
    SIE_feats_params, SVC_feats_params = determine_dim_size(SIE_feats_params, SVC_feats_params, sie_feat_type, svc_feat_type, use_aper_feats)
    _, train_loader = load_primary_dataloader(SIE_feats_params, 'train', SVC_feats_params, subset_size, chosen_class_num, bs=batch_size, workers=num_workers)
    _, val_loader = load_primary_dataloader(SIE_feats_params, 'val', SVC_feats_params, subset_size, chosen_class_num, bs=batch_size, workers=num_workers)
    val_loaders = [(use_loader, val_loader)]

    # initiate model and train
    solver = AutoSvc(SIE_feats_params, SVC_feats_params)
    solver.train(train_loader, val_loaders)
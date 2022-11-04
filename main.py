import os, pickle, random, argparse, yaml, sys, pdb
from torch.backends import cudnn
sys.path.insert(1, '/homes/bdoc3/my_utils') # only need to do this once in the main script
from sv_converter import AutoSvc
from train_params import *
from data_objects.data_loaders import load_primary_dataloader 
from utils import new_dir_setup, determine_dim_size, str2bool


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # path specifications    
    parser.add_argument("-a", "--ask", type=str2bool, default=True, help= "If False and the model name directory already exists, it will be overwritten without asking the user")
    config = parser.parse_args()
    new_dir_setup(config.ask, SVC_models_dir, SVC_model_name)

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
    SIE_feats_params, SVC_feats_params = determine_dim_size(SIE_feats_params, SVC_feats_params, SIE_feat_dir, SVC_feat_dir, use_aper_feats)
    train_dataset, train_loader = load_primary_dataloader(SIE_feats_params, 'train', SVC_feats_params, subset_size, bs=batch_size, workers=num_workers)


    # make eval loaders
    # if eval_all:
    #     val_loaders = load_val_dataloaders(SIE_feats_params, SVC_feats_params)
    # else:
    _, val_loader = load_primary_dataloader(SIE_feats_params, 'val', SVC_feats_params, subset_size, bs=batch_size, workers=num_workers)
    val_loaders = [(use_loader, val_loader)]

    # initiate model
    # pdb.set_trace()
    solver = AutoSvc(SIE_feats_params, SVC_feats_params)
    solver.train(train_loader, val_loaders)



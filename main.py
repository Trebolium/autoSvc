import os, pickle, random, argparse, yaml, sys, pdb
from torch.backends import cudnn
sys.path.insert(1, '/homes/bdoc3/my_utils')
from sv_converter import AutoSvc
from si_encoder.params_model import *
from data_objects.data_loaders import load_primary_dataloader, load_val_dataloaders
from utils import str2bool, new_dir_setup


def main(config):

    cudnn.benchmark = True # For fast training.
    random.seed(1)
    with open(os.path.join(config.SIE_feat_dir, 'feat_params.yaml')) as File:
        SIE_feats_params = yaml.load(File, Loader=yaml.FullLoader)
    if config.diff_svc_feats_dir != '':
        with open(os.path.join(config.diff_svc_feats_dir, 'feat_params.yaml')) as File:
            SVC_feats_params = yaml.load(File, Loader=yaml.FullLoader)

    "Prepare datasets"
    if config.diff_svc_feats_dir != '':
        train_dataset, train_loader = load_primary_dataloader(config, SIE_feats_params, 'train', SVC_feats_params)
    else:
        train_dataset, train_loader = load_primary_dataloader(config, SIE_feats_params, 'train')

    # config.class_layer_dim = train_dataset.num_singers
    if config.eval_all == True:
        if config.diff_svc_feats_dir != '':
            val_loaders = load_val_dataloaders(config, SIE_feats_params, SVC_feats_params)
        else:
            val_loaders = load_val_dataloaders(config, SIE_feats_params)
    else:
        if config.diff_svc_feats_dir != '':
            _, val_loader = load_primary_dataloader(config, SIE_feats_params, 'val', SVC_feats_params)
        else:
            _, val_loader = load_primary_dataloader(config, SIE_feats_params, 'val')
        val_loaders = [('damp', val_loader)]

    if config.diff_svc_feats_dir != '':
        solver = AutoSvc(train_loader, config, SIE_feats_params, SVC_feats_params)
    else:
        solver = AutoSvc(train_loader, config, SIE_feats_params)
    current_iter = solver.get_current_iters()
    log_list = []
    "training phase"
    while current_iter < config.max_iters:
        current_iter, log_list = solver.iterate('train', train_loader, current_iter, config.train_iter, log_list)
        for ds_label, val_loader in val_loaders:
            current_iter, log_list = solver.iterate(f'val_{ds_label}', val_loader, current_iter, int(config.train_iter*0.2), log_list)

    "Finish writing and save log"
    solver.closeWriter()
    with open(os.path.join(config.model_dir, config.file_name, 'log_list.pkl'), 'wb') as File:
        pickle.dump(log_list, File)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # dirs and files
    parser.add_argument('-fn', '--file_name', type=str, default='defaultName')
    parser.add_argument('-md', '--model_dir', type=str, default='/homes/bdoc3/my_data/autovc_models/autoSvc', help='path to config file to use')
    parser.add_argument('-sie_d', '--SIE_feat_dir', type=str, default='/homes/bdoc3/my_data/world_vocoder_data/damp_inton/withF0chandna_to_500_unnormed')
    parser.add_argument('-df', '--diff_svc_feats_dir', type=str, default='', help='dataloader output sequence length')
    parser.add_argument('--ckpt_weights', type=str, default='', help='path to the ckpt model want to use')
    parser.add_argument('--sie_path', type=str, default='/homes/bdoc3/my_data/autovc_models/singer_identity_encoder/dampInton500Voices_chandnaProcessing_unnormed_ge2e', help='toggle checkpoint load function')
    # model inits
    parser.add_argument('--which_cuda', type=int, default=0, help='Determine which cuda driver to use')
    parser.add_argument('--use_ckpt_config', type=str2bool, default=False, help='path to config file to use')
    parser.add_argument('--adam_init', type=float, default=0.0001, help='Define initial Adam optimizer learning rate')
    # Model param architecture
    parser.add_argument('--dim_neck', type=int, default=32)
    parser.add_argument('--dim_emb', type=int, default=256)
    parser.add_argument('--dim_pre', type=int, default=512)
    parser.add_argument('--freq', type=int, default=16)
    # dataset params
    parser.add_argument('--use_loader', type=str, default='damp', help='take singer ids to exclude from the VTEs config.test_list')
    parser.add_argument('--chunk_seconds', type=float, default=0.5, help='dataloader output sequence length')
    parser.add_argument('--chunk_num', type=int, default=6, help='dataloader output sequence length')
    parser.add_argument('--eval_all', type=str2bool, default=False, help='determines whether to evaluate main dataset or all datasets')
    # training and loss params
    parser.add_argument('--which_embs', type=str, default='sie-live', help='path to config file to use')
    parser.add_argument('-pc','--pitch_cond', type=str2bool, default=False, help='path to config file to use')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--max_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--train_size', type=int, default=20, help='Define how many speakers are used in the training set')
    parser.add_argument('--autosvc_crop', type=int, default=192, help='dataloader output sequence length')
    parser.add_argument('--psnt_loss_weight', type=float, default=1.0, help='Determine weight applied to postnet reconstruction loss')
    parser.add_argument('--prnt_loss_weight', type=float, default=1.0, help='Determine weight applied to pre-net reconstruction loss')
    # Scheduler parameters
    parser.add_argument('--patience', type=float, default=30, help='Determine weight applied to pre-net reconstruction loss')
    parser.add_argument('--ckpt_freq', type=int, default=50000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--train_iter', type=int, default=500)
    config = parser.parse_args()

    new_dir_setup(config)
    print(f'CONFIG FILE READS: {config}')
    main(config)

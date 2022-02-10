import os, pickle, random, argparse, shutil, yaml, sys, pdb
from torch.backends import cudnn
from torch.utils.data import DataLoader, SubsetRandomSampler 
from shutil import copyfile
sys.path.insert(1, '/homes/bdoc3/my_utils')
from sv_converter import AutoSvc
from si_encoder.params_model import *
from data_objects.data_loaders import VctkFromMeta, VocalSetDataset, SpecChunksFromPkl, DampDataset, DampMelWorld



def str2bool(v):
    return v.lower() in ('true')

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

"Load the primary dataloader"
def load_primary_dataloader(config, feat_params, subset_name, mel_params=None):
    if config.use_mel != '':
       dataset = DampMelWorld(config, feat_params, mel_params, os.path.join(config.feature_dir, subset_name), os.path.join(config.mel_dir, subset_name)) 
    else:
        dataset = DampDataset(config, feat_params['num_feats'], os.path.join(config.feature_dir, subset_name))
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)
    return dataset, loader

"generate dataloaders for validation"
def load_val_dataloaders(config, feat_params, mel_params=None):
    config.medley_data_path = '/homes/bdoc3/my_data/autovc_data/medleydb_singer_chunks/singer_chunks_metadata.pkl'
    config.vocal_data_path = '/homes/bdoc3/my_data/phonDet/spmel_autovc_params_normalized'
    config.vctk_data_path = '/homes/bdoc3/my_data/autovc_data/all_meta_data.pkl'
    medleydb = SpecChunksFromPkl(config, feat_params)
    vocalset = VocalSetDataset(config, feat_params)
    vctk = VctkFromMeta(config)
    
    if config.use_mel != '':
        damp = DampMelWorld(config, feat_params, mel_params, os.path.join(config.feature_dir, 'val'), os.path.join(config.mel_dir, 'val'))
    else:
        damp = DampDataset(config, feat_params['num_feats'], os.path.join(config.feature_dir, 'val'))
    
    datasets = [medleydb, vocalset, vctk, damp]
    print('Finished loading the datasets...')
    # d_idx_list = list(range(len(datasets)))
    ds_labels = ['medley', 'vocal', 'vctk', 'damp']
    val_loaders = generate_loaders(datasets, ds_labels)
    return val_loaders

"generate dataloaders from a list of datasets"
def generate_loaders(config, datasets, ds_labels):
    ds_ids_train_idxs = []
    val_loaders = []
    for i, ds in enumerate(datasets):
        random.seed(1) # reinstigating this at every iteration ensures the same random numbers are for each dataset
        current_ds_size = len(ds)
        "Take a fraction of the datasets as validation subset"
        d_idx_list = list(range(current_ds_size))
        if i != 3:
            train_song_idxs = random.sample(d_idx_list, int(current_ds_size*0.8))
            ds_ids_train_idxs.append((ds_labels[i], [x[2] for x in ds], train_song_idxs))
            val_song_idxs = [x for x in d_idx_list if x not in train_song_idxs]
            val_sampler = SubsetRandomSampler(val_song_idxs)
            val_loader = DataLoader(ds, batch_size=config.batch_size, sampler=val_sampler, shuffle=False, drop_last=True)
        else: # dataset is the one used in training (DAMP)
            val_loader = DataLoader(ds, batch_size=config.batch_size, shuffle=True, drop_last=True)
        val_loaders.append((ds_labels[i], val_loader))
    with open('dataset_ids_train_idxs.pkl','wb') as File:
        pickle.dump(ds_ids_train_idxs, File) # save dataset ids as pkl for potential hindsight analysis
    return val_loaders 

def main(config):

    cudnn.benchmark = True # For fast training.
    random.seed(1)
    with open(os.path.join(config.feature_dir, 'feat_params.yaml')) as File:
        feat_params = yaml.load(File, Loader=yaml.FullLoader)
    if config.use_mel != '':
        with open(os.path.join(config.use_mel, 'feat_params.yaml')) as File:
            mel_params = yaml.load(File, Loader=yaml.FullLoader)

    "Prepare datasets"
    if config.use_mel != '':
        train_dataset, train_loader = load_primary_dataloader(config, feat_params, 'train', mel_params)
    else:
        train_dataset, train_loader = load_primary_dataloader(config, feat_params, 'train')

    # config.class_layer_dim = train_dataset.num_singers
    if config.eval_all == True:
        if config.use_mel != '':
            val_loaders = load_val_dataloaders(config, feat_params, mel_params)
        else:
            val_loaders = load_val_dataloaders(config, feat_params)
    else:
        if config.use_mel != '':
            _, val_loader = load_primary_dataloader(config, feat_params, 'val', mel_params)
        else:
            _, val_loader = load_primary_dataloader(config, feat_params, 'val')
        val_loaders = [('damp', val_loader)]

    if config.use_mel != '':
        solver = AutoSvc(train_loader, config, feat_params, mel_params)
    else:
        solver = AutoSvc(train_loader, config, feat_params)
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
    parser.add_argument('--file_name', '-fn', type=str, default='defaultName')
    parser.add_argument('--model_dir', type=str, default='/homes/bdoc3/my_data/autovc_data/autoSvc', help='path to config file to use')
    parser.add_argument('--feature_dir', '-fd', type=str, default='/homes/bdoc3/my_data/world_vocoder_data/damp_inton/withF0chandna_to_500_unnormed')
    parser.add_argument('--ckpt_weights', type=str, default='', help='path to the ckpt model want to use')
    parser.add_argument('--sie_path', type=str, default='/homes/bdoc3/singer-identity-encoder/encoder/saved_models/dampInton500Voices_chandnaProcessing_unnormed_ge2e', help='toggle checkpoint load function')
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
    parser.add_argument('--use_loader', type=str, default='vocal', help='take singer ids to exclude from the VTEs config.test_list')
    parser.add_argument('--use_mel', type=str, default='', help='dataloader output sequence length')
    parser.add_argument('--chunk_seconds', type=float, default=0.5, help='dataloader output sequence length')
    parser.add_argument('--chunk_num', type=int, default=6, help='dataloader output sequence length')
    parser.add_argument('--eval_all', type=str2bool, default=False, help='determines whether to evaluate main dataset or all datasets')
    # training and loss params
    parser.add_argument('--which_embs', type=str, default='sie-live', help='path to config file to use')
    parser.add_argument('-pc','--pitch_cond', type=str2bool, default=False, help='path to config file to use')
    parser.add_argument('--batch_size', type=int, default=2, help='mini-batch size')
    parser.add_argument('--max_iters', type=int, default=1000000, help='number of total iterations')
    parser.add_argument('--train_size', type=int, default=20, help='Define how many speakers are used in the training set')
    parser.add_argument('--len_crop', type=int, default=160, help='dataloader output sequence length')
    parser.add_argument('--psnt_loss_weight', type=float, default=1.0, help='Determine weight applied to postnet reconstruction loss')
    parser.add_argument('--prnt_loss_weight', type=float, default=1.0, help='Determine weight applied to pre-net reconstruction loss')
    # Scheduler parameters
    parser.add_argument('--patience', type=float, default=501, help='Determine weight applied to pre-net reconstruction loss')
    parser.add_argument('--ckpt_freq', type=int, default=50000, help='frequency in steps to mark checkpoints')
    parser.add_argument('--spec_freq', type=int, default=10000, help='frequency in steps to print reconstruction illustrations')
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--train_iter', type=int, default=500)
    config = parser.parse_args()

    new_dir_setup(config)
    print(f'CONFIG FILE READS: {config}')
    main(config)

import numpy as np
import os, pickle, random, math, pdb, sys
from multiprocessing import Process, Manager
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from train_params import *
from my_os import recursive_file_retrieval
from my_audio.pitch import midi_as_onehot
from my_audio.midi import freqToMidi
from my_arrays import fix_feat_length
from my_normalise import norm_feat_arr, get_norm_stats

"""
    Retrieves features from 2 directories.
    Assumes that filename is divided by the _ character into singerID and uttrsID (only relevant for troubleshooting later if necessary)
    Assumes both feature sets are the same hop-size (durations per frame)
    Dataset indexed by singerIDs. Each dataset entry contains a list of features related to one singer (the number of uttrs per singer varies)
    Datsset entries are tuples that include features (array), singerID (str) and uttrsID (str)
"""
class DuoFeatureDataset(Dataset):

    def __init__(self, feats1_num_feats,
                       feats2_num_feats,
                       subset_name, ds_size,
                       this_train_params=None, chosen_class_num=None): 

        if this_train_params==None:
            self.SIE_feat_dir = SIE_feat_dir
            self.SVC_feat_dir = SVC_feat_dir 
            self.window_timesteps = window_timesteps 
            self.norm_method = norm_method
            self.pitch_dir = pitch_dir
            self.SVC_pitch_cond = SVC_pitch_cond 
            self.SIE_pitch_cond = SIE_pitch_cond 
            self.midi_range = midi_range
        else:
            self.SIE_feat_dir = this_train_params['SIE_feat_dir']
            self.SVC_feat_dir = this_train_params['SVC_feat_dir']
            self.window_timesteps = this_train_params['window_timesteps']
            self.norm_method = this_train_params['norm_method']
            self.pitch_dir = this_train_params['pitch_dir']
            self.SVC_pitch_cond = this_train_params['SVC_pitch_cond']
            self.SIE_pitch_cond = this_train_params['SIE_pitch_cond']
            self.midi_range = this_train_params['midi_range']

        self.ext = '.npy'

        # FIXME: Much better mechanism to use than dataset percentage option as this is truly adaptable, random, doesn't require making more dirs
        if chosen_class_num != None:
            feats1_subset_dir = os.path.join(self.SIE_feat_dir, subset_name)
            # pdb.set_trace()
            dir_paths, feats1_fps = recursive_file_retrieval(feats1_subset_dir, ignore_hidden_dirs=True, return_parent=False)
            subdirs = random.sample([os.path.basename(dir_path) for dir_path in dir_paths], chosen_class_num)
            feats1_fps = [fp for fp in feats1_fps if os.path.basename(fp).split('_')[0] in subdirs]

        else:
            if ds_size == 1.0:
                feats1_subset_dir = os.path.abspath(os.path.join(self.SIE_feat_dir, subset_name))
                _, feats1_fps = recursive_file_retrieval(feats1_subset_dir, ignore_hidden_dirs=True)
            else:
                feats1_subset_dir = os.path.abspath(os.path.join(self.SIE_feat_dir, subset_name, f'.{ds_size}_size'))
                if not os.path.exists(feats1_subset_dir):
                    pdb.set_trace()
                    raise Exception
                _, feats1_fps = recursive_file_retrieval(feats1_subset_dir)

        numpy_fns = [os.path.basename(fp) for fp in sorted(feats1_fps) if fp.endswith(self.ext) and not fp.startswith('.')] 
        
        if self.norm_method == 'schluter':
            self.f1_total_mean, self.f1_total_std = get_norm_stats(os.path.join(self.SIE_feat_dir +'train'))
            self.f2_total_mean, self.f2_total_std = get_norm_stats(os.path.join(self.SVC_feat_dir +'train'))

        num_songs = 0
        singer_clips = {}
        for fn in numpy_fns:
            # get path components
            singer_id = fn.split('_')[0]

            if singer_id not in singer_clips.keys():
                singer_clips[singer_id] = [fn]
            else:
                singer_clips[singer_id].append(fn)

            num_songs += 1
        self.dataset = [content for content in singer_clips.values()]
        self.num_songs = num_songs
        self.feats1_num_feats = feats1_num_feats
        self.feats2_num_feats = feats2_num_feats
        self.subset_name = subset_name


    def __getitem__(self, index):
        # pick a random speaker
        fns = self.dataset[index]
        fn = random.choice(fns)
        singer_id = fn.split('_')[0]

        # generate feats2 corresponding path for filename
        # if self.subset_name == 'val':
        #     pdb.set_trace()
        # if tiny_run:
        #     path_from_subset = os.path.join(self.subset_name, 'tiny_run', singer_id, fn)
        # else:
        path_from_subset = os.path.join(self.subset_name, singer_id, fn)
        feats1_fp = os.path.join(self.SIE_feat_dir, path_from_subset)
        feats2_fp = os.path.join(self.SVC_feat_dir, path_from_subset)
        # generate feats
        feats1_features = np.load(feats1_fp)
        feats2_features = np.load(feats2_fp)

        # chooses a random uttr here
        spec1, spec2 = feats1_features[:,:self.feats1_num_feats], feats2_features[:,:self.feats2_num_feats]
        # crop feats
        spec1_trimmed, start_step = fix_feat_length(spec1, self.window_timesteps)
        spec2_trimmed, _ = fix_feat_length(spec2, self.window_timesteps, start_step)

        if self.norm_method == 'schluter':
            spec1_normmed = norm_feat_arr(spec1_trimmed, self.norm_method, (self.f1_total_mean, self.f1_total_std))
            spec2_normmed = norm_feat_arr(spec2_trimmed, self.norm_method, (self.f2_total_mean, self.f2_total_std))
        else:
            spec1_normmed = norm_feat_arr(spec1_trimmed, self.norm_method)
            spec2_normmed = norm_feat_arr(spec2_trimmed, self.norm_method)

        # feats2_spec = feats2_spec[:,start_step:(start_step+window_timesteps)]
        # print(spec1_normmed.shape, spec2_normmed.shape, f'offset {start_step}/{spec1.shape[0]}')
        if self.SVC_pitch_cond or self.SIE_pitch_cond:
            
            if 'crepe_data' in self.pitch_dir:
                pitch_fn = fn[:-1]+'z'
            else:
                pitch_fn = fn

            target_file = os.path.join(self.pitch_dir, 'train', singer_id, pitch_fn)
            # find corresponding file from pitch dir and return pitch_predictions
            if not os.path.exists(target_file):
                target_file = os.path.join(self.pitch_dir, 'val', singer_id, pitch_fn)

            if 'crepe_data' in self.pitch_dir:
                crepe_data = np.load(target_file)
                pitches = crepe_data['arr_0']
                conf = crepe_data['arr_1']
                unvoiced = conf < 0.5 #determined by looking at pitch and conf contours against audio in sonic visualizer
                midi_contour = freqToMidi(pitches)
            else:
                pitch_pred = np.load(target_file)[:,-2:]
                midi_contour = pitch_pred[:,0]
                unvoiced = pitch_pred[:,1].astype(int) == 1 # remove the interpretted values generated because of unvoiced sections
            midi_contour[unvoiced] = 0
            
            try:
                if start_step < 0:
                    midi_trimmed, _ = fix_feat_length(midi_contour, self.window_timesteps)
                else:
                    midi_trimmed = midi_contour[start_step:(start_step+self.window_timesteps)]
                onehot_midi = midi_as_onehot(midi_trimmed, self.midi_range)
            except Exception as e:
                print(f'Exception {e} caused by file {fn}')
                pdb.set_trace()
        
        else:
            onehot_midi = np.zeros((self.window_timesteps, len(self.midi_range)+1))

        return spec1_normmed, spec2_normmed, onehot_midi, fn, start_step, index


    def __len__(self):
        """Return the number of spkrs."""
        return len(self.dataset)


"""
    Retrieves features from directory.
    Assumes that filename is divided by the _ character into singerID and uttrsID (only relevant for troubleshooting later if necessary)
    Dataset indexed by singerIDs. Each dataset entry contains a list of features related to one singer (the number of uttrs per singer varies)
    Datsset entries are tuples that include features (array), singerID (str) and uttrsID (str)
"""
# class SingleFeatureDataset(Dataset):

#     def __init__(self, num_feats, feat_dir):
        
#         ext = '.npy'
#         _, fps = recursive_file_retrieval(feat_dir) # explicitly given outside config to specify whether train or val subset used here
#         cleaned_fps = [fp for fp in sorted(fps) if fp.endswith(ext) and not fp.startswith('.')]
#         num_songs = 0
#         singer_clips = {}
#         for file_path in cleaned_fps:
#             singer_id, uttrs_id = os.path.basename(file_path).split('_')[0], os.path.basename(file_path).split('_')[1][:len(ext)]
#             features = np.load(file_path)
#             num_songs += 1
#             if singer_id not in singer_clips.keys():
#                 singer_clips[singer_id] = [(features, singer_id, uttrs_id)]
#             else:
#                 singer_clips[singer_id].append((features, singer_id, uttrs_id))
#         # this compression is necessary for instances when some singers have multiple entries and others do not"
#         self.dataset = [content for content in singer_clips.values()]
#         self.num_songs = num_songs
#         self.num_feats = num_feats

#     def __getitem__(self, index):

#         voice_meta = self.dataset[index]
#         # chooses a random uttr here
#         feats, singer_id, example_id = voice_meta[random.randint(0,len(voice_meta)-1)]
#         # crop feats
#         if SVC_pitch_cond:
#             feats_spec, feats_pitch = process_uttrs_feats(feats, self.num_feats)
#             return feats_spec, feats_pitch, (singer_id, example_id)
#         else:
#             feats_spec = process_uttrs_feats(feats, self.num_feats)
#             return feats_spec, (singer_id, example_id)

#     def __len__(self):
#         """Return the number of spkrs."""
#         return len(self.dataset)


"Load the primary dataloader"
def load_primary_dataloader(SIE_feats_params, subset_name, SVC_feats_params, ds_size, chosen_class_num, bs=None, workers=None, this_train_params=None):

    # pdb.set_trace()
    if bs == None: batch_size = batch_size
    else: batch_size = bs
    if workers == None: num_workers = num_workers
    else: num_workers = workers

    if this_train_params == None:
        dataset = DuoFeatureDataset(SIE_feats_params['num_feats'], SVC_feats_params['num_feats'], subset_name, ds_size, chosen_class_num)
    else:
        dataset = DuoFeatureDataset(SIE_feats_params['num_feats'], SVC_feats_params['num_feats'], subset_name, ds_size, this_train_params, chosen_class_num)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    return dataset, loader


# "generate dataloaders for validation"
# def load_val_dataloaders(SIE_feats_params, SVC_feats_params):
#     # needs to be fixed - do we really need to use multiple different dataset objects? I doubt it!
#     medleydb = SpecChunksFromPkl(SIE_feats_params)
#     vocalset = VocalSetDataset(SIE_feats_params)
#     vctk = VctkFromMeta()
#     damp = SingleFeatureDataset(SIE_feats_params['num_feats'], os.path.join(SIE_feat_dir, 'val'))
    
#     datasets = [medleydb, vocalset, vctk, damp]
#     print('Finished loading the datasets...')
#     # d_idx_list = list(range(len(datasets)))
#     ds_labels = ['medley', 'vocal', 'vctk', 'damp']
#     val_loaders = generate_loaders(datasets, ds_labels)
#     return val_loaders


"generate dataloaders from a list of datasets"
def generate_loaders(datasets, ds_labels):
    ds_ids_train_idxs = []
    val_loaders = []
    for i, ds in enumerate(datasets):
        random.seed(1) # reinstigating this at every iteration ensures the same random numbers are for each dataset
        current_ds_size = len(ds)
        "Take a fraction of the datasets as validation subset"
        d_idx_list = list(range(current_ds_size))
        if i != 3:
            train_uttrs_idxs = random.sample(d_idx_list, int(current_ds_size*0.8))
            ds_ids_train_idxs.append((ds_labels[i], [x[2] for x in ds], train_uttrs_idxs))
            val_uttrs_idxs = [x for x in d_idx_list if x not in train_uttrs_idxs]
            val_sampler = SubsetRandomSampler(val_uttrs_idxs)
            val_loader = DataLoader(ds, batch_size=batch_size, sampler=val_sampler, shuffle=False, drop_last=True)
        else: # dataset is the one used in training (DAMP)
            val_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
        val_loaders.append((ds_labels[i], val_loader))
    with open('dataset_ids_train_idxs.pkl','wb') as File:
        pickle.dump(ds_ids_train_idxs, File) # save dataset ids as pkl for potential hindsight analysis
    return val_loaders

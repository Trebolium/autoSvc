import numpy as np
import os, pickle, random, math, pdb, sys
from multiprocessing import Process, Manager
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from train_params import *
from my_os import recursive_file_retrieval
from my_audio.pitch import midi_as_onehot
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
                       this_train_params=None): 

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
        if ds_size == 1.0:
            feats1_subset_dir = os.path.join(self.SIE_feat_dir, subset_name)
            _, feats1_fps = recursive_file_retrieval(feats1_subset_dir, ignore_hidden_dirs=True)
            numpy_fns = [os.path.basename(fp) for fp in sorted(feats1_fps) if fp.endswith(self.ext) and not fp.startswith('.')]
        else:
            feats1_subset_dir = os.path.join(self.SIE_feat_dir, subset_name, f'.{ds_size}_size')
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
            
            target_file = os.path.join(self.pitch_dir, 'train', singer_id, fn)
            # find corresponding file from pitch dir and return pitch_predictions
            if os.path.exists(target_file):
                pitch_pred = np.load(target_file)[:,-2:]
            else:
                target_file = os.path.join(self.pitch_dir, 'val', singer_id, fn)
                if os.path.exists(target_file):
                    pitch_pred = np.load(target_file)[:,-2:]
                else:
                    raise FileNotFoundError(f'Target file {fn} could not be found in pitch directory {self.pitch_dir}')
            
            midi_contour = pitch_pred[:,0]
            # remove the interpretted values generated because of unvoiced sections
            unvoiced = pitch_pred[:,1].astype(int) == 1
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
def load_primary_dataloader(SIE_feats_params, subset_name, SVC_feats_params, ds_size, bs=None, workers=None, this_train_params=None):

    # pdb.set_trace()
    if bs == None: batch_size = batch_size
    else: batch_size = bs
    if workers == None: num_workers = num_workers
    else: num_workers = workers

    if this_train_params == None:
        dataset = DuoFeatureDataset(SIE_feats_params['num_feats'], SVC_feats_params['num_feats'], subset_name, ds_size)
    else:
        dataset = DuoFeatureDataset(SIE_feats_params['num_feats'], SVC_feats_params['num_feats'], subset_name, ds_size, this_train_params)

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


# class SpecChunksFromPkl(Dataset):
#     """Dataset class for using a pickle object,
#     pickle object second entry (index[1]) is list of spec arrays,
#     generates random windowed subspec examples,
#     associated labels,
#     optional conditioning."""
#     # made originally for medleydb pkl
#     def __init__(self, SIE_feat_params):
#         """Initialize and preprocess the dataset."""
        
#         melsteps_per_second = SIE_feat_params['sr'] / SIE_feat_params['hop_size']
#         self.window_size = math.ceil(chunk_seconds * melsteps_per_second) * chunk_num
#         metadata = pickle.load(open(medley_data_path, 'rb'))
#         dataset = []
#         song_counter = 0
#         previous_filename = metadata[0][0][:-10]
#         list_by_track = []
#         for entry in metadata:
#             file_name = entry[0]
#             # if song path from metadata has changed, update dataset, start empty song list
#             if file_name[:-10] != previous_filename:
#                 dataset.append(list_by_track)
#                 previous_filename = file_name[:-10]
#                 list_by_track = []
#                 song_counter += 1
#             spmel_chunks = entry[2]
#             chunk_counter = 0
#             list_by_mel_chunks = []
#             for spmel_chunk in spmel_chunks:
#                 list_by_mel_chunks.append((spmel_chunk, song_counter, chunk_counter, file_name))
#                 chunk_counter += 1
#             list_by_track.append(list_by_mel_chunks)
#         dataset.append(list_by_track)

#         self.dataset = dataset
#         self.num_specs = len(dataset)

#     def __getitem__(self, index):
#         # pick a random speaker
#         dataset = self.dataset
#         # index specifies 
#         track_list = dataset[index]
#         spmel_chunk_list = track_list[random.randint(0,len(track_list)-1)]
#         spmel, dataset_idx, chunk_counter, file_name = spmel_chunk_list[random.randint(0,len(spmel_chunk_list)-1)]
#         # pick random spmel_chunk with random crop
#         """Ensure all spmels are the length of (self.window_size * chunk_num)"""
#         if spmel.shape[0] >= self.window_size:
#             difference = spmel.shape[0] - self.window_size
#             offset = random.randint(0, difference)
#         else: adjusted_length_spmel = spmel
#         adjusted_length_spmel = spmel[offset : offset + self.window_size]
#         # may need to set chunk_num to constant value so that all tensor sizes are of known shape for the LSTM
#         # a constant will also mean it is easier to group off to be part of the same recording
#         # the smallest is 301 frames. If the window sizes are 44, then that 6 full windows each
#         return adjusted_length_spmel, dataset_idx, os.path.basename(file_name[:-4])

#     def __len__(self):
#         """Return the number of spkrs."""
#         return self.num_specs

# class VocalSetDataset(Dataset):
#     """Dataset class for using a path to spec folders,
#     path for labels,
#     generates random windowed subspec examples,
#     associated labels,
#     optional conditioning."""
#     def __init__(self, SIE_feat_params):
#         """Initialize and preprocess the dataset."""
        
#         melsteps_per_second = SIE_feat_params['sr'] / SIE_feat_params['hop_size']
#         self.window_size = math.ceil(chunk_seconds * melsteps_per_second) * chunk_num
#         style_names = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
#         singer_names = ['m1_','m2_','m3_','m4_','m5_','m6_','m7_','m8_','m9_','m10_','m11_','f1_','f2_','f3_','f4_','f5_','f6_','f7_','f8_','f9_']

#         dir_name, _, fileList = next(os.walk(vocalset_data_path)) #this has changed from unnormalised to normed
#         fileList = sorted(fileList)
#         dataset = []
#         # group dataset by singers
#         for singer_idx, singer_name in enumerate(singer_names):
#             singer_examples = []
#             for file_name in fileList:
#                 if file_name.startswith(singer_name) and file_name.endswith('.npy'):
#                     spmel = np.load(os.path.join(dir_name, file_name))
#                     for style_idx, style_name in enumerate(style_names):
#                         if style_name in file_name:
#                             singer_examples.append((spmel, singer_idx, os.path.basename(file_name[:-4])))
#                             break #if stle found, break stype loop
#             dataset.append(singer_examples)
#         self.dataset = dataset
#         self.num_specs = len(dataset)
        
#     """__getitem__ selects a speaker and chooses a random subset of data (in this case
#     an utterance) and randomly crops that data. It also selects the corresponding speaker
#     embedding and loads that up. It will now also get corresponding pitch contour for such a file"""

#     def __getitem__(self, index):
#         # pick a random speaker
#         dataset = self.dataset
#         # spkr_data is literally a list of skpr_id, emb, and utterances from a single speaker
#         utters_meta = dataset[index]
#         spmel, dataset_idx, example_id = utters_meta[random.randint(0,len(utters_meta)-1)]
#         # pick random spmel_chunk with random crop
#         """Ensure all spmels are the length of (self.window_size * chunk_num)"""
#         if spmel.shape[0] >= self.window_size:
#             difference = spmel.shape[0] - self.window_size
#             offset = random.randint(0, difference)
#         adjusted_length_spmel = spmel[offset : offset + self.window_size]
#         # may need to set chunk_num to constant value so that all tensor sizes are of known shape for the LSTM
#         # a constant will also mean it is easier to group off to be part of the same recording
#         # the smallest is 301 frames. If the window sizes are 44, then that 6 full windows each
#         return adjusted_length_spmel, dataset_idx, example_id

#     def __len__(self):
#         """Return the number of spkrs."""
#         return self.num_specs


# class VctkFromMeta(Dataset):
#     """Dataset class for the Utterances dataset."""

#     # this object will contain both melspecs and speaker embeddings taken from the train.pkl
#     def __init__(self):
#         """Initialize and preprocess the Utterances dataset."""
        
#         self.autosvc_crop = window_timesteps
#         self.step = 10
#         self.file_name = svc_model_name

#         meta_all_data = pickle.load(open(vctk_data_path, "rb"))
#         # split into training data
#         num_training_speakers=train_sizes
#         random.seed(1)
#         training_indices =  random.sample(range(0, len(meta_all_data)), num_training_speakers)
#         training_set = []

#         meta_training_speaker_all_uttrs = []
#         # make list of training speakers
#         for idx in training_indices:
#             meta_training_speaker_all_uttrs.append(meta_all_data[idx])
#         # get training files
#         for speaker_info in meta_training_speaker_all_uttrs:
#             speaker_id_emb = speaker_info[:2]
#             speaker_uttrs = speaker_info[2:]
#             num_files = len(speaker_uttrs) # first 2 entries are speaker ID and speaker_emb)
#             training_file_num = round(num_files*0.9)
#             training_file_indices = random.sample(range(0, num_files), training_file_num)

#             training_file_names = []
#             for index in training_file_indices:
#                 fileName = speaker_uttrs[index]
#                 training_file_names.append(fileName)
#             training_set.append(speaker_id_emb+training_file_names)
#             # training_file_names_array = np.asarray(training_file_names)
#             # training_file_indices_array = np.asarray(training_file_indices)
#             # test_file_indices = np.setdiff1d(np.arange(num_files_in_subdir), training_file_indices_array)
#         # training set contains
#         training_metadata_path = os.path.join(svc_model_dir, svc_model_name,'training_meta_data.pkl')
#         with open(training_metadata_path, 'wb') as train_pack:
#             pickle.dump(training_set, train_pack)

#         training_info = pickle.load(open(training_metadata_path, 'rb'))
#         # self.one_hot_array = np.eye(len(training_info))[num_speakers_seq]
#         self.spkr_id_list = [spkr[0] for spkr in training_info]

#         """Load data using multiprocessing"""
#         manager = Manager()
#         meta = manager.list(training_set)
#         dataset = manager.list(len(meta)*[None])  
#         processes = []
#         # uses a different process thread for every self.steps of the meta content
#         for i in range(0, len(meta), self.step):
#             p = Process(target=self.load_data, 
#                         args=(meta[i:i+self.step],dataset,i))  
#             p.start()
#             processes.append(p)
#         for p in processes:
#             p.join()
        
#         self.train_dataset = list(dataset)
#         self.num_tokens = len(self.train_dataset)
        
#     # this function is called within the class init (after self.data_loader its the arguments) 
#     def load_data(self, submeta, dataset, idx_offset):  
#         for k, sbmt in enumerate(submeta):    
#             uttrs = len(sbmt)*[None]
#             for j, tmp in enumerate(sbmt):
#                 if j < 2:  # fill in speaker id and embedding
#                     uttrs[j] = tmp
#                 else: # load the mel-spectrograms
#                     uttrs[j] = np.load(os.path.join('/homes/bdoc3/my_data/spmel', tmp))
#             dataset[idx_offset+k] = uttrs
                   
#     """__getitem__ selects a speaker and chooses a random subset of data (in this case
#     an utterance) and randomly crops that data. It also selects the corresponding speaker
#     embedding and loads that up. It will now also get corresponding pitch contour for such a file""" 
#     def __getitem__(self, index):
#         # pick a random speaker
#         dataset = self.train_dataset 
#         # list_uttrs is literally a list of utterance from a single speaker
#         list_uttrs = dataset[index]
#         emb_org = list_uttrs[1]
#         speaker_name = list_uttrs[0]
#         # pick random uttr with random crop
#         a = np.random.randint(2, len(list_uttrs))
#         uttr_info = list_uttrs[a]
        
#         spmel_tmp = uttr_info
#         #spmel_tmp = uttr_info[0]
#         #pitch_tmp = uttr_info[1]
#         if spmel_tmp.shape[0] < self.autosvc_crop:
#             len_pad = self.autosvc_crop - spmel_tmp.shape[0]
#             uttr = np.pad(spmel_tmp, ((0,len_pad),(0,0)), 'constant')
#         #    pitch = np.pad(pitch_tmp, ((0,len_pad),(0,0)), 'constant')
#         elif spmel_tmp.shape[0] > self.autosvc_crop:
#             left = np.random.randint(spmel_tmp.shape[0]-self.autosvc_crop)
#             uttr = spmel_tmp[left:left+self.autosvc_crop, :]
#         #    pitch = pitch_tmp[left:left+self.autosvc_crop, :]
#         else:
#             uttr = spmel_tmp
#         #    pitch = pitch_tmp    

#         # find out where speaker is in the order of the training list for one-hot
#         for i, spkr_id in enumerate(self.spkr_id_list):
#             if speaker_name == spkr_id:
#                 spkr_label = i
#                 break
#         # one_hot_spkr_label = self.one_hot_array[spkr_label]
#         # if self.one_hot==False:
#         return uttr, index, speaker_name

# # writing this line as an excuse to update git message
#     def __len__(self):
#         """Return the number of spkrs."""
#         return self.num_tokens

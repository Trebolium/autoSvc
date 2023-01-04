import os, shutil, yaml, torch, pickle, pdb, csv, random
from shutil import copyfile
from model_sie import SingerIdEncoder
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
from model_vc import Generator
import soundfile as sf


# from convert.synthesis import build_model
# from convert.synthesis import wavegen

from my_audio.world import mfsc_to_world_to_audio
from my_audio.pitch import midi_as_onehot
from neural.model_mod import checkpoint_model_optim_keys
from my_os import overwrite_dir, recursive_file_retrieval
from my_arrays import fix_feat_length, container_to_tensor, tensor_to_array, find_runs





"""Convert world pitch info to 1hot midi"""
def cont_to_onehot_midi(midi_voicing, midi_range):

    midi_contour = midi_voicing[:,0]
    unvoiced = midi_voicing[:,1].astype(int) == 1 # remove the interpretted values generated because of unvoiced sections
    midi_contour[unvoiced] = 0
    onehot_midi = midi_as_onehot(midi_contour, midi_range)
    
    return onehot_midi


# """Set up for multi-threading"""
# def synthesize_audio(iterables):
#     converted_feat = iterables[0]
#     config = iterables[1]
#     feat_params = iterables[2]
#     model = iterables[3]
#     wav_dir = iterables[4]
#     converted_id, x_identic, aper_src, onehot_midi_src = converted_feat
#     print(f'Processing {converted_id}...')
#     if config.use_model.startswith('world'):
#         feat_params['fft_size'] = 1024
#         print('synthesizing with world vocoder now...')
#         waveform = mfsc_to_world_to_audio(x_identic, aper_src, onehot_midi_src, feat_params)
#     elif config.use_model.startswith('mel'):
#         print('synthesizing with wavenet vocoder now...')
#         waveform = wavegen(model, config.which_cuda, c=x_identic)
#     sf.write(os.path.join(wav_dir, converted_id +'.wav'), waveform, samplerate=feat_params['sr'])


"""Currently designed to take model ckpts of 2 slightly different dictionary keys"""
def setup_sie(device, loss_device, SIE_path, adam_init):
    sie_checkpoint = torch.load(os.path.join(SIE_path, 'saved_model.pt'), map_location='cpu')
    new_state_dict = OrderedDict()

    if SIE_path.endswith('autoVc_pretrainedOnVctk_Mels80'):
        model_state = 'model_b'
        sie_num_feats_used = sie_checkpoint[model_state]['module.lstm.weight_ih_l0'].shape[1]
    else:
        model_state = 'model_state'
        sie_num_feats_used = sie_checkpoint[model_state]['lstm.weight_ih_l0'].shape[1]
    sie = SingerIdEncoder(device, loss_device, sie_num_feats_used)

    if SIE_path.endswith('autoVc_pretrainedOnVctk_Mels80'):
        new_state_dict['similarity_weight'] = sie.similarity_weight
        new_state_dict['similarity_bias'] = sie.similarity_bias
    
    for (key, val) in sie_checkpoint[model_state].items():

        if SIE_path.endswith('autoVc_pretrainedOnVctk_Mels80'):
            key = key[7:] # gets right of the substring 'module'
            if key.startswith('embedding'):
                key = 'linear.' +key[10:]

        new_state_dict[key] = val
    
    sie.load_state_dict(new_state_dict)

    for param in sie.parameters():
        param.requires_grad = False
    sie_optimizer = torch.optim.Adam(sie.parameters(), adam_init)

    for state in sie_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(device)
    sie.to(device)
    sie.eval()

    return sie, sie_num_feats_used


"""Currently designed to initiate G in two ways, based on train_param variables"""
def setup_gen(dim_neck, dim_emb, dim_pre, sample_freq, num_feats, pitch_dim, device, svc_ckpt_path, adam_init):

    G = Generator(dim_neck, dim_emb, dim_pre, sample_freq, num_feats, pitch_dim)
    g_optimizer = torch.optim.Adam(G.parameters(), adam_init)

    if svc_ckpt_path!='':
        g_checkpoint = torch.load(svc_ckpt_path, map_location='cpu')
        model_key, optim_key = checkpoint_model_optim_keys(g_checkpoint)
        for k in g_checkpoint.keys():
            if k.startswith('model'):
                model_key = k
            if k.startswith('optim'):
                optim_key = k
        # pdb.set_trace() 
        G.load_state_dict(g_checkpoint[model_key])
        g_optimizer.load_state_dict(g_checkpoint[optim_key])

        # fixes tensors on different devices error
        # https://github.com/pytorch/pytorch/issues/2830
        for state in g_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        try:
            train_latest_step = g_checkpoint['step']
        except KeyError as e:
            train_latest_step = 0
    else:
        train_latest_step = 0

    G.to(device)
    return G, g_optimizer, train_latest_step


# this function seems like a hack - find out the standard method for passing boolean values as parser args
def str2bool(v):
    return v.lower() in ('true')


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
def new_dir_setup(ask, svc_model_dir, svc_model_name):
    model_dir_path = os.path.join(svc_model_dir, svc_model_name)
    overwrite_dir(model_dir_path, ask)
    os.makedirs(model_dir_path +'/ckpts')
    os.makedirs(model_dir_path +'/generated_wavs')
    os.makedirs(model_dir_path +'/image_comparison')
    os.makedirs(model_dir_path +'/input_tensor_plots')
    files = ['model_vc.py', 'sv_converter.py', 'main.py', 'utils.py', 'train_params.py']

    for file in files:
        dst_file = os.path.join(model_dir_path, 'this_' + file)
        copyfile(file, dst_file)
        with open(dst_file, 'r') as file:
            filedata = file.read()
        filedata = filedata.replace('from train_params import *', 'from this_train_params import *')
        with open(dst_file, 'w') as file:
            file.write(filedata)


# "Process config object, reassigns values if necessary, raise exceptions"
# def process_config():
#     if svc_model_name == svc_ckpt_path:
#         raise Exception("Your file name and svc_ckpt_path name can't be the same")
#     if not ckpt_freq%int(max_cycle_iters*0.2) == 0 or not ckpt_freq%int(max_cycle_iters*0.2) == 0:
#         raise Exception(f"ckpt_freq {ckpt_freq} and spec_freq {spec_freq} need to be a multiple of val_iter {int(max_cycle_iters*0.2)}")


# check use_aper_feats boolean to produce total num feats being used for training
# this is ignored in the case of mels as they don't have aper aspect
def determine_dim_size(SIE_params, SVC_params, SIE_feat_dir, SVC_feat_dir, use_aper_feats):

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

    if 'mel' in SIE_feat_dir:
        SIE_params['num_feats'] = SIE_params['num_harm_feats']
    if 'mel' in SVC_feat_dir:
        SVC_params['num_feats'] = SVC_params['num_harm_feats']

    return SIE_params, SVC_params


def vctk_id_gender_list(csv_path='/homes/bdoc3/my_data/text_data/vctk/speaker-info.txt'):   
    f = open(csv_path, 'r')
    header = f.readline()
    lines = f.readlines()
    id_list = []
    gender_list = []
    for line in lines:
        line_elements = [el for el in line.split(' ') if el!='']
        id_list.append(line_elements[0])
        gender_list.append(line_elements[2])
    return id_list, gender_list

def get_vocalset_gender_techs():

    singing_techniques = ['belt','lip_trill','straight','vocal_fry','vibrato','breathy']
    gender_group_labels_arr = []
    technique_group_labels_arr = []
    for voice_meta in metad_by_singer_list:
        voice_fns = substring_inclusion(voice_meta[2:], singing_techniques)
        print(len(voice_fns))

        for fn in voice_fns:

            # count += 1

            if fn.startswith('m'):
                gender_group_labels_arr.append('male')
            elif fn.startswith('f'):
                gender_group_labels_arr.append('female')
            
            st_found = False
            for st_i, st in enumerate(singing_techniques):
                if st in fn:
                    technique_group_labels_arr.append(st_i)
                    st_found = True
            if not st_found:
                pdb.set_trace()
                raise Exception('St not found')
                
    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    all_labels_class_sizes.append(2)

    technique_group_labels_arr = np.asarray(technique_group_labels_arr)
    all_labels_arrs.append(technique_group_labels_arr)
    all_label_names.append('singing_technique')
    all_labels_class_sizes.append(config.max_num_techs)
    return gender_group_labels_arr, technique_group_labels_arr


def get_vocadito_gender():
    gender_group_labels_arr = []
    csv_path = '/homes/bdoc3/my_data/text_data/vocadito/vocadito_metadata.csv'
    f = open(csv_path, 'r')
    reader = csv.reader(f)
    header = next(reader)
    singer_meta = [row for row in reader]
    perf_key_meta_list = [row[0] for row in singer_meta]
    gender_meta_list = [row[4] for row in singer_meta]

    for voice_meta in metad_by_singer_list:

        uttrs_fps = voice_meta[2:]
        for fp in uttrs_fps:
            track_name = os.path.basename(fp)[:-4]
            track_int = track_name.split('_')[1]
            try:

                idx = perf_key_meta_list.index(track_int)
            except ValueError as e:
                print(e)
                continue

            gender = gender_meta_list[idx]
            if 'm' in gender.lower():
                gender_group_labels_arr.append('male')
            elif 'f' in gender.lower():
                gender_group_labels_arr.append('female')
            else:
                raise Exception(f'Gender value not recognised for excerpt {track_name} in csv row {idx}')

    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    all_labels_class_sizes.append(1)
    return gender_group_labels_arr


def get_vctk_gender():
    id_list, gender_list = vctk_id_gender_list()
    gender_group_labels_arr = []
    for voice_meta in metad_by_singer_list:
        uttrs_fps = voice_meta[2:]
        for fp in uttrs_fps:
            track_name = os.path.basename(fp)[:-4]
            singer_id = track_name.split('_')[0]
            idx = id_list.index(singer_id)

            gender = gender_list[idx]
            if 'm' in gender.lower():
                gender_group_labels_arr.append('male')
            elif 'f' in gender.lower():
                gender_group_labels_arr.append('female')
            else:
                raise Exception(f'Gender value not recognised for excerpt {track_name} in csv row {idx}')

    gender_group_labels_arr = np.asarray(gender_group_labels_arr)
    all_labels_arrs.append(gender_group_labels_arr)
    all_label_names.append('gender')
    all_labels_class_sizes.append(1)
    return gender_group_labels_arr

def get_damp_gender(ignore_unknowns=False, csv_path='/homes/bdoc3/my_data/text_data/damp/intonation_metadata.csv'):
    """
    Get entries from gender csv file, return single list of performer-gender tuples
    """

    f = open(csv_path, 'r')
    reader = csv.reader(f)
    header = next(reader)
    singer_meta = [row for row in reader]
    if ignore_unknowns:
        performer_gender_list = [(row[0].split('_')[0], row[8]) for row in singer_meta if row[8] != ' None']
    else:
        performer_gender_list = [(row[0].split('_')[0], row[8]) for row in singer_meta]

    return performer_gender_list
# Uses metadata_for_synth.pkl to synthesize/convert audio

import time, sys, os, pdb, pickle, argparse, shutil, yaml, torch, math, time, random, pickle
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
from model_sie import SingerIdEncoder
from collections import OrderedDict

from my_utils.my_os import recursive_file_retrieval

sys.path.insert(1, '/homes/bdoc3/my_utils')
from my_container import str2bool
from my_audio.world import get_world_feats
from my_audio.mel import audio_to_mel_autovc, db_normalize
from my_audio.pitch import midi_as_onehot
from my_arrays import fix_feat_length
from my_normalise import norm_feat_arr, get_norm_stats

def str2bool(v):
    return v.lower() in ('true')


def setup_sie(device):
    sie_checkpoint = torch.load(os.path.join(SIE_path, 'saved_model.pt'))
    new_state_dict = OrderedDict()
    sie_num_feats_used = sie_checkpoint['model_state']['lstm.weight_ih_l0'].shape[1]
    # sie_num_voices_used = sie_checkpoint['model_state']['class_layer.weight'].shape[0]
    for i, (key, val) in enumerate(sie_checkpoint['model_state'].items()):
        new_state_dict[key] = val 
    sie =  SingerIdEncoder(device, torch.device("cpu"), sie_num_feats_used)
    for param in sie.parameters():
        param.requires_grad = False
    sie_optimizer = torch.optim.Adam(sie.parameters(), 0.0001)
    sie.load_state_dict(new_state_dict)
    for state in sie_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(device)
    sie.to(device)
    sie.eval()
    return sie


def setup_gen(Generator, num_feats, device):
    G = Generator(dim_neck, dim_emb, dim_pre, sample_freq, num_feats)
    g_optimizer = torch.optim.Adam(G.parameters(), adam_init)
    g_checkpoint = torch.load(autovc_model_ckpt, map_location='cpu')
    G.load_state_dict(g_checkpoint['model_state_dict'])
    g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
    # fixes tensors on different devices error
    # https://github.com/pytorch/pytorch/issues/2830
    for state in g_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.cuda(which_cuda)
    G.to(device)
    return G


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ad','--audio_dir', type=str, default='/homes/bdoc3/my_data/audio_data/damp_desilenced_concat', help='name the model used for inferring')
    parser.add_argument('-ft','--feat_type', type=str, help='name the model used for inferring')
    parser.add_argument('-um','--use_model', type=str, default='worldHarmsOnly_pitchCond_Window608', help='name the model used for inferring')
    parser.add_argument('-wc','--which_cuda', type=int, default=0, help='Determine which cuda to use')
    parser.add_argument('c','--convert', type=str2bool, default=True)

    inputs = parser.parse_args()
    which_cuda = inputs.which_cuda
    cudnn.benchmark = True
    convert_style = inputs.convert_style
    autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
    autovc_model_dir = os.path.join(autovc_model_saves_dir, inputs.use_model)
    autovc_model_ckpt = os.path.join(autovc_model_dir, 'saved_model.pt')
    device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else 'cpu')


    # use variables to add dir to sys and load more files/params
    sys.path.insert(1, autovc_model_dir)
    from this_train_params import *
    from this_model_vc import Generator
    from this_sv_converter import AutoSvc
    import this_utils


    #setup models
    SIE_ckpt = os.path.join(SIE_path, 'saved_model.pt')
    autovc_ckpt = os.path.join(autovc_model_dir +'saved_model')
    with open(os.path.join(SVC_feat_dir, 'feat_params.yaml')) as File:
        feat_params = yaml.load(File, Loader=yaml.FullLoader)
    if use_aper_feats == True:
        num_spectral_feats = feat_params['num_harm_feats'] + feat_params['num_harm_feats']
    else:
        num_spectral_feats = feat_params['num_harm_feats']
    SIE = utils.setup_sie(SIE_path, device)
    G = utils.setup_gen(Generator, num_spectral_feats)

    import torch
    import librosa
    import soundfile as sf
    import pickle
    from synthesis import build_model
    from synthesis import wavegen

    model = build_model().to(device)

    # sys.path.insert(1, '/homes/bdoc3/my_data/autovc_models') # usually the cwd is priority, so index 1 is good enough for our purposes here
    # from hparams import hparams
    checkpoint = torch.load("/homes/bdoc3/my_data/autovc_models/checkpoint_step001000000_ema.pth")
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)

    _, singer_list = recursive_file_retrieval(inputs.audio_dir)
    singer_set = {}
    random.shuffle(singer_list)
    for fp in singer_list:
        if os.path.basename(fp).split('_')[0] not in singer_set:
            singer_set.add(os.path.basename(fp).split('_')[0])
        else:
            singer_list.remove(fp)   
    assert len(singer_list) == len(singer_set)

    if inputs.feat_type == 'mel':
        mel_filter = mel(feat_params['sr'], feat_params['fft_size'], fmin=feat_params['fmin'], fmax=feat_params['fmax'], n_mels=feat_params['num_harm_feats']).T
        # self.mel_filter = mel(16000, 1024, fmin=90, fmax=7600, n_mels=80).T
        min_level = np.exp(-100 / 20 * np.log(10))
        hop_size = int((feat_params['frame_dur_ms']/1000) * feat_params['sr'])

    for sp in singer_list:
        fn = os.path.basename(sp)
        singer_id = fn.split('_')[0]
        y, samplerate = librosa.load(sp, sr=feat_params['sr'])
        
        if inputs.feat_type == 'mel':
            db_unnormed_melspec = audio_to_mel_autovc(y, feat_params['fft_size'], hop_size, mel_filter)
            feats = db_normalize(db_unnormed_melspec, min_level)
        else:
            feats = get_world_feats(y, feat_params)

        feats, start_step = fix_feat_length(feats, window_timesteps)

        if not use_aper_feats:
            feats = feats[:,:-(feat_params['num_aper_feats'])]

        if SVC_pitch_cond:
            
            # find corresponding file from pitch dir and return pitch_predictions
            for subset in ['dev', 'test', 'train', None]:
                if subset == None:
                    raise FileNotFoundError(f'Target file {fn} could not be found in pitch directory {pitch_dir}')
                target_file = os.path.join(pitch_dir, subset, singer_id, fn)
                if os.path.exists(target_file):
                    pitch_pred = np.load(target_file)[:,-2:]
                    break
            
            midi_contour = pitch_pred[:,0]
            # remove the interpretted values generated because of unvoiced sections
            unvoiced = pitch_pred[:,1].astype(int) == 1
            midi_contour[unvoiced] = 0
            
            try:
                if start_step < 0:
                    midi_trimmed, _ = fix_feat_length(midi_contour, window_timesteps)
                else:
                    midi_trimmed = midi_contour[start_step:(start_step+window_timesteps)]
                onehot_midi = midi_as_onehot(midi_trimmed, midi_range)
            except Exception as e:
                print(f'Exception {e} caused by file {fn}')
                pdb.set_trace()
        
        else:
            onehot_midi = np.zeros((window_timesteps, len(midi_range)+1))

        feat_list.append(feats, onehot_midi)

    subdir_for_wavs = os.path.join(autovc_model_dir, 'generated_wavs')
    # # convert all in metadata_list to new cuda
    # tmp = []
    # for example in metadata_list:
    #     tmp.append([tens_ob.to(config.device) for tens_ob in example])
    # metadata_list = tmp    

    # for i, metadata in enumerate(metadata_list):
    #     print(i,'/',len(metadata_list))
    #     x_real, org_style_idx, singer_idx, emb_org = metadata
    #     all_spmels = []
    #     #all_spmels = [x_real.squeeze(1)[0].cpu().detach().numpy()]
    #     # start converting
    #     _, x_identic_psnt, _, _, _ = G(x_real, emb_org, emb_org)
    #     all_spmels.append(x_identic_psnt.squeeze(1)[0].cpu().detach().numpy())
    #     num_unconv_styles = 2
    #     if convert_style == True:
    #         for trg_style_idx in range(len(avg_embs)):
    #             emb_trg = torch.tensor(avg_embs[trg_style_idx]).to(config.device).unsqueeze(0)
    #             _, x_identic_psnt, _, _, _ = G(x_real, emb_org, emb_trg)
    #             all_spmels.append(x_identic_psnt.squeeze(1)[0].cpu().detach().numpy())

    #     plt.figure(figsize=(20,5))
    #     for j in range(len(all_spmels)):
    #         plt.subplot(1,len(all_spmels),j+1)
    #         if j == 0: plt.title('original_' +singer_names[singer_idx][:-1] +'_' +style_names[org_style_idx])    
    #         elif j == 1: plt.title('resynthOrg_' +singer_names[singer_idx][:-1] +'_' +style_names[org_style_idx])
    #         else:
    #             plt.title(singer_names[singer_idx][:-1] +style_names[org_style_idx] +'_to_' +str(style_names[j-num_unconv_styles]))
    #         plt.imshow(np.rot90(all_spmels[j]))
    #     plt.savefig(subdir_for_wavs +'/example' +str(counter) +'_spmels')

    #     # synthesize nu shit
    #     for k, spmel  in enumerate(all_spmels):
    #         # x_identic_psnt = tensor.squeeze(0).squeeze(0).detach().cpu().numpy()
    #         waveform = wavegen(model, which_cuda, c=spmel)   
    #         #     librosa.output.write_wav(name+'.wav', waveform, sr=16000)
    # #        if k == 0:
    # #            sf.write(subdir_for_wavs +f'/example{counter}_{singer_names[singer_idx]}{style_names[org_style_idx]}_ORG.wav', waveform, samplerate=16000)
    #         if k == 0:
    #             sf.write(subdir_for_wavs +f'/example{counter}_{singer_names[singer_idx]}{style_names[org_style_idx]}_synthed_from_org.wav', waveform, samplerate=16000)
    #         else:
    #             sf.write(subdir_for_wavs +f'/example{counter}_{singer_names[singer_idx]}{style_names[org_style_idx]}_to_{style_names[k-1]}.wav', waveform, samplerate=16000)
    #     counter +=2

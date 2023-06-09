from re import A
import sys, os, pdb, argparse, yaml, torch, random, csv, librosa
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import librosa
from librosa.filters import mel
import soundfile as sf
from tqdm import tqdm

import sys, os
if os.path.abspath('../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('../my_utils'))
import utils
from model_sie import SingerIdEncoder
from collections import OrderedDict
from my_audio.world import get_world_feats
from my_audio.mel import audio_to_mel_autovc
from my_os import recursive_file_retrieval
from synth_params import *

"""

This script uses:
Variables from the synth_params file,
A csv file that contains the file_name (singer/song ID), gender and timestep index

It does the following:
Finds the relevant chunk of the desired audio file using the csv file
COnverts these to features
Saves a plot of these features as figures
Uses wavenet to convert features back to audio

"""

parser = argparse.ArgumentParser()
parser.add_argument('-ad','--audio_dir', type=str, default='../singer-identity-encoder/example_feats/train', help='name the model used for inferring')
parser.add_argument('-ft','--feat_type', type=str, default='mel', help='name the model used for inferring')
parser.add_argument('-ae','--audio_ext', type=str, default='.m4a', help='name the model used for inferring')
parser.add_argument('-wc','--which_cuda', type=int, default=0, help='Determine which cuda to use')

config = parser.parse_args()


print('setting up variables...')
which_cuda = config.which_cuda
cudnn.benchmark = True
device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else 'cpu')

with open(os.path.join(SVC_feat_dir, 'feat_params.yaml')) as File:
    feat_params = yaml.load(File, Loader=yaml.FullLoader)

mel_filter = mel(feat_params['sr'], feat_params['fft_size'], fmin=feat_params['fmin'], fmax=feat_params['fmax'], n_mels=feat_params['num_harm_feats']).T
min_level = np.exp(-100 / 20 * np.log(10))
hop_size = int(feat_params['sr'] * (feat_params['frame_dur_ms']/1000))


print('getting selected files from csv...')
fp_gender_list = []
f = open(use_loader +'_test_examples.csv', 'r')
reader = csv.reader(f)
header = next(reader)
fp_gender_list = [row for row in reader]


print('find and list the audio files associated with these files...')
_, audio_fps = recursive_file_retrieval(config.audio_dir)
audio_fps = [fp for fp in os.path.abspath(audio_fps) if fp.endswith(config.audio_ext) and not fp.startswith('.')]
audio_list = []
for entry in fp_gender_list:
    fn = entry[0]
    for afp in audio_fps:
        afn = os.path.basename(afp)[:-len(config.audio_ext)]
        if fn == afn:
            audio_list.append(afp)
            break


print('selecting audio chunks...')
audio_chunk_list = []
random.seed(0)
for i, entry in enumerate(fp_gender_list):
    fn = entry[0]
    gender = entry[1]
    singer_id = gender +'-' +fn +'_' +str(int(int(entry[2])*feat_params['frame_dur_ms']/1000)) +'s'
    
    y, _ = librosa.load(audio_list[i], sr=feat_params['sr'])
    start_sample = int(int(entry[2]) * hop_size)
    end_sample = int((int(entry[2]) + window_timesteps) * hop_size)
    print(y.shape, start_sample, end_sample)
    y_chunk = y[start_sample:end_sample]

    if config.feat_type == 'mel':
        final_feats = audio_to_mel_autovc(y_chunk, feat_params['fft_size'], hop_size, mel_filter)
        audio_chunk_list.append((singer_id, final_feats))

    elif config.feat_type == 'world':
        feat_params['fft_size'] = 1024
        feats = get_world_feats(y_chunk, feat_params)
        harms = feats[:,:feat_params['num_harm_feats']]
        apers = feats[:,feat_params['num_harm_feats']:-2]
        midi_contour = feats[:,-2:]
        onehot_midi = utils.get_onehot_midi(midi_contour, midi_range)
        audio_chunk_list.append((singer_id, harms, apers, onehot_midi))
    

wav_dir = os.path.join('../generated_audio', svc_model_name)
if not os.path.exists(wav_dir):
    os.makedirs(wav_dir)

print('saving plots of features')
plt.figure(figsize=(20,5))
rows = len(audio_chunk_list)//2
cols = 2
for j, entry in enumerate(audio_chunk_list):
    grouped_4 = j % 4
    if grouped_4 == 0:
        fig, axs = plt.subplots(rows,cols)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle(entry[0] +'_audio2feats2audio')
    row = grouped_4 // cols
    col = grouped_4 % 2  
    feat_recon_npy = entry[1]
    axs[row,col].imshow(np.rot90(feat_recon_npy))
    plt.tight_layout()
    plt.savefig(os.path.join(wav_dir, entry[0] +'_audio2feats2audio'))
plt.close()

print('loading libraries for conversion...')
from my_audio.world import mfsc_to_world_to_audio
from convert.synthesis import build_model
from convert.synthesis import wavegen

model = build_model().to(device)

# sys.path.insert(1, '/homes/bdoc3/my_data/autovc_models') # usually the cwd is priority, so index 1 is good enough for our purposes here
# from hparams import hparams
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_models/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])
model.to(device)

print('converting audio...')
# args = [config, feat_params, model, wav_dir]
# multithread_chunks(utils.synthesize_audio, converted_feats, 2, args)

for entry in audio_chunk_list:
    
    if config.feat_type == 'mel':
        waveform = wavegen(model, config.which_cuda, c=entry[1])
    elif config.feat_type == 'world':
        waveform = mfsc_to_world_to_audio(entry[1], entry[2], entry[3], feat_params)

    sf.write(os.path.join(wav_dir, entry[0] +'_audio2feats2audio.wav'), waveform, samplerate=feat_params['sr'])


print('done!')
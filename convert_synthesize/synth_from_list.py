import os, pickle, argparse, shutil, yaml, torch, random, pickle, csv, librosa
os.system('module load ffmpeg')
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import librosa
import soundfile as sf
from tqdm import tqdm

import sys
if os.path.abspath('../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('../my_utils'))
from my_arrays import fix_feat_length
from my_os import recursive_file_retrieval
from my_interaction import binary_answer

import utils
from model_sie import SingerIdEncoder
from collections import OrderedDict



parser = argparse.ArgumentParser()
parser.add_argument('-um','--use_model', type=str, default='damp60Harms_NOzerodPitchCond_Window384_take2', help='name the model used for inferring')
parser.add_argument('-ol','--old_loss', type=int, default=0, help='Used as boolean. Determine whether the autosvc model used the old loss or the new loss (same to loss in autovc_basic)')
parser.add_argument('-ad','--audio_dir', type=str, default='/homes/bdoc3/my_data/audio_data/damp_desilenced_concat', help='name the model used for inferring')
parser.add_argument('-ae','--audio_ext', type=str, default='.m4a', help='name the model used for inferring')
parser.add_argument('-ss','--subset', type=str, default='', help='name the model used for inferring')
parser.add_argument('-wc','--which_cuda', type=int, default=0, help='Determine which cuda to use')
parser.add_argument('-ns','--num_singers', type=int, default=4, help='Determine which cuda to use')
parser.add_argument('-sp','--start_point', type=int, default=0, help='Determine which cuda to use')
parser.add_argument('-fp','--finish_point', type=int, default=1000, help='Determine which cuda to use')

config = parser.parse_args()

if 'mel' in config.use_model or 'Mel' in config.use_model:
    print('Are you using the correectly pretrained WaveNet, and have you set the hparams file accordingly?')
    if not binary_answer():
        exit(1)
    print('Have you edited the WaneNet synthesis file to reflect which cuda ur using?')
    if not binary_answer():
        exit(1)

# declare variables
if config.num_singers % 2 !=0:
    raise Exception('number of singers must be even.')

print('setting up variables...')
which_cuda = config.which_cuda
cudnn.benchmark = True
autovc_model_saves_dir = '/homes/bdoc3/my_data/autovc_models/autoSvc'
autosvc_audio = '/homes/bdoc3/s/autoSvc/generated_audio'
autovc_model_dir = os.path.join(autovc_model_saves_dir, config.use_model)
autovc_model_ckpt = os.path.join(autovc_model_dir, 'saved_model.pt')
device = torch.device(f'cuda:{which_cuda}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
loss_device = torch.device("cpu")

# use variables to add dir to sys and load more files/params
sys.path.insert(1, autovc_model_dir)
from this_train_params import *

if use_avg_singer_embs:
    metadata_path = '/homes/bdoc3/singer-identity-encoder/metadata'
    # train_singer_metadata = pickle.load(open(os.path.join(metadata_path, os.path.basename(SIE_path), os.path.basename(SIE_feat_dir) +'_train_singers_metadata.pkl'), 'rb'))
    subset_singer_metadata = pickle.load(open(os.path.join(metadata_path, os.path.basename(SIE_path), os.path.basename(SIE_feat_dir) +f'_{config.subset}_singers_metadata.pkl'), 'rb'))
    sid_list = [i[0] for i in subset_singer_metadata]

def gen_feat_params(feat_dir):

    with open(os.path.join(feat_dir, 'feat_params.yaml')) as File:
        feat_params = yaml.load(File, Loader=yaml.FullLoader)

    # there's a better way to write this
    if use_aper_feats:
        num_spectral_feats = feat_params['num_harm_feats'] + feat_params['num_aper_feats']
    else:
        num_spectral_feats = feat_params['num_harm_feats']
    if 'mel' in feat_dir:
        num_spectral_feats = feat_params['num_harm_feats']

    return feat_params, num_spectral_feats


SIE_feat_params, SIE_num_used_feats = gen_feat_params(SIE_feat_dir)
SVC_feat_params, SVC_num_used_feats = gen_feat_params(SVC_feat_dir)


print('gathering audio paths for conversion...')

fp_gender_list = []
f = open(use_loader +f'_{config.subset}_examples.csv', 'r')
reader = csv.reader(f)
header = next(reader)
for row in reader:
    row[0] = os.path.join(SVC_feat_dir, '/'.join(row[0].split('/')[-3:])) # fix this messiness later
    fp_gender_list.append(row)
if len(fp_gender_list[0]) == 2:
    fpgl_missing_timestep = True
else:
    fpgl_missing_timestep = False



# find and list the audio files associated with the chosen annotated files preceeding this cell
_, audio_fps = recursive_file_retrieval(config.audio_dir)
audio_fps = [fp for fp in audio_fps if fp.endswith(config.audio_ext) and not fp.startswith('.')]
audio_list = []
for entry in fp_gender_list[:config.num_singers]:
    fp = entry[0]
    fn = os.path.basename(fp)[:-4]
    for afp in audio_fps:
        afn = os.path.basename(afp)[:-len(config.audio_ext)]
        if fn == afn:
            audio_list.append(afp)

print('extracting features from audio...')
# hopsizes are identical, so no need to do both
hop_size = int(SVC_feat_params['sr'] * (SVC_feat_params['frame_dur_ms']/1000))
updated_fpgl = []
feat_list = []
audio_chunk_list = []
random.seed(0)
for i, entry in enumerate(fp_gender_list):
    fp = entry[0]
    gender = entry[1]
    fn = os.path.basename(fp)
    SVC_feats = np.load(fp)

    if not fpgl_missing_timestep:
        SVC_feats, start_step = fix_feat_length(SVC_feats, window_timesteps, int(entry[2]))
    else:
        SVC_feats, start_step = fix_feat_length(SVC_feats, window_timesteps)

    SVC_input_feats_npy = SVC_feats[:,:SVC_num_used_feats]
    aper_feats_npy = SVC_feats[:,SVC_num_used_feats:-2]
    midi_contour_npy = SVC_feats[:,-2:]

    singer_id = gender +'-' +os.path.basename(fp)[:-4] +'_' +str(int(start_step*SVC_feat_params['frame_dur_ms']/1000)) +'s'
    
    corresponding_SIE_fp = os.path.join(SIE_feat_dir, '/'.join(fp.split('/')[-3:]))
    SIE_input_feats_npy = np.load(corresponding_SIE_fp)[:,:SIE_num_used_feats]
    
    feat_list.append((singer_id, midi_contour_npy, SVC_input_feats_npy, SIE_input_feats_npy, aper_feats_npy))
    try:
        y, _ = librosa.load(audio_list[i], sr=SVC_feat_params['sr'])
    except Exception as e:
        print(e)
    start_sample = int(start_step * hop_size)
    end_sample = int((start_step + window_timesteps) * hop_size)
    # print(y.shape, start_sample, end_sample)
    y_chunk = y[start_sample:end_sample]
    audio_chunk_list.append((singer_id, y_chunk))
    updated_fpgl.append([fp, gender, start_step])
# save updated contents to test examples
f = open(use_loader +'_test_examples.csv', 'w')
writer = csv.writer(f)
writer.writerow(header)
for entry in updated_fpgl:
    writer.writerow(entry)
f.close()


print('Setting up SIE model...')
SIE_ckpt = os.path.join(SIE_path, 'saved_model.pt')
autovc_ckpt = os.path.join(autovc_model_dir, 'saved_model.pt')
SIE, _ = utils.setup_sie(device, loss_device, SIE_path, adam_init)
if SVC_pitch_cond:
    pitch_dim = len(midi_range)+1
else:
    pitch_dim = 0

print('Setting up SVC model...')
G, _, _ = utils.setup_gen(dim_neck, dim_emb, dim_pre, sample_freq, SVC_num_used_feats, pitch_dim, device, autovc_ckpt, adam_init)

print('now converting features using models...')
converted_feats = []

def get_timbre_emb(model, device, input_npy):
    input_tns = torch.from_numpy(input_npy).to(device).float().unsqueeze(0)
    assert model.lstm.weight_ih_l0.shape[1] == input_tns.shape[2]
    timbre_emb = model(input_tns)
    return timbre_emb

for i, (singer_id_src, midi_contour_npy_src, SVC_input_feats_npy_src, SIE_input_feats_npy_src, aper_npy_src) in enumerate(feat_list):
    for j, (singer_id_trg, midi_contour_npy_trg, SVC_input_feats_npy_trg, SIE_input_feats_npy_trg, aper_npy_trg) in enumerate(feat_list):

#         uncomment this bit if you want to exclude original reconstructions for model testing
#         if i == j:
#             continue

        print(j+(i*len(feat_list))+1,'/',len(feat_list)**2)

        # src/trg feats to SIE
        if use_avg_singer_embs:
            # pdb.set_trace()
            sid_src = singer_id_src[2:].split('_')[0]
            sid_src_idx = sid_list.index(sid_src)
            timbre_src_npy = subset_singer_metadata[sid_src_idx][1]
            timbre_src_tns = torch.from_numpy(timbre_src_npy).to(device).float().unsqueeze(0)

            sid_trg = singer_id_trg[2:].split('_')[0]
            sid_trg_idx = sid_list.index(sid_trg)
            timbre_trg_npy = subset_singer_metadata[sid_trg_idx][1]
            timbre_trg_tns = torch.from_numpy(timbre_trg_npy).to(device).float().unsqueeze(0)


        else:
            timbre_src_tns = get_timbre_emb(SIE, device, SIE_input_feats_npy_src)
            timbre_trg_tns = get_timbre_emb(SIE, device, SIE_input_feats_npy_trg)

        # src feats to SVC
        SVC_input_feats_tns_src = torch.from_numpy(SVC_input_feats_npy_src).to(device).float().unsqueeze(0)

        # src pitch to SVC
        if SVC_pitch_cond:
            onehot_midi_npy_src = utils.get_onehot_midi(midi_contour_npy_src, midi_range)
            onehot_midi_tns_src = torch.from_numpy(onehot_midi_npy_src).to(device).float().unsqueeze(0)
            feat_tns_prnt, feat_tns_psnt, code_real, _, _ = G(SVC_input_feats_tns_src, timbre_src_tns, timbre_trg_tns, onehot_midi_tns_src)
        else:
            feat_tns_prnt, feat_tns_psnt, code_real, _, _ = G(SVC_input_feats_tns_src, timbre_src_tns, timbre_trg_tns)
        
        # recently changed loss function from old to new. Make sure you are using the right one
        if config.old_loss:
            feat_recon_npy = (feat_tns_prnt + feat_tns_psnt).squeeze(1)[0].cpu().detach().numpy()
        else:
            feat_recon_npy = feat_tns_psnt.squeeze(1)[0].cpu().detach().numpy()
        
        converted_id = singer_id_src +'__X__' +singer_id_trg

        converted_feats.append((converted_id, feat_recon_npy, aper_npy_src, midi_contour_npy_src))

wav_dir = os.path.join(autosvc_audio, svc_model_name)
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)

print('generating and saving plots...')
plt.figure(figsize=(20,5))
rows = config.num_singers//2
cols = 2
for j, entry in enumerate(converted_feats):
    grouped_4 = j % config.num_singers
    if grouped_4 == 0:
        fig, axs = plt.subplots(rows,cols)
        fig.set_size_inches(18.5, 10.5)
        fig.suptitle(entry[0].split('X')[0])
    row = grouped_4 // cols
    col = grouped_4 % 2  
    feat_recon_npy = entry[1]
    axs[row,col].imshow(np.rot90(feat_recon_npy))
    plt.tight_layout()
    plt.savefig(os.path.join(wav_dir, entry[0].split('X')[0]))
plt.close()

print('loading libraries for conversion...')
from my_audio.world import mfsc_to_world_to_audio
from convert.synthesis import build_model
from convert.synthesis import wavegen
from my_threads import multithread_chunks

model = build_model().to(device)

# sys.path.insert(1, '/homes/bdoc3/my_data/autovc_models') # usually the cwd is priority, so index 1 is good enough for our purposes here
# from hparams import hparams
checkpoint = torch.load("/homes/bdoc3/my_data/autovc_models/checkpoint_step001000000_ema.pth")
model.load_state_dict(checkpoint["state_dict"])
model.to(device)


wav_dir = os.path.join(autosvc_audio, svc_model_name)
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)

converted_feats = converted_feats[config.start_point:config.finish_point]

print('converting audio...')
args = [config, SVC_feat_params, model, wav_dir]
# multithread_chunks(utils.synthesize_audio, converted_feats, 2, args)

for i, (converted_id, x_identic, aper_src, onehot_midi_src)  in tqdm(enumerate(converted_feats)):
    if 'world' in config.use_model or 'harms' in config.use_model or 'Harms' in config.use_model:
        SVC_feat_params['fft_size'] = 1024
        waveform = mfsc_to_world_to_audio(x_identic, aper_src, onehot_midi_src, SVC_feat_params)
    elif 'mel' in config.use_model:
        waveform = wavegen(model, which_cuda, c=x_identic)
    sf.write(os.path.join(wav_dir, converted_id +'.wav'), waveform, samplerate=SVC_feat_params['sr'])

print('saving original audio chunks...')
for singer_id, y_chunk in audio_chunk_list:
    sf.write(os.path.join(wav_dir, singer_id +'.wav'), y_chunk, samplerate=SVC_feat_params['sr'])

print('done!')


import sys, os, pdb, pickle, argparse, shutil, yaml, torch, random, pickle, csv, librosa

os.system("module load ffmpeg")
from torch.backends import cudnn
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import librosa
import soundfile as sf
from tqdm import tqdm

import sys, os

if os.path.abspath("../my_utils") not in sys.path:
    sys.path.insert(1, os.path.abspath("../my_utils"))
import utils
from model_sie import SingerIdEncoder
from collections import OrderedDict
from my_container import str2bool
from my_audio.world import get_world_feats
from my_audio.mel import audio_to_mel_autovc, db_normalize
from my_audio.pitch import midi_as_onehot
from my_arrays import fix_feat_length
from my_normalise import norm_feat_arr, get_norm_stats
from my_os import recursive_file_retrieval
from my_interaction import binary_answer


def gen_feat_params(feat_dir):
    with open(os.path.join(feat_dir, "feat_params.yaml")) as File:
        feat_params = yaml.load(File, Loader=yaml.FullLoader)

    # there's a better way to write this
    if use_aper_feats:
        num_spectral_feats = (
            feat_params["num_harm_feats"] + feat_params["num_aper_feats"]
        )
    else:
        num_spectral_feats = feat_params["num_harm_feats"]
    if "mel" in feat_dir:
        num_spectral_feats = feat_params["num_harm_feats"]

    return feat_params, num_spectral_feats


parser = argparse.ArgumentParser()
parser.add_argument(
    "-um",
    "--use_model",
    type=str,
    default="damp60Harms_NOzerodPitchCond_Window384_take2",
    help="name the model used for inferring",
)
parser.add_argument(
    "-fd",
    "--feat_dir",
    type=str,
    default="",
    help="Used as boolean. Determine whether the autosvc model used the old loss or the new loss (same to loss in autovc_basic)",
)
parser.add_argument(
    "-sp", "--start_point", type=int, default=0, help="Determine which cuda to use"
)
parser.add_argument(
    "-fp", "--finish_point", type=int, default=1000, help="Determine which cuda to use"
)
parser.add_argument("-flp", type=str, default="", help="Determine which cuda to use")
parser.add_argument("-wc", type=int, default=0, help="Determine which cuda to use")
config = parser.parse_args()

print("setting up variables...")
feat_list_path = config.flp
which_cuda = config.wc
cudnn.benchmark = True
device = torch.device(f"cuda:{which_cuda}" if torch.cuda.is_available() else "cpu")

SVC_feat_params, SVC_num_used_feats = gen_feat_params(config.feat_dir)

print("loading libraries for conversion...")
from my_audio.world import mfsc_to_world_to_audio
from convert.synthesis import build_model
from convert.synthesis import wavegen

model = build_model().to(device)

# sys.path.insert(1, '/homes/bdoc3/my_data/autovc_models') # usually the cwd is priority, so index 1 is good enough for our purposes here
# from hparams import hparams
checkpoint = torch.load(
    "/homes/bdoc3/my_data/autovc_models/checkpoint_step001000000_ema.pth"
)
model.load_state_dict(checkpoint["state_dict"])
model.to(device)


wav_dir = os.path.join("/homes/bdoc3/s/autoSvc/generated_audio", config.use_model)
if not os.path.exists(wav_dir):
    os.mkdir(wav_dir)

converted_feats = pickle.load(open(feat_list_path, "rb"))
converted_feats = converted_feats[config.start_point : config.finish_point]

print("converting audio...")
args = [config, SVC_feat_params, model, wav_dir]

# multithread_chunks(utils.synthesize_audio, converted_feats, 2, args)

for i, entry in tqdm(enumerate(converted_feats)):
    if (
        "world" in config.use_model
        or "harms" in config.use_model
        or "Harms" in config.use_model
    ):
        converted_id, x_identic, aper_src, onehot_midi_src = entry
        SVC_feat_params["fft_size"] = 1024
        waveform = mfsc_to_world_to_audio(
            x_identic, aper_src, onehot_midi_src, SVC_feat_params
        )
    elif "mel" in config.use_model or "Mel" in config.use_model:
        converted_id, x_identic = entry
        waveform = wavegen(model, config.which_cuda, c=x_identic)
    sf.write(
        os.path.join(wav_dir, converted_id + ".wav"),
        waveform,
        samplerate=SVC_feat_params["sr"],
    )

print("done!")

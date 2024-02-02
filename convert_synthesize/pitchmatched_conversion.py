import argparse
from collections import OrderedDict
import importlib
import os
import pdb
import pickle
import random
import sys

import numpy as np
import soundfile as sf
import torch
import yaml
from torch.backends import cudnn
import matplotlib.pyplot as plt

this_script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(this_script_dir)
super_dir = os.path.dirname(root_dir)
my_utils_dir = os.path.join(super_dir, 'my_utils')
sie_dir = os.path.join(super_dir, 'singer-identity-encoder')
if os.path.abspath(my_utils_dir) not in sys.path: sys.path.insert(1, os.path.abspath(my_utils_dir))
if os.path.abspath(root_dir) not in sys.path: sys.path.insert(1, os.path.abspath(root_dir))

import utils
from synthesis import build_model, wavegen
from convert_utils import (
    get_gender_lists,
    best_pitch_matching_idx,
    get_relevant_avg_pitches,
    NoMatchError,
)

from my_arrays import fix_feat_length, find_runs
from neural.torch_utils import container_to_tensor, tensor_to_array
from my_os import recursive_file_retrieval
from my_audio.pitch import midi_as_onehot
from my_audio.midi import freqToMidi
from my_audio.world import get_world_feats
from my_audio.utils import audio_io


class Obj:
    pass


def get_feats(path):
    spec_feats = np.load(path)
    fn = os.path.basename(path)
    if "crepe" in config.pitch_dir:
        crepe_data = np.load(
            os.path.join(config.pitch_dir, config.subset, fn.split("_")[0], fn[:-4] + ".npz")
        )
        pitches = crepe_data["arr_0"]
        conf = crepe_data["arr_1"]
        unvoiced = (
            conf < 0.5
        )  # determined by looking at pitch and conf contours against audio in sonic visualizer
        midi_contour = freqToMidi(pitches)
    elif "world" in config.pitch_dir:
        world_feats = np.load(
            os.path.join(
                config.pitch_dir,
                config.subset,
                fn.split("_")[0],
                fn))
        pitches = world_feats[:, -2:]
        midi_contour = pitches[:, 0]
        unvoiced = pitches[:, 1].astype(int) == 1
    else:
        #get pitch from audio - world and crepe dims don't seem to match the example dims
        audio_path = os.path.join(config.audio_dir, config.subset, fn.split("_")[0], fn[:-4] + config.audio_ext)
        y = audio_io(audio_path, SVC_feat_params['sr'])
        SVC_feat_params_for_pitch = SVC_feat_params.copy()
        SVC_feat_params_for_pitch['w2w_process'] = 'wav2world'
        SVC_feat_params_for_pitch['dim_red_method'] = 'world'
        SVC_feat_params_for_pitch['num_aper_feats'] = 4
        world_feats = get_world_feats(y, SVC_feat_params_for_pitch)
        pitches = world_feats[:, -2:]
        midi_contour = pitches[:, 0]
        unvoiced = pitches[:, 1].astype(int) == 1

    midi_contour[unvoiced] = 0
    return spec_feats, midi_contour


def get_song_path(gender):
    gender_list = gender_separated_lists[gender]
    rand_int = random.randint(0, len(gender_list) - 1)
    name = gender_list[rand_int]
    song_list = os.listdir(os.path.join(os.path.join(config.feat_path, config.subset), name))
    song_name = random.choice(song_list)
    song_path = os.path.join(config.feat_path, config.subset, name, song_name)

    return song_path, rand_int


def matching_pitch_clip(
    trg_gender,
    avg_src_midi,
    src_path,
    pitch_match,
    track_search_tolerance=11,
    voiced_percent_tolerance=0.7,
    avoid_same_singer_conv=False
):
    
    matched_singer_found = False
    attempt_num = 0
    while matched_singer_found == False:
        # choose random target
        trg_path, trg_rand_gend_int = get_song_path(trg_gender)
        if src_path == trg_path:
            print('Same path for source and target. Regenerating target from gendered list')
        if avoid_same_singer_conv:
            if os.path.dirname(trg_path) == os.path.dirname(src_path):
                print('target path matches source path. Attempting new trg_gender')
                continue
        print(
            f"attempt num: {attempt_num}, candidate_song: {os.path.basename(trg_path)}"
        )

        # get feats
        trg_spec_feats, trg_midi_continuous = get_feats(trg_path)
        trg_midi_continuous = np.round(trg_midi_continuous)
        trg_midi_onehot = midi_as_onehot(
            trg_midi_continuous, this_train_params.midi_range
        )

        if pitch_match:
            average_trg_pitches = get_relevant_avg_pitches(
                trg_midi_continuous, this_train_params.window_timesteps
            )
            start_of_chunk_idx = best_pitch_matching_idx(
                average_trg_pitches, avg_src_midi
            )
        else:
            start_of_chunk_idx = random.randint(
                0, len(trg_midi_continuous[: -
                                           this_train_params.window_timesteps])
            )

        # get pitch of trg and compute distance from src
        if start_of_chunk_idx >= 0:
            trg_midi_continuous_clip, _ = fix_feat_length(
                trg_midi_continuous,
                this_train_params.window_timesteps,
                offset=start_of_chunk_idx,
            )
            trg_midi_onehot_clip, _ = fix_feat_length(
                trg_midi_onehot,
                this_train_params.window_timesteps,
                offset=start_of_chunk_idx,
            )
            # ensure there's enough voiced content, otherwise search again
            voiced = trg_midi_continuous_clip != 0
            if (sum(voiced) / len(voiced)) < voiced_percent_tolerance:
                print(f'Sampled clip from file {trg_path} at index {start_of_chunk_idx} contained too much silence. Resampling...')
                continue
            print('Voiced sample found for trg_audio')
            diff = np.average(trg_midi_continuous_clip[voiced]) - avg_src_midi
            octave = 12
            number_of_octave_diff = round(diff / octave)
            print(
                "avg_trg_pitch",
                np.average(trg_midi_continuous_clip),
                "avg_src_midi",
                avg_src_midi,
                "octave_diff",
                number_of_octave_diff,
            )
            matched_singer_found = True
            break
        else:
            # negative start_of_chunk_idx value means no good matching pitch.
            # Therefore log attempt and try search for new trg clip again
            attempt_num += 1
            if attempt_num >= track_search_tolerance:
                raise NoMatchError(
                    f"No matching pitches after searching {attempt_num} target candidates"
                )

    trg_spec_clip, _ = fix_feat_length(
        trg_spec_feats, this_train_params.window_timesteps, offset=start_of_chunk_idx
    )

    return (
        trg_spec_clip,
        trg_midi_onehot_clip,
        trg_rand_gend_int,
        start_of_chunk_idx,
        trg_path,
        number_of_octave_diff,
    )


def pitch_matched_src_trg(
    src_gender, trg_gender, voiced_percent_tolerance, pitch_match=True
):
    matching_target_found = False

    while not matching_target_found:

        src_path, src_rand_gend_int = get_song_path(src_gender)
        #     src_path = '/import/c4dm-02/bdoc3/spmel/damp_qianParams/test/434587164/434587164_2141814685.npy'

        src_spec_feats, src_midi_continous = get_feats(src_path)
        src_midi_continous = np.round(src_midi_continous)
        src_rand_ts = random.randint(
            0, len(src_spec_feats) - this_train_params.window_timesteps - 1
        )
        src_spec_clip, _ = fix_feat_length(
            src_spec_feats, this_train_params.window_timesteps, offset=src_rand_ts
        )
        src_midi_continuous_clip, _ = fix_feat_length(
            src_midi_continous, this_train_params.window_timesteps, offset=src_rand_ts
        )

        # ensure we do not include avereaging over zero values which represents
        voiced = src_midi_continuous_clip != 0
        if (sum(voiced) / len(voiced)) < voiced_percent_tolerance:
            print(f'Sampled clip from file {src_path} at index {src_rand_ts} contained too much silence. Resampling...')
            continue
        print('Voiced sample found for src_audio')
        avg_src_midi = round(np.average(src_midi_continuous_clip[voiced]))
        try:
            spec_pitch_gendint_randts_path_pitchdiff = matching_pitch_clip(
                trg_gender,
                avg_src_midi,
                src_path,
                pitch_match,
                voiced_percent_tolerance=voiced_percent_tolerance,
                avoid_same_singer_conv=config.avoid_same_singer_conv
            )

            (
                trg_spec_clip,
                trg_midi_onehot_clip,
                trg_rand_gend_int,
                trg_rand_ts,
                trg_path,
                octave_pitch_diff,
            ) = spec_pitch_gendint_randts_path_pitchdiff
            if config.update_gender_list:
                src_lst_idx = gender_separated_lists[src_gender].index(
                    os.path.basename(src_path).split("_")[0]
                )
                gender_separated_lists[src_gender].pop(src_lst_idx)
                trg_lst_idx = gender_separated_lists[trg_gender].index(
                    os.path.basename(trg_path).split("_")[0]
                )
                gender_separated_lists[trg_gender].pop(trg_lst_idx)

            matching_target_found = True
        except NoMatchError as e:
            print(NoMatchError)
            continue

    if not pitch_match:
        # then we must transpose the pitch contours to be a believable voice
        # conversion to a vocalist exhibiting different octaves
        print("avg_src_midi", avg_src_midi)
        adjusted_src_midi_continuous_clip = src_midi_continuous_clip.copy()
        adjusted_src_midi_continuous_clip[voiced] = adjusted_src_midi_continuous_clip[
            voiced
        ] + (
            12 * octave_pitch_diff
        )  # plus becomes minus if octave_pitch_diff is neg
        print(
            "adjusted_avg_src_midi",
            round(np.average(adjusted_src_midi_continuous_clip[voiced])),
        )

    else:
        adjusted_src_midi_continuous_clip = src_midi_continuous_clip


    src_midi_onehot_clip = midi_as_onehot(
        adjusted_src_midi_continuous_clip, this_train_params.midi_range
    )

    src_data = src_spec_clip, src_midi_onehot_clip, src_rand_ts, src_path
    trg_data = trg_spec_clip, trg_midi_onehot_clip, trg_rand_ts, trg_path

    return src_data, trg_data


def parse_data(data):
    clipped_spec, clipped_pitches, rand_ts, path = data
    voice_id = os.path.basename(path).split("_")[0]
    sie_emb = subset_metadata[subset_names.index(voice_id)][1]
    arr_list = [clipped_spec, clipped_pitches, sie_emb]
    tns_list = [
        container_to_tensor(arr, add_batch_dim=True, device=device) for arr in arr_list
    ]
    clipped_spec, clipped_pitches, sie_emb = tns_list
    fn = os.path.basename(path)
    return clipped_spec, clipped_pitches, sie_emb, rand_ts, fn


def get_fn_string(src_fn, trg_fn, src_rand_ts):
    model_str = model_names[model_name]
    src_str = gend_dict[src_gender] + src_fn.split("_")[0]
    trg_str = gend_dict[trg_gender] + trg_fn.split("_")[0]
    return model_str + f"_{src_str}" + \
        f"_timestep{src_rand_ts}" + f"_{trg_str}"


def write_to_disk(fn_name, feats):
    # synthesis
    dst_dir = os.path.join(trg_dir, fn_name)
    plt.imshow(feats)
    plt.title(fn_name)
    if os.path.exists(dst_dir):
        return
    plt.savefig(dst_dir + '.png')
    print(f"Synthesizing converted audio to disk at: {dst_dir}")
    waveform = wavegen(synth_model, this_cuda, c=feats)
    sf.write(dst_dir + '.wav', waveform, samplerate=SVC_feat_params["sr"])


# use model_name to build models, metadata, and training param variables
def make_gen_dependancies(model_name):
    this_svc_model_dir = os.path.join(config.saved_models_dir, model_name)
    ckpt_path = os.path.join(this_svc_model_dir, ckpt)

    # important and confusing way of reloading a params py script and its
    # variables
    global this_train_params
    if "this_train_params" not in globals():
        sys.path.insert(1, this_svc_model_dir)
        import this_train_params
    else:
        del sys.path[1]
        sys.path.insert(1, this_svc_model_dir)
        importlib.reload(this_train_params)

    if not hasattr(this_train_params, "pkl_fn_extras"):
        this_train_params.pkl_fn_extras = ""

    global SVC_feat_params
    SVC_feat_dir = os.path.basename(this_train_params.SVC_feat_dir)
    with open(os.path.join(sie_dir, feat_dir, "feat_params.yaml")) as File:
        SVC_feat_params = yaml.load(File, Loader=yaml.FullLoader)

    pitch_dim = len(this_train_params.midi_range) + 1
    if not this_train_params.SVC_pitch_cond:
        pitch_dim = 0

    print(f"Loading model: {model_name}")
    global G
    G, _, _ = utils.setup_gen(
        this_train_params.dim_neck,
        this_train_params.dim_emb,
        this_train_params.dim_pre,
        this_train_params.sample_freq,
        80,
        pitch_dim,
        device,
        ckpt_path,
        this_train_params.adam_init,
    )

    sie_model_name = os.path.basename(this_train_params.SIE_model_path)
    SIE_dataset_name = os.path.basename(this_train_params.SIE_feat_dir)
    metadata_path = os.path.join(
        config.metadata_root_dir,
        sie_model_name,
        feat_dir,
        config.subset,
        f"voices_metadata{this_train_params.pkl_fn_extras}.pkl",
    )
    global subset_metadata
    subset_metadata = pickle.load(open(metadata_path, "rb"))
    global subset_names
    subset_names = [metad[0] for metad in subset_metadata]


def collect_feats_convert_synth(
        src_gender, trg_gender, self_convert, pitch_match):
    """Collect source & target features, convert timbre of src to trg if pitch ranges matches"""
    src_data, trg_data = pitch_matched_src_trg(
        src_gender, trg_gender, voiced_percent_tolerance, pitch_match=pitch_match
    )
    (
        src_clipped_spec_tns,
        src_clipped_pitch_tns,
        src_emb_tns,
        src_randts,
        src_fn,
    ) = parse_data(src_data)
    (
        trg_clipped_spec_tns,
        _,
        trg_emb_tns,
        trg_randts,
        trg_fn,
    ) = parse_data(trg_data)

    original_src_feats = tensor_to_array(src_clipped_spec_tns)
    original_trg_feats = tensor_to_array(trg_clipped_spec_tns)

    if not this_train_params.SVC_pitch_cond:
        src_clipped_pitch_tns = None

    # conversion
    if self_convert:
        trg_emb_tns = src_emb_tns
    _, converted_feats, _, _, _ = G(
        src_clipped_spec_tns, src_emb_tns, trg_emb_tns, src_clipped_pitch_tns
    )
    converted_feats = tensor_to_array(converted_feats)
    converted_name = get_fn_string(src_fn, trg_fn, src_randts)

    if not os.path.exists(trg_dir):
        os.makedirs(trg_dir)
    if config.test_mode:
        converted_feats = converted_feats[:1]
        original_trg_feats = original_trg_feats[:1]
        original_src_feats = original_src_feats[:1]

    write_to_disk(converted_name, converted_feats)
    if config.resynth_orgs:
        write_to_disk(converted_name + "_source", original_src_feats)
        write_to_disk(converted_name + "_target", original_trg_feats)


parser = argparse.ArgumentParser()
parser.add_argument("-wc", "--which_cuda", type=int, default=0)
parser.add_argument("-cl", "--cond_list", nargs="+", default=[(0, 0, 1), (0, 1, 0)])
parser.add_argument(
    "-sc", "--self_convert", type=int, default=0
)  # caution: self-convert means get autovc to recreate its input
parser.add_argument(
    "-ro", "--resynth_orgs", type=int, default=0
)  # synthesize the source and target audio from unprocessed mels
parser.add_argument("-tm", "--test_mode", type=int, default=0)
parser.add_argument("-pm", "--pitch_match", type=int, default=0, help="Requirement for making sure avg pitches match")
parser.add_argument("-ass", "--avoid_same_singer_conv", type=int, default=1, help='If the src and trg singer are same, resample')
parser.add_argument("-ugl", "--update_gender_list", type=int, default=0, help="Whether to remove paths from list after they're used")
parser.add_argument("-mn", "--model_name", type=str, default="damp_mel_Size1-avgEmbs_EmbLossL2loss__-default_model-deletable")
# parser.add_argument("-mn", "--model_name", type=str, default="damp_mel_Size0.25-avgEmbs_EmbLossL1loss__-bestPerformingSIE_mel80-Sep2023Version")
parser.add_argument("-ae", "--audio_ext", type=str, default=".m4a")
parser.add_argument("-s", "--subset", type=str, default="val")
parser.add_argument("-td", "--trg_dir", type=str, default=os.path.join(root_dir, 'converted_audio'))
parser.add_argument("-smd", "--saved_models_dir", type=str, default=os.path.join(root_dir, 'models'))
parser.add_argument("-fd", "--feat_path", type=str, default=os.path.join(sie_dir, 'damp_example_feats'))
parser.add_argument("-pd", "--pitch_dir", type=str, default=os.path.join(root_dir, 'damp_example_pitch'))
parser.add_argument("-ad", "--audio_dir", type=str, default=os.path.join(sie_dir, 'damp_example_audio'))
parser.add_argument("-mrd", "--metadata_root_dir", type=str, default=os.path.join(super_dir, 'voice_embs_visuals_metadata'))
parser.add_argument("-gc", "--gender_csv", type=str, default=os.path.join(sie_dir, 'example_metadata.csv'))

config = parser.parse_args()
feat_dir = os.path.basename(config.feat_path)

if config.trg_dir is None:
    trg_dir = str(random.randint(1000, 9998))
else:
    trg_dir = config.trg_dir

if config.test_mode:
    trg_dir = "9999"

num_model_conds = 4
num_convert_conds = 4
this_cuda = config.which_cuda
voiced_percent_tolerance = 0.6

model_names = {
    config.model_name: config.model_name,
    # "damp_mel_Size1-avgEmbs_EmbLossL1loss_svcPitchCond_-bestPerformingSIE_mel80-FullDatasetWithCrepePitch": "M-SieLr-L1-CrepeCond-SngFullDs",
    # 'vctk_mel_Size0.25-avgEmbs_L2loss__-autoVc_pretrainedOnVctk_Mels80-':'M-L2-Spk'
    #    'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-':'M---Sng',
    #    'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2':'M-E-Sng',
    #    'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80-':'M-C-Spk',
    #    'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2':'M-C-Sng'
}

codes_to_model = {
    "0": config.model_name,
    # '0': 'vctk_mel_Size0.25-avgEmbs_L2loss__-autoVc_pretrainedOnVctk_Mels80-'
    # '0':'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-',
    #   '1':'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2',
    #   '2':'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80    ',
    #   '3':'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2'
}

# ckpt = 'ckpt_1000000.pt'
ckpt = "saved_model.pt"

wavenet_model_params = os.path.join(root_dir, "checkpoint_step001000000_ema.pth")
device = f"cuda:{this_cuda}"
gender_conds = [(i, j) for i in range(2) for j in range(2)]
gend_dict = {0: "M", 1: "F"}
gender_separated_lists = get_gender_lists(os.path.join(config.feat_path, config.subset), config.gender_csv)

# SETUP SYNTH MODEL
print("building synth model")
cudnn.benchmark = True
torch.cuda.set_device(device)
synth_model = build_model().to(device)
checkpoint = torch.load(wavenet_model_params, map_location="cpu")
new_state_dict = OrderedDict()
for k, v in checkpoint["state_dict"].items():
    new_state_dict[k] = v.cuda(device)
synth_model.load_state_dict(new_state_dict)


# START CONVERSION PROCESS
if len(config.cond_list) != 0:
    for conds_str in config.cond_list:
        model_name = codes_to_model[str(conds_str[0])]
        make_gen_dependancies(model_name)
        src_gender = int(conds_str[1])
        trg_gender = int(conds_str[2])
        collect_feats_convert_synth(
            src_gender, trg_gender, config.self_convert, config.pitch_match
            )
else:
    for j, model_name in enumerate(model_names.keys()):
        make_gen_dependancies(model_name)

        for i, (src_gender, trg_gender) in enumerate(gender_conds):
            print(f"Getting feats for model {model_names[model_name]} under src-trg gender condition: {(src_gender, trg_gender)}")
            collect_feats_convert_synth(
                src_gender, trg_gender, config.self_convert, config.pitch_match
            )

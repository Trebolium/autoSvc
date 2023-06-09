import numpy as np
import soundfile as sf
import random, argparse, pickle, torch, yaml, importlib
from torch.backends import cudnn
from collections import OrderedDict

import sys, os

if os.path.abspath("../my_utils") not in sys.path:
    sys.path.insert(1, os.path.abspath("../my_utils"))
from my_arrays import fix_feat_length, find_runs
from neural.torch_utils import container_to_tensor, tensor_to_array
from my_os import recursive_file_retrieval
from my_audio.pitch import midi_as_onehot
from my_audio.midi import freqToMidi

from .. import utils
from convert_utils import (
    get_gender_lists,
    best_pitch_matching_idx,
    get_relevant_avg_pitches,
    NoMatchError,
)
from synthesis import build_model, wavegen


class Obj:
    pass


def get_feats(path):
    spec_feats = np.load(path)
    fn = os.path.basename(path)
    if "crepe" in pitch_dir:
        crepe_data = np.load(
            os.path.join(pitch_dir, subset, fn.split("_")[0], fn[:-4] + ".npz")
        )
        pitches = crepe_data["arr_0"]
        conf = crepe_data["arr_1"]
        unvoiced = (
            conf < 0.5
        )  # determined by looking at pitch and conf contours against audio in sonic visualizer
        midi_contour = freqToMidi(pitches)
    elif "world" in pitch_dir:
        world_feats = np.load(os.path.join(pitch_dir, subset, fn.split("_")[0], fn))
        pitches = world_feats[:, -2:]
        midi_contour = pitches[:, 0]
        unvoiced = pitches[:, 1].astype(int) == 1
    else:
        raise NotImplementedError

    midi_contour[unvoiced] = 0
    return spec_feats, midi_contour


def get_song_path(gender):
    gender_list = gender_separated_lists[gender]
    rand_int = random.randint(0, len(gender_list) - 1)
    name = gender_list[rand_int]
    song_list = os.listdir(os.path.join(SVC_data_dir, name))
    song_name = random.choice(song_list)
    song_path = os.path.join(SVC_data_dir, name, song_name)

    return song_path, rand_int


def matching_pitch_clip(
    trg_gender,
    avg_src_midi,
    src_path,
    pitch_match,
    track_search_tolerance=11,
    voiced_percent_tolerance=0.7,
):
    matched_singer_found = False
    attempt_num = 0
    while matched_singer_found == False:
        # choose random target
        trg_path, trg_rand_gend_int = get_song_path(trg_gender)
        if os.path.dirname(trg_path) == os.path.dirname(src_path):
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
                0, len(trg_midi_continuous[: -this_train_params.window_timesteps])
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
                continue
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
            # negative start_of_chunk_idx value means no good matching pitch. Therefore log attempt and try search for new trg clip again
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
        src_midi_onehot = midi_as_onehot(
            src_midi_continous, this_train_params.midi_range
        )

        src_rand_ts = random.randint(
            0, len(src_spec_feats) - this_train_params.window_timesteps - 1
        )
        #     src_rand_ts = 2981

        src_spec_clip, _ = fix_feat_length(
            src_spec_feats, this_train_params.window_timesteps, offset=src_rand_ts
        )
        src_midi_continuous_clip, _ = fix_feat_length(
            src_midi_continous, this_train_params.window_timesteps, offset=src_rand_ts
        )
        # ensure we do not include avereaging over zero values which represents unvoiced
        voiced = src_midi_continuous_clip != 0
        if (sum(voiced) / len(voiced)) < voiced_percent_tolerance:
            continue
        avg_src_midi = round(np.average(src_midi_continuous_clip[voiced]))

        print(
            f"src_song: {os.path.basename(src_path)}, rand_int: {src_rand_ts}, src_gend: {gend_dict[src_gender]}, avg_src_midi: {avg_src_midi}"
        )

        try:
            spec_pitch_gendint_randts_path_pitchdiff = matching_pitch_clip(
                trg_gender,
                avg_src_midi,
                src_path,
                pitch_match,
                voiced_percent_tolerance=voiced_percent_tolerance,
            )

            (
                trg_spec_clip,
                trg_midi_onehot_clip,
                trg_rand_gend_int,
                trg_rand_ts,
                trg_path,
                octave_pitch_diff,
            ) = spec_pitch_gendint_randts_path_pitchdiff
            try:
                src_lst_idx = gender_separated_lists[src_gender].index(
                    os.path.basename(src_path).split("_")[0]
                )
                gender_separated_lists[src_gender].pop(src_lst_idx)
                trg_lst_idx = gender_separated_lists[trg_gender].index(
                    os.path.basename(trg_path).split("_")[0]
                )
                gender_separated_lists[trg_gender].pop(trg_lst_idx)
            except Exception as e:
                print(e)
                pdb.set_trace()
            matching_target_found = True
        except NoMatchError as e:
            continue

    if not pitch_match:
        # then we must transpose the pitch contours to be a believable voice conversion to a vocalist exhibiting different octaves
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

    # pdb.set_trace()

    src_midi_onehot_clip = midi_as_onehot(
        adjusted_src_midi_continuous_clip, this_train_params.midi_range
    )

    src_data = src_spec_clip, src_midi_onehot_clip, src_rand_ts, src_path
    trg_data = trg_spec_clip, trg_midi_onehot_clip, trg_rand_ts, trg_path

    return src_data, trg_data


# def get_fn_string(src_fn, trg_fn, src_rand_ts, model_str, gen_pair):
#     src_gender, trg_gender = gen_pair
#     src_str = gend_dict[src_gender] +src_fn.split('_')[0]
#     trg_str = gend_dict[trg_gender] +trg_fn.split('_')[0]
#     return model_str +f'_{src_str}' +f'_timestep{src_rand_ts}' +f'_{trg_str}'


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
    return model_str + f"_{src_str}" + f"_timestep{src_rand_ts}" + f"_{trg_str}"


def write_to_disk(fn_name, feats):
    # synthesis
    dst_dir = os.path.join(converted_voices_dir, fn_name) + ".wav"
    if os.path.exists(dst_dir):
        return
    print(f"Synthesizing converted audio at: {dst_dir}")
    waveform = wavegen(synth_model, this_cuda, c=feats)
    # pdb.set_trace()
    sf.write(dst_dir, waveform, samplerate=SVC_feat_params["sr"])


# use model_name to build models, metadata, and training param variables
def make_gen_dependancies(model_name):
    this_svc_model_dir = os.path.join(saved_models_dir, model_name)
    ckpt_path = os.path.join(this_svc_model_dir, ckpt)

    # important and confusing way of reloading a params py script and its variables
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
    with open(os.path.join(this_train_params.SVC_feat_dir, "feat_params.yaml")) as File:
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
        metadata_root_dir,
        sie_model_name,
        SIE_dataset_name,
        subset,
        f"voices_metadata{this_train_params.pkl_fn_extras}.pkl",
    )
    global subset_metadata
    subset_metadata = pickle.load(open(metadata_path, "rb"))
    global subset_names
    subset_names = [metad[0] for metad in subset_metadata]


def collect_feats_convert_synth(src_gender, trg_gender, self_convert, pitch_match):
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
        trg_clipped_pitch_tns,
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

    if not os.path.exists(converted_voices_dir):
        os.makedirs(converted_voices_dir)
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
parser.add_argument("-td", "--trg_dir", type=str, default=None)
parser.add_argument("-cl", "--cond_list", nargs="+", default=[])
parser.add_argument(
    "-sc", "--self_convert", type=int, default=0
)  # caution: self-convert means get autovc to recreate its input
parser.add_argument(
    "-ro", "--resynth_orgs", type=int, default=0
)  # synthesize the source and target audio from unprocessed mels
parser.add_argument("-tm", "--test_mode", type=int, default=0)
parser.add_argument("-pm", "--pitch_match", type=int, default=1)
config = parser.parse_args()

if config.trg_dir == None:
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
    "damp_mel_Size1-avgEmbs_EmbLossL1loss_svcPitchCond_-bestPerformingSIE_mel80-FullDatasetWithCrepePitch": "M-SieLr-L1-CrepeCond-SngFullDs",
    # 'vctk_mel_Size0.25-avgEmbs_L2loss__-autoVc_pretrainedOnVctk_Mels80-':'M-L2-Spk'
    #    'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-':'M---Sng',
    #    'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2':'M-E-Sng',
    #    'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80-':'M-C-Spk',
    #    'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2':'M-C-Sng'
}

codes_to_model = {
    "0": "damp_mel_Size1-avgEmbs_EmbLossL1loss_svcPitchCond_-bestPerformingSIE_mel80-FullDatasetWithCrepePitch",
    # '0': 'vctk_mel_Size0.25-avgEmbs_L2loss__-autoVc_pretrainedOnVctk_Mels80-'
    # '0':'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-',
    #   '1':'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2',
    #   '2':'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80    ',
    #   '3':'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2'
}

# ckpt = 'ckpt_1000000.pt'
ckpt = "saved_model.pt"
saved_models_dir = "/homes/bdoc3/my_data/autovc_models/autoSvc"
subset = "test"
SVC_data_dir = "/import/c4dm-02/bdoc3/spmel/damp_qianParams/" + subset
metadata_root_dir = "/homes/bdoc3/my_data/voice_embs_visuals_metadata"
# pitch_dir = '/import/c5dm-02/bdoc3/world_data/damp_80_16ms'
pitch_dir = "/homes/bdoc3/my_data/crepe_data/damp"
wavenet_model_params = (
    "/homes/bdoc3/my_data/autovc_models/checkpoint_step001000000_ema.pth"
)
converted_voices_dir = (
    "/homes/bdoc3/my_data/audio_data/output_audio/listeningTest3audio/" + trg_dir
)
device = f"cuda:{this_cuda}"
gender_conds = [(i, j) for i in range(2) for j in range(2)]
gend_dict = {0: "M", 1: "F"}

gender_separated_lists = get_gender_lists(SVC_data_dir)

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


### START CONVERSION PROCESS

if len(config.cond_list) != 0:
    for conds_str in config.cond_list:
        model_name = codes_to_model[conds_str[0]]
        make_gen_dependancies(model_name)
        src_gender = int(conds_str[1])
        trg_gender = int(conds_str[2])
        collect_feats_convert_synth(src_gender, trg_gender, config.self_convert)
else:
    for j, model_name in enumerate(model_names.keys()):
        make_gen_dependancies(model_name)

        for i, (src_gender, trg_gender) in enumerate(gender_conds):
            print(f"Getting feats for condition: {j, i}")
            collect_feats_convert_synth(
                src_gender, trg_gender, config.self_convert, config.pitch_match
            )

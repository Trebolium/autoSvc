""" Currently does not cater for:
        non-mel conversions
            with/without apers
        without avg embs
            different feats for SIE and SVC
        de-normalisation
    Untested for:
        pitch-conditioning
"""

import os, random, sys, pdb
import pickle, yaml
import torch
import numpy as np
import librosa
from librosa.filters import mel
from statistics import mean
from tqdm import tqdm

pdb.set_trace()

from convert_params import *

if os.path.abspath("../my_utils") not in sys.path:
    sys.path.insert(
        1, os.path.abspath("../my_utils")
    )  # only need to do this once in the main script
from my_csv import list_to_csvfile, csvfile_to_list
from my_os import recursive_file_retrieval
from my_arrays import fix_feat_length
from my_audio.mel import raw_audio_to_mel_autovc
from my_audio.world import get_world_feats
from my_audio.midi import midiToFreq
from neural.model_mod import checkpoint_model_optim_keys, set_optimizer_device

sys.path.insert(1, os.path.abspath("./"))
import model_vc
import utils


def fp_gendered_list_from_vctk_txt():
    # collect even number of gender singers using the VCTK documentation
    print("use info txt")
    # convert txt format into format readable by row and elements
    with open(
        "/import/c4dm-datasets/VCTK-Corpus-0.92/speaker-info.txt", "r"
    ) as in_file:
        stripped = (line.strip() for line in in_file)
        males = []
        females = []
        # load file contents into male/female list
        header = [i for i in next(stripped).split(" ") if i != ""]
        print(header)
        for row in stripped:
            row_elements = [i for i in row.split(" ") if i != ""]
            if row_elements[0] in subset_names:
                if row_elements[2] == "M":
                    males.append([row_elements[0], "m"])

                else:
                    females.append([row_elements[0], "f"])

    # choose a number of singers from these samples
    each_gender = num_singers // 2
    try:
        chosen_spkrs = random.sample(males, k=each_gender) + random.sample(
            females, k=each_gender
        )
    except ValueError as e:
        Exception(
            f"No entries in gender lists led to error {e}. Consider checking that the contents of the subset_names match those of the VCTK text file"
        )

    # choose random utterances from these singers to use
    list_for_csv = []
    for spkr, gender in chosen_spkrs:
        _, _, fps = next(os.walk(os.path.join(SVC_feat_dir, subset, str(spkr))))
        fp = random.choice(fps)
        fn = os.path.basename(fp)
        # fp = os.path.join(SVC_feat_dir, subset, str(spkr), fn)
        list_for_csv.append([fn[: -len(npy_ext)], gender])

    # save to csv file and make chosen_spkrs the uttrs list now
    list_for_csv = [["filename", "gender"]] + list_for_csv
    list_to_csvfile(list_for_csv, csv_example_fp)
    chosen_spkrs = list_for_csv[1:]
    return chosen_spkrs


def SVC_data_from_csv(csv_example_fp):
    chosen_spkrs = csvfile_to_list(csv_example_fp)
    if len(chosen_spkrs[0]) == 2:
        fpgl_missing_timestep = True
    else:
        fpgl_missing_timestep = False
    return chosen_spkrs, fpgl_missing_timestep


# check use_aper_feats boolean to produce total num feats being used for training
# this is ignored in the case of mels as they don't have aper aspect
def determine_dim_size(
    SIE_params, SVC_params, SIE_feat_dir, SVC_feat_dir, use_aper_feats
):
    if use_aper_feats:
        if (
            "world" in SIE_feat_dir
        ):  # requires no else a mel feature set means leave num_feats as is
            SIE_params["num_feats"] = (
                SIE_params["num_harm_feats"] + SIE_params["num_aper_feats"]
            )
        if "world" in SVC_feat_dir:
            SVC_params["num_feats"] = (
                SVC_params["num_harm_feats"] + SVC_params["num_aper_feats"]
            )

    else:
        if "world" in SIE_feat_dir:
            SIE_params["num_feats"] = SIE_params["num_harm_feats"]
        if "world" in SVC_feat_dir:
            SVC_params["num_feats"] = SVC_params["num_harm_feats"]

    if "mel" in SIE_feat_dir:
        SIE_params["num_feats"] = SIE_params["num_harm_feats"]
    if "mel" in SVC_feat_dir:
        SVC_params["num_feats"] = SVC_params["num_harm_feats"]

    return SIE_params, SVC_params


def calculate_transposed_pitch(pitch_for_trans, src_avg_pitch, trg_avg_pitch):
    avg_pitch_diff = src_avg_pitch - trg_avg_pitch

    if avg_pitch_diff > 0:  # then src is higher than trg and might have to go down
        src_avg_octaves = [src_avg_pitch] + [
            src_avg_pitch * (1 / 2 * i) for i in range(1, 8)
        ]

    else:  # then src is lower than trg and might have to go up
        src_avg_octaves = [src_avg_pitch] + [
            src_avg_pitch * (2 * i) for i in range(1, 8)
        ]

    closest_octave_value = min(src_avg_octaves, key=lambda x: abs(x - trg_avg_pitch))
    transposing_value = closest_octave_value / src_avg_pitch
    transed_pitch = pitch_for_trans.copy()
    transed_pitch[:, 0] = transed_pitch[:, 0] * transposing_value
    return transed_pitch


def setup_variables():
    global SVC_feat_params
    global SIE_feat_params
    global SVC_is_mel
    global SVC_mel_filter
    global SVC_min_level
    global SVC_hop_size
    global SIE_dataset_name
    global SVC_dataset_name
    global this_model_dir

    with open(os.path.join(SVC_feat_dir, "feat_params.yaml")) as File:
        SVC_feat_params = yaml.load(File, Loader=yaml.FullLoader)
    with open(os.path.join(SIE_feat_dir, "feat_params.yaml")) as File:
        SIE_feat_params = yaml.load(File, Loader=yaml.FullLoader)

    # create varialbes based on features
    if "mel" in SVC_feat_dir.lower():
        SVC_is_mel = True
        SVC_mel_filter = mel(
            SVC_feat_params["sr"],
            SVC_feat_params["fft_size"],
            fmin=SVC_feat_params["fmin"],
            fmax=SVC_feat_params["fmax"],
            n_mels=SVC_feat_params["num_harm_feats"],
        ).T
        SVC_min_level = np.exp(-100 / 20 * np.log(10))
        SVC_hop_size = int(
            (SVC_feat_params["frame_dur_ms"] / 1000) * SVC_feat_params["sr"]
        )

    elif "world" in SVC_feat_dir.lower():
        SVC_is_mel = False

    else:
        Exception("Model name does not describe which features it used. Amend this.")

    SIE_dataset_name = os.path.basename(SIE_feat_dir)
    SVC_dataset_name = os.path.basename(SVC_feat_dir)
    this_model_dir = os.path.join(
        saved_models_dir, os.path.basename(vc_dir), svc_model_name
    )


if __name__ == "__main__":
    # setup variables
    random.seed(random_seed)
    loss_device = torch.device("cpu")
    setup_variables()
    metadata_path = os.path.join(
        metadata_root_dir,
        sie_model_name,
        f"{SIE_dataset_name}_{subset}_singers_metadata.pkl",
    )
    subset_metadata = pickle.load(open(metadata_path, "rb"))
    subset_names = [metad[0] for metad in subset_metadata]
    # if 'damp' in ds_name:
    #     csv_example_fp = os.path.join(csv_dir, 'damp', f'damp_{subset}_examples.csv')
    # elif 'vctk' in ds_name:
    #     csv_example_fp = os.path.join(csv_dir, 'vctk', f'vctk_{subset}_examples.csv')

    # if using a model generated by AutoVC author's script and specs
    if os.path.basename(vc_dir) == "basic_autovc":
        checkpoint_path = os.path.join(
            this_model_dir, "ckpts", "ckpt_" + str(ckpt_iters) + ".pth.tar"
        )
        config = pickle.load(
            open(os.path.join(this_model_dir, svc_model_name, "config.pkl"), "rb")
        )
        sample_freq = config.freq
        dim_neck = config.dim_neck
        dim_emb = config.dim_emb
        dim_pre = config.dim_pre
        SVC_feat_dir = config.spmel_dir
        SIE_feat_dir = SVC_feat_dir  # hack for quick coding - separate feats shouldn't exists for basic_autovc
        window_timesteps = config.len_crop
        use_aper_feats = False
        use_avg_singer_embs = True
        adam_init = 0.0001
        pitch_dir = "../singer-identity-encoder/example_pitch"

    # else use our own saved train parameters to load model
    elif os.path.basename(vc_dir) == "autoSvc":
        # import models path as autosvc uses param py files
        sys.path.insert(1, this_model_dir)
        from this_train_params import *

        checkpoint_path = os.path.join(this_model_dir, "saved_model.pt")

    # use new variables taken from train params file to generate new variables
    if (SVC_feat_dir != SIE_feat_dir) and not use_avg_singer_embs:
        raise Exception(
            "Code is not preppared to handle 2 difference inputs while not using avg embs"
        )

    SIE_feat_params, SVC_feat_params, determine_dim_size(
        SIE_feat_params, SVC_feat_params, SIE_feat_dir, SVC_feat_dir, use_aper_feats
    )

    if not use_avg_singer_embs:
        sie, sie_num_feats_used = utils.setup_sie(
            device, loss_device, os.path.join(sie_dir, sie_model_name, adam_init)
        )

    # Load Generator model
    if os.path.basename(vc_dir) == "basic_autovc":
        G = (
            model_vc.Generator(dim_neck, dim_emb, dim_pre, sample_freq)
            .eval()
            .to(device)
        )
    # if autosvc used, a lot more options available
    elif os.path.basename(vc_dir) == "autoSvc":
        SVC_num_feats = SVC_feat_params["num_feats"]

        if SVC_pitch_cond:
            pitch_dim = len(midi_range) + 1

        else:
            pitch_dim = 0

        G, _, _ = utils.setup_gen(
            dim_neck,
            dim_emb,
            dim_pre,
            sample_freq,
            SVC_num_feats,
            pitch_dim,
            device,
            checkpoint_path,
            adam_init,
        )

    # load up checkpoints and set device for model components
    g_checkpoint = torch.load(checkpoint_path)
    model_key, optim_key = checkpoint_model_optim_keys(g_checkpoint)
    G.load_state_dict(g_checkpoint[model_key])
    g_optimizer = torch.optim.Adam(G.parameters(), 0.0001)
    g_optimizer.load_state_dict(g_checkpoint[optim_key])
    g_optimizer = set_optimizer_device(g_optimizer, device)
    G.to(device)

    # get list of singers
    if os.path.exists(csv_example_fp):
        chosen_spkrs, fpgl_missing_timestep = SVC_data_from_csv(csv_example_fp)

    else:
        raise FileExistsError(f"{csv_example_fp} file not found.")
        # if 'vctk' in SVC_dataset_name:
        #     chosen_spkrs = fp_gendered_list_from_vctk_txt()
        #     fpgl_missing_timestep = False

        # elif 'damp' in SVC_dataset_name:
        #     raise Exception(f'No csv file named {SVC_dataset_name}_{subset}_examples.csv exists for determining example gender-balanced audio snippets. Make one manually to proceed')

    # collect features for each singer: name, spec feats, pitch feats, mean pitch
    chosen_spkr_name_feats = []
    org_audio_paths = []
    for entry in chosen_spkrs:
        if fpgl_missing_timestep:
            uttr_name, gender = entry
        else:
            uttr_name, gender, _ = entry
        voice_dir = uttr_name.split("_")[0]
        _, fps = recursive_file_retrieval(os.path.join(pitch_dir, subset, voice_dir))
        track_pitch_means = []
        for fp in tqdm(fps):
            pitch_info = np.load(fp)[:, -2:]
            contour = pitch_info[:, 0]
            voiced = (
                pitch_info[:, 1].astype(int) == 0
            )  # remove the interpretted values generated because of unvoiced sections
            track_pitch_means.append(np.mean(contour[voiced]))
        singer_pitch_mean = mean(track_pitch_means)
        audio_path = os.path.join(audio_root, voice_dir, uttr_name + audio_ext)
        org_audio_paths.append(audio_path)
        SVC_feats = np.load(
            os.path.join(SVC_feat_dir, subset, voice_dir, uttr_name + npy_ext)
        )
        SIE_feats = np.load(
            os.path.join(SIE_feat_dir, subset, voice_dir, uttr_name + npy_ext)
        )
        pitch_feats = np.load(
            os.path.join(pitch_dir, subset, voice_dir, uttr_name + npy_ext)
        )[:, -2:]
        chosen_spkr_name_feats.append(
            (uttr_name, SVC_feats, SIE_feats, pitch_feats, singer_pitch_mean)
        )

    # get feat chunks, relevant sie embs, octave transpositional for conversion
    reconed_name_feats = []
    name_offsets = []
    for i in range(len(chosen_spkr_name_feats)):
        # pdb.set_trace()
        print(f"Processing {chosen_spkr_name_feats[i][0]}")
        src_uttr_name = chosen_spkr_name_feats[i][0]
        svc_feats = chosen_spkr_name_feats[i][1]
        src_sie_feats = chosen_spkr_name_feats[i][2]
        src_pitch = chosen_spkr_name_feats[i][3]
        src_voice_dir = src_uttr_name.split("_")[0]

        # format feat chunks
        if fpgl_missing_timestep:
            src_feats, offset = fix_feat_length(svc_feats, window_timesteps)
        else:
            offset = int(chosen_spkrs[i][2])
            src_feats, _ = fix_feat_length(svc_feats, window_timesteps, offset=offset)
        src_feats = torch.from_numpy(src_feats).to(device).float().unsqueeze(0)
        if SVC_is_mel:
            src_specs = src_feats
        else:
            src_specs = src_feats[:, :, : SVC_feat_params["num_feats"]]
            aper_feats = src_feats[:, :, SVC_feat_params["num_harm_feats"] : -2]
        src_pitch, _ = fix_feat_length(src_pitch, window_timesteps, offset=offset)

        # determine source embs and spec feats
        if use_avg_singer_embs:
            src_emb = subset_metadata[subset_names.index(src_voice_dir)][
                1
            ]  # subset_n and subset_m are paired indices.
            src_emb = torch.from_numpy(src_emb).to(device).float().unsqueeze(0)

        else:
            src_sie_feats = (
                torch.from_numpy(src_sie_feats).to(device).float().unsqueeze(0)
            )
            src_emb = sie(src_sie_feats)

        # convert feats
        for j in range(len(chosen_spkr_name_feats)):
            trg_uttr_name = chosen_spkr_name_feats[j][0]
            trg_sie_feats = chosen_spkr_name_feats[j][2]
            trg_avg_midi = chosen_spkr_name_feats[j][4]
            trg_voice_dir = trg_uttr_name.split("_")[0]

            if use_avg_singer_embs:
                trg_emb = subset_metadata[subset_names.index(trg_voice_dir)][
                    1
                ]  # subset_n and subset_m are paired indices.
                trg_emb = torch.from_numpy(trg_emb).to(device).float().unsqueeze(0)

            else:
                trg_sie_feats = (
                    torch.from_numpy(trg_sie_feats).to(device).float().unsqueeze(0)
                )
                trg_emb = sie(src_sie_feats)

            if SVC_pitch_cond:
                onehot_midi_npy_src = utils.get_onehot_midi(src_pitch, midi_range)
                onehot_midi_tns_src = (
                    torch.from_numpy(onehot_midi_npy_src)
                    .to(device)
                    .float()
                    .unsqueeze(0)
                )
                _, feat_tns_psnt, _, _, _ = G(
                    src_specs, src_emb, trg_emb, onehot_midi_tns_src
                )

            else:
                _, feat_tns_psnt, _, _, _ = G(src_specs, src_emb, trg_emb, None)

            recon_feats = feat_tns_psnt.squeeze(1).squeeze(0).cpu()

            # calculate appropriate octave for target to sing at
            voiced = (
                src_pitch[:, 1].astype(int) == 0
            )  # remove the interpretted values generated because of unvoiced sections
            pitch_contour = src_pitch[:, 0]
            src_avg_midi = np.mean(pitch_contour[voiced])
            src_avg_pitch = midiToFreq(src_avg_midi)
            trg_avg_pitch = midiToFreq(trg_avg_midi)
            trg_pitch = calculate_transposed_pitch(
                src_pitch, src_avg_pitch, trg_avg_pitch
            )

            # determine name and save to feats and fn to list
            fn = src_uttr_name + "_X_" + trg_uttr_name

            if SVC_is_mel:
                reconed_name_feats.append((fn, recon_feats))

            else:
                reconed_name_feats.append((fn, recon_feats, aper_feats, trg_pitch))

        name_offsets.append((src_uttr_name, offset))

    # generate feats for original audio unchanged
    if include_unprocessed_audio:
        for i in range(len(org_audio_paths)):
            org_path = org_audio_paths[i]
            offset = name_offsets[i][1]
            fn = os.path.basename(org_path)
            y, sr = librosa.load(org_path, sr=SVC_feat_params["sr"])
            if SVC_is_mel:
                feats = raw_audio_to_mel_autovc(
                    y, SVC_mel_filter, SVC_min_level, SVC_hop_size, SVC_feat_params
                )
                reconed_name_feats.append((fn[:-4], feats))
            else:
                feats = get_world_feats(y, SVC_feat_params)
                feats, _ = fix_feat_length(feats, window_timesteps, offset=offset)
                feats = torch.from_numpy(feats).cpu().float()
                harm_feats = feats[:, : SVC_feat_params["num_harm_feats"]]
                aper_feats = feats[:, SVC_feat_params["num_harm_feats"] : -2]
                pitch_info = feats[:, -2:]
                reconed_name_feats.append((fn, harm_feats, aper_feats, pitch_info))

    # if offsets not in the text file, put save them to it
    if fpgl_missing_timestep:
        list_for_csv = [["filename", "gender", "chunk_start_timestep"]]
        for i in range(len(chosen_spkrs)):
            spkr, gender = chosen_spkrs[i]
            offset = name_offsets[i][1]
            list_for_csv.append([spkr, gender, offset])
        list_to_csvfile(list_for_csv, csv_example_fp)

    os.makedirs(os.path.dirname(name_feat_npy_list_name), exist_ok=True)

    with open(name_feat_npy_list_name, "wb") as handle:
        pickle.dump(reconed_name_feats, handle)

import os
import torch
import pdb
import csv
from shutil import copyfile

import numpy as np

from model_sie import SingerIdEncoder
from collections import OrderedDict
from model_vc import Generator

from my_audio.pitch import midi_as_onehot
from neural.model_mod import checkpoint_model_optim_keys
from my_os import overwrite_dir





def cont_to_onehot_midi(midi_voicing, midi_range):
    """Convert world pitch info to 1hot midi"""
    midi_contour = midi_voicing[:, 0]
    unvoiced = (
        midi_voicing[:, 1].astype(int) == 1
    )  # remove the interpretted values generated because of unvoiced sections
    midi_contour[unvoiced] = 0
    onehot_midi = midi_as_onehot(midi_contour, midi_range)

    return onehot_midi


"""Currently designed to take model ckpts of 2 slightly different dictionary keys"""


def setup_sie(device, loss_device, SIE_path, adam_init, qians_pretrained_model=False):
    """Build SIE encoder model and optimizer using ckpt pathway"""
    sie_checkpoint = torch.load(
        os.path.join(SIE_path, "saved_model.pt"), map_location="cpu"
    )
    new_state_dict = OrderedDict()

    if qians_pretrained_model:
        model_state = "model_b"
        sie_num_feats_used = sie_checkpoint[model_state][
            "module.lstm.weight_ih_l0"
        ].shape[1]
    else:
        model_state = "model_state"
        sie_num_feats_used = sie_checkpoint[model_state]["lstm.weight_ih_l0"].shape[1]
    sie = SingerIdEncoder(device, loss_device, sie_num_feats_used)

    if qians_pretrained_model:
        new_state_dict["similarity_weight"] = sie.similarity_weight
        new_state_dict["similarity_bias"] = sie.similarity_bias

    for key, val in sie_checkpoint[model_state].items():
        if qians_pretrained_model:
            key = key[7:]  # gets right of the substring 'module'
            if key.startswith("embedding"):
                key = "linear." + key[10:]

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





def setup_gen(
    dim_neck,
    dim_emb,
    dim_pre,
    sample_freq,
    num_feats,
    pitch_dim,
    device,
    svc_ckpt_path,
    adam_init,
):
    """Currently designed to initiate G in two ways, based on train_param variables"""
    G = Generator(
        dim_neck,
        dim_emb,
        dim_pre,
        sample_freq,
        num_feats,
        pitch_dim)
    g_optimizer = torch.optim.Adam(G.parameters(), adam_init)

    if svc_ckpt_path != "":
        g_checkpoint = torch.load(svc_ckpt_path, map_location="cpu")
        model_key, optim_key = checkpoint_model_optim_keys(g_checkpoint)
        for k in g_checkpoint.keys():
            if k.startswith("model"):
                model_key = k
            if k.startswith("optim"):
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
            train_latest_step = g_checkpoint["step"]
        except KeyError as e:
            train_latest_step = 0
    else:
        train_latest_step = 0

    G.to(device)
    return G, g_optimizer, train_latest_step


def new_song_idx(dataset):
    "finds the index for each new song in dataset"
    new_song_idxs = []
    song_idxs = list(range(255))
    for song_idx in song_idxs:
        for ex_idx, ex in enumerate(dataset):
            if ex[1] == song_idx:
                new_song_idxs.append(ex_idx)
                break
    return new_song_idxs


def new_dir_setup(ask, svc_model_dir, svc_model_name):
    """Make new directory, ask whether to overwrite if it already exists"""
    model_dir_path = os.path.join(svc_model_dir, svc_model_name)
    overwrite_dir(model_dir_path, ask)
    os.makedirs(model_dir_path + "/ckpts")
    os.makedirs(model_dir_path + "/generated_wavs")
    os.makedirs(model_dir_path + "/image_comparison")
    os.makedirs(model_dir_path + "/input_tensor_plots")
    files = ["train_params.py"]
    copy_to_new_dir(model_dir_path, files)


def copy_to_new_dir(dst_path, files):
    """Make copy of file in new directory"""
    for file in files:
        dst_file = os.path.join(dst_path, "this_" + file)
        file = os.path.join(os.path.dirname(os.path.abspath(__file__)), file)
        copyfile(file, dst_file)
        with open(dst_file, "r") as file:
            filedata = file.read()
        filedata = filedata.replace(
            "from train_params import *", "from this_train_params import *"
        )
        with open(dst_file, "w") as file:
            file.write(filedata)


def determine_dim_size(
    SIE_params, SVC_params, SIE_feat_dir, SVC_feat_dir, use_aper_feats
):
    """
    Calculate total dim-size of all (not just spectral) features.
    
    Args:
        SIE_params (dict): Details of features sent to SIE encoder input
        SVC_params (dict): Details of features sent to SVC encoder input
        SIE_feat_dir (str): Pathway to features for SIE encoder input
        SVC_feat_dir (str): Pathway to features for SVC encoder input
        use_aper_feats (bool): Determines whether feats include aperiodicity info

    Return:
        dict: updated version of SIE_params
        dict: updated version of SVC_params

    """
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

    try:
        SIE_params["num_feats"]
        SIE_params["num_feats"]
    except KeyError as e:
        raise Exception(
            f"KeyError: {e} \n To fix this error, edit the input features directory (default called 'example_feats' to contain either the string 'mel' or 'world'"
        )

    return SIE_params, SVC_params


def get_damp_gender(
    ignore_unknowns=False,
    csv_path="/homes/bdoc3/my_data/text_data/damp/intonation_metadata.csv",
):
    """
    Get entries from gender csv file, return single list of performer-gender tuples
    """

    f = open(csv_path, "r")
    reader = csv.reader(f)
    header = next(reader)
    singer_meta = [row for row in reader]
    if ignore_unknowns:
        performer_gender_list = [
            (row[0].split("_")[0], row[8]) for row in singer_meta if row[8] != " None"
        ]
    else:
        performer_gender_list = [(row[0].split("_")[0], row[8])
                                 for row in singer_meta]

    return performer_gender_list

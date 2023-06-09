# user-perscribed variables
import os

vc_dir = "/homes/bdoc3/autoSvc"
sie_dir = "/homes/bdoc3/singer-identity-encoder"
svc_model_name = "tinyRun_damp_sieMelSvcMel"
sie_model_name = "bestPerformingSIE_mel80"
subset = "test"
audio_root = "/homes/bdoc3/my_data/audio_data/damp_desilenced_concat"
# audio_root = '/import/c4dm-datasets/VCTK-Corpus-0.92/wav48_silence_trimmed'
# SVC_feat_dir = '/import/c4dm-02/bdoc3/spmel/vctk_qianParams' # MAKE SURE THESE MATCH THOSE OF THIS_TRAIN_PARAMS.PY FOR THE GIVEN SVC MODEL ABOVE
SVC_feat_dir = "../singer-identity-encoder/example_feats"
SIE_feat_dir = "../singer-identity-encoder/example_feats"  # MAKE SURE THESE MATCH THOSE OF THIS_TRAIN_PARAMS.PY FOR THE GIVEN SVC MODEL ABOVE
ds_name = "damp"
audio_ext = ".m4a"
npy_ext = ".npy"
name_feat_npy_list_name = os.path.join(
    "converted_feats", svc_model_name, f"{subset}_name_feat_npy_list.pkl"
)
device = f"cuda:{0}"
SVC_pitch_cond = False
ckpt_iters = None
num_singers = 4
include_unprocessed_audio = False

# constants
# csv_dir = '/homes/bdoc3/my_data/text_data'
csv_example_fp = "./damp_test_examples"
saved_models_dir = "./autosvc_models/"
metadata_root_dir = "../voice_embs_visuals_metadata"
random_seed = 1

model_hidden_size = 768
model_embedding_size = 256
model_num_layers = 3

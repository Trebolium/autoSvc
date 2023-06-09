import pickle
import torch
import time
import datetime
import torch.nn.functional as F
import yaml

import sys, os
if os.path.abspath('../my_utils') not in sys.path: sys.path.insert(1, os.path.abspath('../my_utils'))
from neural.eval import get_accuracy

import utils
from model_vc import Aux_Voice_Classifier
from torch.utils.tensorboard import SummaryWriter
from data_loaders import load_primary_dataloader


which_cuda = 2 #when you change this, make sure synthesis.py's device is set to same

# svc_model_name = 'damp_mel_Size0.25-avgEmbs_withCcLoss-autoVc_pretrainedOnVctk_Mels80-'
# svc_model_name = 'damp_mel_Size0.25-avgEmbs_with-bestPerformingSIE_mel80-'
# svc_model_name = 'damp_mel_Size0.25-avgEmbs_EmbLoss__-bestPerformingSIE_mel80-Cont2'
svc_model_name = 'damp_mel_Size0.25-avgEmbs_CcLoss__-bestPerformingSIE_mel80-to500kIters-Cont2'

# sie_model_name = 'autoVc_pretrainedOnVctk_Mels80'
sie_model_name = 'bestPerformingSIE_mel80'
testing_with = 'bneck'
use_avg_embs = True
extra_note = 'TestSet20Singers_500kIters'

test_on = 'damp'
ds_size = None
chosen_class_num = 20 # if this parameter is not None and an int, this overwrites the functionality of ds_size as it specifies the number of singers to use from the subset.
ex_per_spkr = 2
num_epochs = 600

if use_avg_embs: emb_type = 'avg_'
else: emb_type = 'live_'

vc_verion = 'autoSvc'
SVC_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{test_on}_qianParams'
SIE_feat_dir = f'/import/c4dm-02/bdoc3/spmel/{test_on}_qianParams'
subset = 'test'
device = f'cuda:{which_cuda}'
SVC_pitch_cond = False

log_step = 10
sie_emb_size = 256

with open(os.path.join(SVC_feat_dir, 'feat_params.yaml')) as File:
    SVC_feats_params = yaml.load(File, Loader=yaml.FullLoader)
with open(os.path.join(SIE_feat_dir, 'feat_params.yaml')) as File:
    SIE_feats_params = yaml.load(File, Loader=yaml.FullLoader)

sie_dir = '/homes/bdoc3/singer-identity-encoder'
csv_dir = '/homes/bdoc3/my_data/text_data'
saved_models_dir = '/homes/bdoc3/my_data/autovc_models/'
metadata_root_dir = '/homes/bdoc3/my_data/voice_embs_visuals_metadata'

SIE_dataset_name = os.path.basename(SIE_feat_dir)
# SVC_dataset_name = os.path.basename(SVC_feat_dir)
this_svc_model_dir = os.path.join(saved_models_dir, os.path.basename(vc_verion), svc_model_name)
checkpoint_path = os.path.join(this_svc_model_dir, 'ckpt_500000.pt')
SIE_path =  os.path.join(saved_models_dir, os.path.basename(sie_dir), sie_model_name)

### MODELS
if testing_with == 'bneck':
    # build models and lookup table
    if vc_verion == 'autovc_basic':
        dim_neck = 32
        dim_emb = 256
        dim_pre = 512
        sample_freq=32
        adam_init = 0.0001
        window_timesteps = 128
        G, _, _ = utils.setup_gen(dim_neck, dim_emb, dim_pre, sample_freq, 80, 0, device, checkpoint_path, adam_init)
    else:
        sys.path.insert(1, this_svc_model_dir)
        try:
            from this_train_params import dim_neck, dim_emb, dim_pre, sample_freq, adam_init, window_timesteps, batch_size, num_workers, norm_method, pitch_dir, SVC_pitch_cond, SIE_pitch_cond, midi_range, pkl_fn_extras
        except ImportError as e:
            from this_train_params import dim_neck, dim_emb, dim_pre, sample_freq, adam_init, window_timesteps, batch_size, num_workers, norm_method, pitch_dir, SVC_pitch_cond, SIE_pitch_cond, midi_range
            pkl_fn_extras = ''
        G, _, _ = utils.setup_gen(dim_neck, dim_emb, dim_pre, sample_freq, 80, 0, device, checkpoint_path, adam_init)
        G.eval()
        this_train_params = {'SIE_feat_dir':SIE_feat_dir,
                            'SVC_feat_dir':SVC_feat_dir,
                            'window_timesteps':window_timesteps,
                            'norm_method':norm_method,
                            'pitch_dir':pitch_dir,
                            'SVC_pitch_cond':SVC_pitch_cond,
                            'SIE_pitch_cond':SIE_pitch_cond,
                            'midi_range':midi_range} 
elif testing_with == 'emb':
    this_train_params = {'SIE_feat_dir':SIE_feat_dir,
                        'SVC_feat_dir':SVC_feat_dir,
                        'window_timesteps':128,
                        'norm_method':None,
                        'pitch_dir':'/import/c4dm-02/bdoc3/world_data/damp_80_16ms',
                        'SVC_pitch_cond':SVC_pitch_cond,
                        'SIE_pitch_cond':False,
                        'midi_range':range(36, 82)}
    batch_size = 2
    num_workers = 2
    adam_init = 0.0001
    pkl_fn_extras = ''

SIE_feats_params, SVC_feats_params = utils.determine_dim_size(SIE_feats_params, SVC_feats_params, SIE_feat_dir, SVC_feat_dir, use_aper_feats=False)
train_dataset, train_loader = load_primary_dataloader(SIE_feats_params, subset, SVC_feats_params, ds_size, chosen_class_num, batch_size, num_workers, this_train_params)
print('number of labels: ', len(train_dataset))

if testing_with == 'bneck': h_layer_outs = []
elif testing_with == 'emb': h_layer_outs = [sie_emb_size//2]

vc_classer = Aux_Voice_Classifier(sie_emb_size, h_layer_outs, len(train_dataset)).to(device).train()
classer_optimizer = torch.optim.Adam(vc_classer.parameters(), adam_init)


if use_avg_embs:
    metadata_path = os.path.join(metadata_root_dir, sie_model_name, SIE_dataset_name, subset, f'voices_metadata{pkl_fn_extras}.pkl')
    subset_metadata = pickle.load(open(metadata_path, "rb"))
    subset_names = [metad[0] for metad in subset_metadata]
else:
    loss_device = torch.device("cpu")
    SIE, _ = utils.setup_sie(device, loss_device, SIE_path, adam_init)

start_time = time.time()

if testing_with == 'bneck':
    writer = SummaryWriter(comment = 'MLPclassifyFrom_' +emb_type +testing_with +'-' +svc_model_name +'-testOn_' +test_on +'_' +subset +'_size' +str(ds_size) +'-' +extra_note)
elif testing_with == 'emb':
    writer = SummaryWriter(comment = 'MLPclassifyFrom_' +emb_type +testing_with +'-' +sie_model_name +'-testOn_' +test_on +'_' +subset +'_size' +str(ds_size) +'-' +extra_note)
else:
    raise NotImplementedError


total_pred_loss = 0
total_acc = 0
for k in range(num_epochs):
    for j in range(ex_per_spkr): 
        for i, mb in enumerate(train_loader):
            
            this_step = k*ex_per_spkr*len(train_loader) +j*len(train_loader) + i
            SIE_feats, SVC_feats, onehot_midi, example_id, start_step, y_data = mb
            y_data = y_data.to(device)

            if use_avg_embs:
                for l, id in enumerate(example_id):
                    # pdb.set_trace()
                    s_id = id.split('_')[0]
                    singermeta_index = subset_names.index(s_id)

                    assert s_id == subset_metadata[singermeta_index][0]
                    if l == 0:
                        emb_org = torch.from_numpy(subset_metadata[singermeta_index][1]).to(device).float().unsqueeze(0)
                    else:
                        emb_org = torch.cat((emb_org, torch.from_numpy(subset_metadata[singermeta_index][1]).to(device).float().unsqueeze(0)), axis=0)
            else:
                emb_org = SIE(SIE_feats.to(device).float()) 
            
            if testing_with == 'bneck':
                x_identic_prnt, x_identic_psnt, code_real, _, _ = G(SVC_feats.to(device).float(), emb_org, emb_org, None)
                encoding = code_real.detach().clone()
            elif testing_with == 'emb':
                encoding = emb_org.detach().clone()
            else:
                raise NotImplementedError

            predictions = vc_classer(encoding)
            pred_loss = F.cross_entropy(predictions, y_data)
            accuracy = get_accuracy(predictions, y_data)
            pred_loss.backward()
            classer_optimizer.step()
            total_pred_loss += pred_loss
            total_acc += accuracy

            if this_step % log_step == 0:
                # print(k,j,i)
                avg_acc = total_acc / log_step
                avg_loss = total_pred_loss / log_step
                et = time.time() - start_time
                et = str(datetime.timedelta(seconds=et))[:-7]
                log = "Elapsed [{}], Mode {}, Iter [{}/{}]".format(et, 'train', this_step, (num_epochs)*(ex_per_spkr)*len(train_loader))
                log += ", loss: {:.4f}".format(avg_loss)
                print(log)
                writer.add_scalar(f"VoiceClassLoss/{'loss'}", avg_loss, this_step)
                writer.add_scalar(f"VoiceClassLoss/{'Accuracy'}", avg_acc, this_step)
                total_acc = 0
                total_pred_loss = 0

writer.close()
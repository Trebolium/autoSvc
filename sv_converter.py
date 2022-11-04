from model_vc import Aux_Voice_Classifier
from torch.utils.tensorboard import SummaryWriter
import torch
import os, time, datetime, math, pdb, pickle
import numpy as np
import torch.nn.functional as F
from neural.scheduler import EarlyStopping
from train_params import *
from my_plot import array_to_img
from my_os import recursive_file_retrieval
import utils


# SOLVER IS THE MAIN SETUP FOR THE NN ARCHITECTURE. INSIDE SOLVER IS THE GENERATOR (G)
class AutoSvc(object):

    def __init__(self, SIE_params, SVC_params):
        """Initialize configurations.""" 
        
        if SVC_model_name == 'defaultName' or SVC_model_name == 'deletable':
            self.writer = SummaryWriter('runs/tests')
        else:
            self.writer = SummaryWriter(comment = '_' +SVC_model_name)

        self.SIE_params = SIE_params
        if SVC_feat_dir == '': # if svc dir isn't established, then its ok to pass it same feats as SIE
            self.SVC_params = SIE_params
        else: # if svc dir is stablished, give SVC its own params
            self.SVC_params = SVC_params

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{which_cuda}' if self.use_cuda else 'cpu')
        self.loss_device = torch.device("cpu")
        metadata_path = '/homes/bdoc3/my_data/voice_embs_visuals_metadata'
        self.train_singer_metadata = pickle.load(open(os.path.join(metadata_path, os.path.basename(SIE_model_path), os.path.basename(SIE_feat_dir), 'train', f'voices_metadata{pkl_fn_extras}.pkl'), 'rb'))
        self.val_singer_metadata = pickle.load(open(os.path.join(metadata_path, os.path.basename(SIE_model_path), os.path.basename(SIE_feat_dir), 'val', f'voices_metadata{pkl_fn_extras}.pkl'), 'rb'))
        
        # if tiny_run:
        #     # because tiny_run dirs carry copies of whats in train subset
        #     self.val_singer_metadata = self.train_singer_metadata

        if subset_size == 1.0:
            self.train_iter_multiple = self.get_iter_size(SVC_feat_dir, 'train')
            self.val_iter_multiple = self.get_iter_size(SVC_feat_dir, 'val')
        else:
            self.train_iter_multiple = self.get_iter_size(SVC_feat_dir, 'train' +f'/.{subset_size}_size')
            self.val_iter_multiple = self.get_iter_size(SVC_feat_dir, 'val' +f'/.{subset_size}_size')       

        if SVC_pitch_cond:
            pitch_dim = len(midi_range)+1
        else:
            pitch_dim = 0

        # Build the model and tensorboard.
        print('Building model...\n')
        self.sie, self.sie_num_feats_used = utils.setup_sie(self.device, self.loss_device, SIE_model_path, adam_init)
        self.G, self.g_optimizer, self.train_latest_step = utils.setup_gen(dim_neck, dim_emb, dim_pre, sample_freq,
                                                                            self.SVC_params['num_feats'],
                                                                            pitch_dim,
                                                                            self.device,
                                                                            svc_ckpt_path,
                                                                            adam_init
                                                                            )

        self.EarlyStopping = EarlyStopping(patience, threshold=patience_thresh)
        self.start_time = time.time()
        self.prev_lowest_val_loss = math.inf
        self.make_examples = True
        self.times = []
        self.first_time_pdb = True
        if use_aux_classer:
            num_classes = len(self.train_singer_metadata)
            self.dis_ev = self.load_disentangle_eval(num_classes)

    def load_disentangle_eval(self, num_classes):
        bottleneck = int((window_timesteps/sample_freq) * dim_neck)
        self.v_classer = Aux_Voice_Classifier(bottleneck, num_classes)
        self.v_classer = self.v_classer.to(self.device)
        return self.v_classer

    # since each runthrough of the dataloader retieves 1 example per singer, we must repeat the dataloader a certain number of times
    # this multiple is represented by iter_multiple
    def get_iter_size(self, feature_dir, subset):
        num_subset_subdirs = len(os.listdir(os.path.join(feature_dir, subset)))
        _, subset_fps = recursive_file_retrieval(os.path.join(feature_dir, subset))
        if use_loader == 'damp':
            chunks_per_track = 30 #based on the average time of a track
        elif use_loader == 'vctk':
            chunks_per_track = 1
        elif use_loader == 'vocadito':
            chunks_per_track = 6
        avg_uttrs_per_spkr = len(subset_fps) / num_subset_subdirs
        iter_multiple = int( (avg_uttrs_per_spkr*chunks_per_track) //batch_size)
        return iter_multiple

    def save_by_val_loss(self, current_loss):

        # Overwrite the latest version of the model whenever validation's ge2e loss is lower than previously
        if current_loss <= self.prev_lowest_val_loss:
            dst_path = os.path.join(SVC_models_dir, SVC_model_name, 'saved_model.pt')
            torch.save(
                {
                "step": self.train_latest_step,
                "ge2e_loss": current_loss,
                "model_state": self.G.state_dict(),
                "optimizer_state": self.g_optimizer.state_dict(),
                },
                dst_path)
            print(f'Saved model at {dst_path}')
            self.prev_lowest_val_loss = current_loss
            return True

    def plot_comparisons(self, feat_npys, i):
        print('Plotting specs')
        ex_path = os.path.join(SVC_models_dir, SVC_model_name, 'image_comparison', f'step_{i}')
        array_to_img(feat_npys, ex_path)


    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      

    def closeWriter(self):
        self.writer.close()


    
    #=====================================================================================================================================#
   

    def batch_iterate(self, mode, data_loader, this_cycle_initial_step, singer_metadata, num_loader_iters):
        
        if mode == 'train':
            iter_multiple = self.train_iter_multiple
        else:
            iter_multiple = self.val_iter_multiple

        singer_ids = [i[0] for i in singer_metadata]
        intercycle_loss = 0
        interlog_loss = 0
        interlog_counter = 0 # exists because a proceeding train cycle may not start with (iter % log_step) = 0 - relevant when calculating avg losses and printing
        for _ in range(iter_multiple):
            data_iter = iter(data_loader)

            for i in range(this_cycle_initial_step+1, (this_cycle_initial_step+1 + num_loader_iters)): # using +1 because of some scheduler sync issues (quick hack)         

                interlog_counter += 1
                # get data from loader
                # try:
                SIE_feats, SVC_feats, onehot_midi, example_id, start_step, y_data = next(data_iter)
                # except UnboundLocalError:
                #     data_iter = iter(data_loader)
                #     SIE_feats, SVC_feats, onehot_midi, example_id, start_step = next(data_iter)

                if SVC_model_name == 'defaultName':
                    pdb.set_trace()
                # convert to numpy
                SIE_feats_npy = SIE_feats.numpy()
                SVC_feats_npy = SVC_feats.numpy()
                onehot_midi_npy = onehot_midi.numpy()

                # concat where necessary
                if SIE_pitch_cond:
                    SIE_input = np.concatenate((SIE_feats_npy, onehot_midi_npy), axis=2)
                else:
                    SIE_input = SIE_feats_npy 
                # if SVC_pitch_cond:
                #     SVC_input = np.concatenate((SVC_feats_npy, onehot_midi_npy), axis=2)
                # else:
                SVC_input = SVC_feats_npy
                
                # save plots
                if self.make_examples:
                    inputs = [np.rot90(SIE_input[0]), np.rot90(SVC_input[0])]
                    ex_path = os.path.join(SVC_models_dir, SVC_model_name, 'input_tensor_plots', f'STEP-{i}_ID-{example_id[0][:-4]}_TIMESTAMP-{start_step[0]}')
                    array_to_img(inputs, ex_path)
                self.make_examples = False

                # convert back to tensor
                onehot_midi_input = torch.from_numpy(onehot_midi_npy).to(self.device).float()
                SIE_input = torch.from_numpy(SIE_input).to(self.device).float()
                if SVC_feat_dir != '':
                    SVC_input = torch.from_numpy(SVC_input).to(self.device).float()
                else:
                    SVC_input = SIE_input
                
                if not SVC_pitch_cond:
                    onehot_midi_input = None

                if SVC_model_name == 'defaultName':
                    pdb.set_trace()

                if self.sie_num_feats_used != SIE_input.shape[2]:
                    SIE_input = SIE_input[:,:,:self.sie_num_feats_used]
                emb_org = self.sie(SIE_input)
                
                if use_avg_singer_embs:
                    for j, ex_id in enumerate(example_id):
                        s_id = ex_id.split('_')[0]
                        singermeta_index = singer_ids.index(s_id)
                        assert s_id == singer_metadata[singermeta_index][0]
                        try:
                            if j == 0:
                                emb_org = torch.from_numpy(singer_metadata[singermeta_index][1]).to(self.device).float().unsqueeze(0)
                            else:
                                emb_org = torch.cat((emb_org, torch.from_numpy(singer_metadata[singermeta_index][1]).to(self.device).float().unsqueeze(0)), axis=0)
                        except Exception as e:
                            pdb.set_trace()

                # emb_org =  torch.from_numpy(np.zeros((2,256))).to(self.device).float()

                x_identic_prnt, x_identic_psnt, code_real, _, _ = self.G(SVC_input, emb_org, emb_org, onehot_midi_input)

                ########## OLD LOSS VERSION FROM AUTOVC_BASIC ########
                x_identic_prnt = x_identic_prnt.squeeze(1)
                x_identic_psnt = x_identic_psnt.squeeze(1)
                # residual_from_psnt = residual_from_psnt.squeeze(1)

                g_loss_id_prnt = F.l1_loss(SVC_input, x_identic_prnt)  
                g_loss_id_psnt = F.l1_loss(SVC_input, x_identic_psnt)   

                g_loss = (prnt_loss_weight * g_loss_id_prnt) + (psnt_loss_weight * g_loss_id_psnt)

                if include_code_loss:
                    # Code semantic loss. For calculating this, there is no target embedding
                    code_reconst = self.G(x_identic_psnt, emb_org, None)
                    # gets the l1 loss between original encoder output and reconstructed encoder output
                    g_loss_cd = F.l1_loss(code_real, code_reconst)
                    g_loss += (code_loss_weight * g_loss_cd)

                if use_aux_classer:
                    predictions = self.v_classer(code_real)
                    # accuracy = self.get_accuracy(predictions, y_data)
                    pred_loss = F.cross_entropy(predictions, y_data)
                    pred_loss = pred_loss.detach().clone()
                    confusion_pred_loss = -pred_loss
                    g_loss += confusion_pred_loss

                if use_emb_loss:
                    cc_emb = self.sie(x_identic_psnt)
                    cc_emb_loss = F.l1_loss(emb_org, cc_emb)
                    cc_emb_loss = cc_emb_loss.detach().clone()
                    g_loss += cc_emb_loss

                ########## OLD LOSS VERSION ^^^

                # remember that l1_loss gives you mean over batch, unless specificed otherwise
                interlog_loss += g_loss

                # logging losses
                # loss_vals = [g_loss_id.item(), g_loss_id_prnt.item(), g_loss_cd.item()]
                # for j, k in enumerate(keys):
                #     set_of_loss_list[k].append(loss_vals[j]) # to be averaged when logging and reset
                    # losses_list[j] += loss_vals[j] # gradually summing losses as cycle progresses   
                
                # SIE_midi = np.concatenate((SIE_feats_npy[0], onehot_midi_npy[0]), axis=1)
                # try: 
                feat_npys = [np.rot90(onehot_midi_npy[0]), np.rot90(SIE_feats_npy[0]),
                            np.rot90(SVC_feats_npy[0]), np.rot90(x_identic_prnt.cpu().data.numpy()[0])]


                # if train do backprop
                if mode == 'train':
                    self.train_latest_step = i
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    # spec nad freq have to be multiple of len(data_loader)
                    if i % spec_freq == 0:
                        self.plot_comparisons(feat_npys, self.train_latest_step) 

                # Print out training information.
                if i % log_step == 0 or i == (this_cycle_initial_step + num_loader_iters):
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    # if mode == 'train':
                    log = "Elapsed [{}], Mode {}, Iter [{}/{}]".format(et, mode, i, max_iters)
                    # else:
                    #     i_in_cycle =  i - this_cycle_initial_step
                    #     log = "Elapsed [{}], Mode {}, Iter [{}/{}] (".format(et, mode, i_in_cycle, len(data_loader))
                    avg_loss_per_pass = interlog_loss / interlog_counter
                    log += ", average_loss_per_logstep: {:.4f}".format(avg_loss_per_pass)
                    intercycle_loss += avg_loss_per_pass
                    interlog_loss = 0
                    interlog_counter = 0
                    # log += ", batch_processing: {:.4f}".format(sum(self.times) / len(self.times))
                    print(log)

                    if i % ckpt_freq == 0:
                        dst_path = os.path.join(SVC_models_dir, SVC_model_name, f'ckpt_{i}.pt')
                        torch.save(
                            {
                            "step": self.train_latest_step,
                            "ge2e_loss": ('avg_loss_per_pass', avg_loss_per_pass),
                            "model_state": self.G.state_dict(),
                            "optimizer_state": self.g_optimizer.state_dict(),
                            },
                            dst_path)
                        print(f'Saved model at {dst_path}')

                # if tiny_run:
                #     if this_cycle_initial_step >= max_iters or this_cycle_initial_step % ckpt_freq == 0:
                #          self.save_by_val_loss(0)
                
                # track_time = time.time()-time_before
                # self.times.append(track_time)

            # update this_cycle_initial_step for when you start next cycle of iter_multiple
            this_cycle_initial_step = i
        """FINISH CYCLE"""

        # Print average loss over cycle
        cycle_loss = intercycle_loss / math.ceil(num_loader_iters/log_step)
        log = "This cycle average loss: {:.4f}".format(cycle_loss)
        print(log)
        
        # when cycle finished in val mode, save loss, EarlyStopping check, save plot
        if mode.startswith('val'): # because this string is extended to represent tthe different training sets if evall_all is chosen
            # NEED TO EMPLOY THIS ONLY AFTER THE END OF EACH VAL ITER CYCLE, NOT INDIVIDUALLY AFTER EACH VAL
                
            # if EarlyStopping, read the docstrings if necessary, this section is unorthodox
            check = self.EarlyStopping.check(cycle_loss)
            if check == 'lowest_loss':
                self.save_by_val_loss(cycle_loss)
                print(f"Saved model (step {self.train_latest_step}) at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
            elif check == 0:
                print(f'Early stopping employed on model {SVC_model_name}.')
                self.closeWriter()
                exit(0)
        
        return cycle_loss


#=====================================================================================================================================#

    def logs(self, cycle_loss, mode):
        if mode=='train':
            self.writer.add_scalar(f"Loss_id_psnt_{use_loader}/{mode}", cycle_loss, self.train_latest_step)
        else:
            self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", cycle_loss, self.train_latest_step)

        self.writer.flush()

#=====================================================================================================================================#        

    # the core parent method in this object
    def train(self, train_loader, val_loaders):

        while self.train_latest_step < max_iters:
            
            if len(train_loader) < max_cycle_iters:
                num_loader_iters = len(train_loader)
            else:
                num_loader_iters = max_cycle_iters

            mode = 'train'
            print('TRAIN REPORT\n')
            self.G.train()
            cycle_loss = self.batch_iterate(mode, train_loader, self.train_latest_step, self.train_singer_metadata, num_loader_iters)
            
            self.logs(cycle_loss, mode)
            # then use validation loaders (or just the one)
            for ds_label, val_loader in val_loaders:
                vt_fraction = len(val_loader) / len(train_loader)
                num_loader_iters = int(num_loader_iters * vt_fraction)
                # if len(val_loader) < max_cycle_iters:
                #     num_vloader_iters = len(val_loader)
                # else:
                #     num_vloader_iters = max_cycle_iters

                mode = f'val_{ds_label}'
                print('VAL REPORT\n')
                self.G.eval()
                with torch.no_grad():
                    cycle_loss = self.batch_iterate(mode, val_loader, 0, self.val_singer_metadata, num_loader_iters)
                self.logs(cycle_loss, mode)

        self.closeWriter()
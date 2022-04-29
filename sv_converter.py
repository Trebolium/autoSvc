from model_vc import Generator
from model_sie import SingerIdEncoder
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch
import os, time, datetime, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from neural.scheduler import EarlyStopping
from train_params import *
from my_plot import save_array_img

def save_ckpt(model, model_optimizer, loss, iteration, save_path):
    print('Saving model...')
    checkpoint = {'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': model_optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss}
    torch.save(checkpoint, save_path)



# SOLVER IS THE MAIN SETUP FOR THE NN ARCHITECTURE. INSIDE SOLVER IS THE GENERATOR (G)
class AutoSvc(object):

    def __init__(self, data_loader, SIE_params, SVC_params):
        """Initialize configurations.""" 
        
        if svc_model_name == 'defaultName' or svc_model_name == 'deletable':
            self.writer = SummaryWriter('runs/tests')
        else:
            self.writer = SummaryWriter(comment = '_' +svc_model_name)

        self.SIE_params = SIE_params
        if SVC_feat_dir == '': # if svc dir isn't established, then its ok to pass it same feats as SIE
            self.SVC_params = SIE_params
        else: # if svc dir is stablished, give SVC its own params
            self.SVC_params = SVC_params

        self.data_loader = data_loader

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{which_cuda}' if self.use_cuda else 'cpu')
        self.loss_device = torch.device("cpu")

        # Build the model and tensorboard.
        self.build_model()
        self.EarlyStopping = EarlyStopping(patience)
        self.start_time = time.time()

    def build_model(self):

        sie_checkpoint = torch.load(os.path.join(SIE_path, 'saved_model.pt'))
        new_state_dict = OrderedDict()
        sie_num_feats_used = sie_checkpoint['model_state']['lstm.weight_ih_l0'].shape[1]
        sie_num_voices_used = sie_checkpoint['model_state']['class_layer.weight'].shape[0]
        for i, (key, val) in enumerate(sie_checkpoint['model_state'].items()):
            print(f'layer {key} weight shape: {val.shape}')    
            new_state_dict[key] = val
        
        self.sie = SingerIdEncoder(self.device, self.loss_device, sie_num_voices_used, sie_num_feats_used) # don't know why this continues to change
        for param in self.sie.parameters():
            param.requires_grad = False
        self.sie_optimizer = torch.optim.Adam(self.sie.parameters(), adam_init)
        self.sie.load_state_dict(new_state_dict)
        for state in self.sie_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self.device)
        self.sie.to(self.device)
        self.sie.eval() 

        #if feats between SIE and SVC should be different
        self.G = Generator(dim_neck, dim_emb, dim_pre, sample_freq, self.SVC_params['num_feats'])        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), )
        if svc_ckpt_path!='':
            g_checkpoint = torch.load(svc_ckpt_path)
            self.G.load_state_dict(g_checkpoint['model_state_dict'])
            self.g_optimizer.load_state_dict(g_checkpoint['optimizer_state_dict'])
            # fixes tensors on different devices error
            # https://github.com/pytorch/pytorch/issues/2830
            for state in self.g_optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

            self.current_iter = g_checkpoint['iteration']
        else:
            self.current_iter = 0
        self.G.to(self.device)

    def save_by_val_loss(self, current_loss):

        # Overwrite the latest version of the model whenever validation's ge2e loss is lower than previously
        if current_loss < self.prev_lowest_val_loss: 
            torch.save(
                {
                "step": self.train_current_step,
                "ge2e_loss": current_loss,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                },
                os.path.join(svc_model_dir, svc_model_name, 'saved_model.pt'))
            self.prev_lowest_val_loss = current_loss
            return True

    def get_current_iters(self):
        return self.current_iter

    def reset_grad(self):
        """Reset the gradient buffers."""
        self.g_optimizer.zero_grad()
      
    def closeWriter(self):
        self.writer.close()
    
    #=====================================================================================================================================#
   
    def iterate(self, mode, data_loader, current_iter, cycle_size, log_list): 

        def batch_iterate():
    
            for i in range(current_iter+1, (current_iter+1 + cycle_size)):          
                
                try:
                    specs_x, pitch_x, example_id = next(data_iter)
                except UnboundLocalError:
                    data_iter = iter(data_loader)
                    specs_x, pitch_x, example_id = next(data_iter)
                except RuntimeError:
                    pdb.set_trace()

                if SVC_feat_dir != '':
                    SIE_specs = specs_x[:,0].to(self.device).float()
                    SVC_specs = specs_x[:,1].to(self.device).float()
                else:
                    SIE_specs = specs_x[0].to(self.device).float()

                if True:
                    ex_path = os.path.join(svc_model_dir, svc_model_name, 'input_tensor_plots', f'step {step}')
                    save_array_img(np.rot90(SIE_specs), ex_path)

                if pitch_cond:
                    # SIE_pitch = pitch_x[0].to(self.device).float()
                    SVC_pitch = pitch_x.to(self.device).float()
                    SVC_input = np.concatenate(SVC_specs, SVC_pitch, axis =1)
                else:
                    SVC_input = SVC_specs

                emb_org, _ = self.sie(SIE_specs)
                x_identic, x_identic_psnt, code_real, _, _ = self.G(SVC_input, emb_org, emb_org)
                
                residual_from_psnt = x_identic_psnt - x_identic
                x_identic = x_identic.squeeze(1)
                x_identic_psnt = x_identic_psnt.squeeze(1)
                residual_from_psnt = residual_from_psnt.squeeze(1)
                g_loss_id = F.l1_loss(SVC_specs, x_identic)   
                g_loss_id_psnt = F.l1_loss(SVC_specs, x_identic_psnt)   
                code_reconst = self.G(x_identic_psnt, emb_org, None)
                g_loss_cd = F.l1_loss(code_real, code_reconst)

                # Logging.
                loss = {}
                loss['G/loss_id'] = g_loss_id.item()
                loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
                loss['G/loss_cd'] = g_loss_cd.item()     
                losses_list[0] += g_loss_id.item()
                losses_list[1] += g_loss_id_psnt.item()
                losses_list[2] += g_loss_cd.item()



                # Print out training information.
                if i % log_step == 0 or i == (current_iter + cycle_size):
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    if mode == 'train':
                        log = "Elapsed [{}], Mode {}, Iter [{}/{}]".format(et, mode, i, max_iters)
                    else: log = "Elapsed [{}], Mode {}".format(et, mode)
                    for tag in keys:
                        log += ", {}: {:.4f}".format(tag, loss[tag])
                    print(log)
                    log_list.append(log)

                if mode == 'train':

                    g_loss = (prnt_loss_weight * g_loss_id) + (psnt_loss_weight * g_loss_id_psnt) #+ ((config.lambda_cd  * (i / 100000)) * g_loss_cd)
                    
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    # spec nad freq have to be multiple of cycle_size
                    if i % spec_freq == 0:
                        print('plotting specs')
                        SVC_specs = SVC_specs.cpu().data.numpy()
                        x_identic = x_identic.cpu().data.numpy()
                        x_identic_psnt = x_identic_psnt.cpu().data.numpy()
                        residual_from_psnt = residual_from_psnt.cpu().data.numpy()
                        specs_list = []
                        for arr in SVC_specs:
                            specs_list.append(arr)
                        for arr in x_identic:
                            specs_list.append(arr)
                        for arr in residual_from_psnt:
                            specs_list.append(arr)
                        for arr in x_identic_psnt:
                            specs_list.append(arr)
                        columns = 2
                        rows = 4
                        fig, axs = plt.subplots(4,2)
                        fig.tight_layout()
                        for j in range(0, columns*rows):
                            spec = np.rot90(specs_list[j])
                            fig.add_subplot(rows, columns, j+1)
                            if j == 5 or j == 6:
                                spec = spec - np.min(spec)
                                plt.clim(0,1)
                            plt.imshow(spec)
                            name = 'Egs ' +str(example_id[j%2])
                            plt.title(name)
                            plt.colorbar()
                        plt.savefig(os.path.join(svc_model_dir, svc_model_name, '/image_comparison/',  str(i) +'iterations'))
                        plt.close()

                elif mode.startswith('val'): # because this string is extended to represent tthe different training sets if evall_all is chosen
                    if self.save_by_val_loss(g_loss):
                        print(f"Saved model (step {self.train_current_step}) at time {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
                    # if EarlyStopping, stop training entirely
                    if self.EarlyStopping.check(g_loss): 
                        print(f'Early stopping employed.')
                        self.closeWriter()
                        exit(0)

            return losses_list, (current_iter + cycle_size)

#=====================================================================================================================================#

        def logs(losses_list, mode, current_iter): 
            if mode[4:]=='vocal':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*batch_size), current_iter)
            elif mode[4:]=='medley':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*batch_size), current_iter)
            elif mode[4:]=='vctk':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*batch_size), current_iter)
            elif mode[4:]=='damp':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*batch_size), current_iter)
            elif mode=='train':
                self.writer.add_scalar(f"Loss_id_psnt_{use_loader}/{mode}", losses_list[1]/(cycle_size*batch_size), current_iter)
            else: exit(1)
            losses_list = [0.,0.,0.]
            self.writer.flush()
            print('writer flushed')

#=====================================================================================================================================#        

        # Print logs in specified order
        keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
        losses_list = [0., 0., 0.]
        # Start training.
        print('Start training...')

        if mode == 'train':
            self.G.train()
            losses_list, current_iter = batch_iterate()
            logs(losses_list, mode, current_iter)
        elif mode.startswith('val'):
            self.G.eval()
            with torch.no_grad():
                losses_list, _ = batch_iterate()
            logs(losses_list, mode, current_iter)
            print(losses_list[1])
            if self.EarlyStopping.check(losses_list[1]):
                print(f"""Early stopping called at iteration {current_iter}.\n
                    The lowest loss {self.EarlyStopping.lowest_loss} has not decreased since {patience} iterations.\n
                    Saving model...""")
                save_path = svc_model_dir +'/' +svc_model_name +'/ckpts/' +'ckpt_' +str(current_iter) +'.pth.tar'
                save_ckpt(self.G, self.g_optimizer, losses_list, current_iter, save_path)
                exit(0)

        return current_iter, log_list 
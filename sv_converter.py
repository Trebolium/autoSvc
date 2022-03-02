from model_vc import Generator
from si_encoder.model import SingerIdEncoder
from collections import OrderedDict
from torch.utils.tensorboard import SummaryWriter
import torch
import os, time, datetime, sys, pdb
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from neural.scheduler import EarlyStopping

def save_ckpt(model, model_optimizer, loss, iteration, save_path):
    print('Saving model...')
    checkpoint = {'model_state_dict' : model.state_dict(),
        'optimizer_state_dict': model_optimizer.state_dict(),
        'iteration': iteration,
        'loss': loss}
    torch.save(checkpoint, save_path)



# SOLVER IS THE MAIN SETUP FOR THE NN ARCHITECTURE. INSIDE SOLVER IS THE GENERATOR (G)
class AutoSvc(object):

    def __init__(self, data_loader, config, SIE_params, SVC_params=None):
        """Initialize configurations.""" 
        self.config = config

        if self.config.file_name == 'defaultName' or self.config.file_name == 'deletable':
            self.writer = SummaryWriter('runs/tests')
        else:
            self.writer = SummaryWriter(comment = '_' +self.config.file_name)

        self.SIE_params = SIE_params
        if config.diff_svc_feats_dir != '':
            self.SVC_params = SVC_params
        self.data_loader = data_loader

        # Miscellaneous.
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device(f'cuda:{self.config.which_cuda}' if self.use_cuda else 'cpu')
        self.loss_device = torch.device("cpu")

        # Build the model and tensorboard.
        self.build_model()
        self.EarlyStopping = EarlyStopping(self.config.patience)
        self.start_time = time.time()

    def build_model(self):

        sie_checkpoint = torch.load(os.path.join(self.config.sie_path, 'saved_model.pt'))
        new_state_dict = OrderedDict()
        sie_num_feats_used = sie_checkpoint['model_state']['lstm.weight_ih_l0'].shape[1]
        sie_num_voices_used = sie_checkpoint['model_state']['class_layer.weight'].shape[0]
        for i, (key, val) in enumerate(sie_checkpoint['model_state'].items()):
            print(f'layer {key} weight shape: {val.shape}')    
            new_state_dict[key] = val
        
        self.sie = SingerIdEncoder(self.device, self.loss_device, sie_num_voices_used, sie_num_feats_used) # don't know why this continues to change
        for param in self.sie.parameters():
            param.requires_grad = False
        self.sie_optimizer = torch.optim.Adam(self.sie.parameters(), 0.0001)
        self.sie.load_state_dict(new_state_dict)
        for state in self.sie_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(self.device)
        self.sie.to(self.device)
        self.sie.eval()


        #if feats between SIE and SVC should be different
        if self.config.diff_svc_feats_dir != '': 
            self.G = Generator(self.config.dim_neck, self.config.dim_emb, self.config.dim_pre, self.config.freq, self.SVC_params['num_feats'])        
        else:
            self.G = Generator(self.config.dim_neck, self.config.dim_emb, self.config.dim_pre, self.config.freq, self.SIE_params['num_feats'])        
        self.g_optimizer = torch.optim.Adam(self.G.parameters(), self.config.adam_init)
        if self.config.ckpt_weights!='':
            ckpt_path = os.path.join(self.config.model_dir, self.config.ckpt_weights)
            g_checkpoint = torch.load(ckpt_path)
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

                if i > 200:
                    pdb.set_trace()
                
                if self.config.diff_svc_feats_dir != '':
                    try:
                        SIE_x, SVC_x, dataset_idx, example_id = next(data_iter)
                    except UnboundLocalError:
                        data_iter = iter(data_loader)
                        SIE_x, SVC_x, dataset_idx, example_id = next(data_iter)
                    except RuntimeError:
                        pdb.set_trace()
                    SVC_x = SVC_x.to(self.device).float()
                    SIE_x = SIE_x.to(self.device).float() 
                    emb_org, _ = self.sie(SIE_x)
                    x_identic, x_identic_psnt, code_real, _, _ = self.G(SVC_x, emb_org, emb_org)
                    main_x = SVC_x

                else:
                    try:
                        main_x, dataset_idx, example_id = next(data_iter)
                    except:
                        data_iter = iter(data_loader)
                        main_x, dataset_idx, example_id = next(data_iter)
                    main_x = main_x.to(self.device).float() 
                    emb_org, _ = self.sie(main_x)                
                    x_identic, x_identic_psnt, code_real, _, _ = self.G(main_x, emb_org, emb_org)
                
                residual_from_psnt = x_identic_psnt - x_identic
                x_identic = x_identic.squeeze(1)
                x_identic_psnt = x_identic_psnt.squeeze(1)
                residual_from_psnt = residual_from_psnt.squeeze(1)
                g_loss_id = F.l1_loss(main_x, x_identic)   
                g_loss_id_psnt = F.l1_loss(main_x, x_identic_psnt)   
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
                if i % self.config.log_step == 0 or i == (current_iter + cycle_size):
                    et = time.time() - self.start_time
                    et = str(datetime.timedelta(seconds=et))[:-7]
                    if mode == 'train':
                        log = "Elapsed [{}], Mode {}, Iter [{}/{}]".format(et, mode, i, self.config.max_iters)
                    else: log = "Elapsed [{}], Mode {}".format(et, mode)
                    for tag in keys:
                        log += ", {}: {:.4f}".format(tag, loss[tag])
                    print(log)
                    log_list.append(log)

                if mode == 'train':

                    g_loss = (self.config.prnt_loss_weight * g_loss_id) + (self.config.psnt_loss_weight * g_loss_id_psnt) #+ ((self.config.lambda_cd  * (i / 100000)) * g_loss_cd)
                    
                    self.reset_grad()
                    g_loss.backward()
                    self.g_optimizer.step()
                    # spec nad freq have to be multiple of cycle_size
                    if i % self.config.spec_freq == 0:
                        print('plotting specs')
                        main_x = main_x.cpu().data.numpy()
                        x_identic = x_identic.cpu().data.numpy()
                        x_identic_psnt = x_identic_psnt.cpu().data.numpy()
                        residual_from_psnt = residual_from_psnt.cpu().data.numpy()
                        specs_list = []
                        for arr in main_x:
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
                            name = 'Egs ' +str(example_id[j%2]) +', ds_idx ' +str(dataset_idx[j%2])
                            plt.title(name)
                            plt.colorbar()
                        plt.savefig(self.config.model_dir +'/' +self.config.file_name +'/image_comparison/' +str(i) +'iterations')
                        plt.close()
        
            return losses_list, (current_iter + cycle_size)

#=====================================================================================================================================#

        def logs(losses_list, mode, current_iter): 
            if mode[4:]=='vocal':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode[4:]=='medley':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode[4:]=='vctk':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode[4:]=='damp':
                self.writer.add_scalar(f"Loss_id_psnt_{mode[4:]}/{mode[:4]}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
            elif mode=='train':
                self.writer.add_scalar(f"Loss_id_psnt_{self.config.use_loader}/{mode}", losses_list[1]/(cycle_size*self.config.batch_size), current_iter)
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
                    The lowest loss {self.EarlyStopping.lowest_loss} has not decreased since {self.config.patience} iterations.\n
                    Saving model...""")
                save_path = self.config.model_dir +'/' +self.config.file_name +'/ckpts/' +'ckpt_' +str(current_iter) +'.pth.tar'
                save_ckpt(self.G, self.g_optimizer, losses_list, current_iter, save_path)
                exit(0)

        return current_iter, log_list   
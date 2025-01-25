import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math
import time
import psutil
import os
import gc
import tqdm
from ema_pytorch import EMA
from copy import deepcopy

from src.utils import edgemask
from src.reverse_models import GNNModel
from torch.utils.data import DataLoader, Dataset

norm_dict = {
    'id': nn.Identity,
    'layer': nn.LayerNorm,
    'batch': nn.BatchNorm1d
}

def get_scheduler(optimizer, cfg):
    if cfg.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.maxIter, 0)
    elif cfg.scheduler == 'constant':
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, total_iters=0)
    elif cfg.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.anneal_freq, gamma=cfg.anneal_rate)
    return scheduler

class Decoder(nn.Module):
    def __init__(self, out_dim, latent_dim=2, hidden_units=512, norm='id', num_layers=2, resnet=True):
        super().__init__()
        self.resnet = resnet
        self.model = nn.ModuleList()
        self.model.append(nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            norm_dict[norm](hidden_units),
            nn.ELU())
        )
        for i in range(num_layers-1):
            self.model.append(nn.Sequential(
                nn.Linear(hidden_units, hidden_units),
                norm_dict[norm](hidden_units),
                nn.ELU())
            )
        self.model.append(nn.Linear(hidden_units, out_dim))
        
    def forward(self, x):
        for idx, layer in enumerate(self.model):
            if 0 < idx < len(self.model) - 1 and self.resnet:
                x = layer(x) + x
            else:
                x = layer(x)
        return x
        
        
class Encoder(nn.Module):
    def __init__(self, ntips, edge_mask, cfg):
        super().__init__()
        self.ntips = ntips
        self.ndim = (self.ntips-3) * (self.ntips-1)
        self.device = cfg.base.device
        self.latent_dim = cfg.encoder.latent_dim
        self.encoder = GNNModel(self.ntips, 2*cfg.encoder.latent_dim, hidden_dim=cfg.encoder.hidden_dim, num_layers=cfg.encoder.num_layers, gnn_type=cfg.encoder.gnn_type, aggr=cfg.encoder.aggr, bias=cfg.encoder.bias, device=self.device)            
    
    def multi_sample_and_logq(self, mean, std, n_particles=1):        
        samp_z_sn = torch.randn(len(mean), n_particles, self.latent_dim, device=self.device)
        samp_z = mean.unsqueeze(1) + torch.exp(std).unsqueeze(1) * samp_z_sn
        logq_z = torch.sum(-0.5*math.log(2*math.pi) - std.unsqueeze(1) - 0.5*samp_z_sn**2, dim=-1)
        
        return samp_z, logq_z
        
    def forward(self, node_features, edge_index, min_clip=-5., max_clip=3.0):
        mean, std = torch.chunk(self.encoder.mp_forward(node_features, edge_index), 2, dim=-1)
        std = torch.clamp(std, min_clip, max_clip)
        return mean, std
        
		
		
class LVMTree(nn.Module):
    def __init__(self, taxa, cfg):
        super().__init__()
        self.taxa = taxa
        self.ntips = len(taxa)
        self.latent_dim = cfg.decoder.latent_dim
        self.edge_mask = edgemask(self.ntips).to(cfg.base.device)
        self.device = cfg.base.device
        
        self.decoder = Decoder((self.ntips - 3) * (self.ntips - 1), latent_dim=cfg.decoder.latent_dim, hidden_units=cfg.decoder.hidden_units, norm=cfg.decoder.norm, num_layers=cfg.decoder.num_layers, resnet=cfg.decoder.resnet).to(cfg.base.device)
        
    def sample_z(self, n_particles):
        samp_z = torch.randn(n_particles, self.latent_dim, device=self.device)
        return samp_z, torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_z**2, dim=-1)
          
    def cond_prob_mat(self, latent_logits):
        temp_mat = torch.zeros(len(latent_logits), *self.edge_mask.size(), device=self.device)
        temp_mat.masked_scatter_(self.edge_mask, latent_logits)
        masked_temp_mat = temp_mat.masked_fill(~self.edge_mask, -float('inf'))
        
        return F.softmax(masked_temp_mat, dim=-1)
    
    def forward(self, tree_vec, samp_z):
        latent_z = self.decoder(samp_z.reshape(-1, self.latent_dim))
        n_particles = len(latent_z) // len(tree_vec)
        cond_probs_mat = self.cond_prob_mat(latent_z)
        log_tree_probs = torch.sum(torch.log(torch.gather(cond_probs_mat, -1, tree_vec.repeat_interleave(n_particles, dim=0).unsqueeze(-1)).squeeze(-1).clamp(1e-6)), dim=-1)
        
        return log_tree_probs.reshape(samp_z.shape[:2])
        

class VAETree(nn.Module):
    EPS = np.finfo(float).eps
    def __init__(self, taxa, dataloader, emp_dataloader=None, cfg=None):
        super().__init__()
        self.ntips = len(taxa)
        self.cfg = cfg
        self.latent_dim = cfg.decoder.latent_dim
        
        self.dataloader = dataloader
        
        self.emp_dataloader = emp_dataloader
        
        if self.emp_dataloader is not None:
            self.emp_freqs = self.emp_dataloader.dataset.wts
        else:
            self.emp_freqs = self.dataloader.dataset.wts
        
        self.negDataEnt = np.sum(self.emp_freqs * np.log(np.maximum(self.emp_freqs, self.EPS)))
        
        self.latent_tree_model = LVMTree(taxa, cfg=cfg)
        self.encoder = Encoder(self.ntips, self.latent_tree_model.edge_mask, cfg=cfg)    
        
    @torch.no_grad()
    def kl_div(self, n_particles=1000):
        lower_bound = []
        for i in range(self.emp_dataloader.dataset.length):
            node_features, edge_index, vec = self.emp_dataloader.dataset.__getitem__(i)
            node_features, edge_index, vec = node_features.to(self.encoder.device), edge_index.to(self.encoder.device), vec.to(self.encoder.device)
            lower_bound.append(self.multi_sample_lower_bound(node_features.unsqueeze(0), edge_index.unsqueeze(0), vec.unsqueeze(0), n_particles))
        estimated_prob = torch.stack(lower_bound).squeeze().exp().cpu().numpy()

        return self.negDataEnt - np.sum(self.emp_freqs * np.log(estimated_prob.clip(self.EPS) / np.sum(estimated_prob.clip(self.EPS)))), estimated_prob
    
    @torch.no_grad()
    def lower_bound_batch(self, n_particles=1000):
        lower_bound = []
        for i in range(self.dataloader.dataset.length):
            node_features, edge_index, vec = self.dataloader.dataset.__getitem__(i)
            node_features, edge_index, vec = node_features.to(self.encoder.device), edge_index.to(self.encoder.device), vec.to(self.encoder.device)
            lower_bound.append(self.multi_sample_lower_bound(node_features.unsqueeze(0), edge_index.unsqueeze(0), vec.unsqueeze(0), n_particles))
        return np.sum(self.dataloader.dataset.wts * torch.stack(lower_bound).squeeze().cpu().numpy())
    
    @torch.no_grad()
    def compute_rep(self):
        loader = DataLoader(self.dataloader.dataset, batch_size=self.cfg.objective.batch_size)
        representations = []
        with tqdm.tqdm(total=len(self.dataloader.dataset)) as pbar:
            for i, (node_features, edge_index, vec) in enumerate(loader):
                pbar.update(node_features.shape[0])
                mean, std = self.encoder(node_features, edge_index)
                representations.append(mean)
        representations = torch.concat(representations, dim=0)
        representations = representations.cpu().numpy()
        return representations

    def iwae_loss(self, n_particles):
        node_features, edge_index, vec = self.dataloader.next()
        return self.multi_sample_lower_bound(node_features, edge_index, vec, n_particles).mean(0)
    
    def multi_sample_lower_bound(self, node_features, edge_index, vec, n_particles):
        mean, std = self.encoder(node_features, edge_index)  
        samp_z, logq_z = self.encoder.multi_sample_and_logq(mean, std, n_particles) 
        logp_z = torch.sum(-0.5*math.log(2*math.pi) - 0.5*samp_z**2, dim=-1)
        log_cond_probs = self.latent_tree_model(vec, samp_z)

        lower_bound = torch.logsumexp(log_cond_probs + logp_z - logq_z - math.log(n_particles), dim=-1)
        return lower_bound
    
    def learn(self, cfg, logger, tb_logger):
        objective_cfg, optimizer_cfg = cfg.objective, cfg.optimizer
        test_lbs, test_kls, ema_test_lbs, ema_test_kls, times = [], [], [], [], []
        lbs, gradnorms = [], []
        optimizer = torch.optim.Adam([
                {'params': self.latent_tree_model.parameters(), 'lr':optimizer_cfg.dec_stepsz},
                {'params': self.encoder.parameters(), 'lr': optimizer_cfg.enc_stepsz}
            ])
        scheduler = get_scheduler(optimizer, optimizer_cfg)
        run_time = -time.time()
        ema = EMA(self, beta=cfg.optimizer.ema_beta, update_every=cfg.optimizer.ema_update_every, update_after_step=cfg.optimizer.ema_update_after_step)
        with tqdm.tqdm(total=self.cfg.optimizer.maxIter) as pbar:
            for i, data in enumerate(self.dataloader):
                pbar.update(1)
                it = i + 1
                node_features, edge_index, vec = data
                node_features, edge_index, vec = node_features.to(cfg.base.device), edge_index.to(cfg.base.device), vec.to(cfg.base.device)
                if objective_cfg.method == 'iwae':
                    lb = self.multi_sample_lower_bound(node_features, edge_index, vec, cfg.objective.n_particles).mean(0)
                else:
                    raise NotImplementedError
                    
                lbs.append(lb.item())
                    
                optimizer.zero_grad()
                (-lb).backward()
                optimizer.step()
                scheduler.step()
                ema.update()

                gradnorm = torch.nn.utils.clip_grad.clip_grad_norm_(self.parameters(), max_norm=float('inf'), error_if_nonfinite=True)
                gradnorms.append(gradnorm.item())

                if it % optimizer_cfg.test_freq == 0:
                    run_time += time.time()
                    times.append(run_time)
                    memory = psutil.Process(os.getpid()).memory_info().rss/1024/1024
                    logger.info('{} Iter {}:({:.1f}s) Lower Bound: {:.4f} | GradNorm: {:.4f} | Memory: {:.4f} MB'.format(time.asctime(time.localtime(time.time())), it, run_time, np.mean(lbs), np.mean(gradnorms), memory))
                    gc.collect()

                    tb_logger.add_scalar('MLB', lb.item(), it)
                    tb_logger.add_scalar('Gradient Norm', gradnorm.item(), it)
                    tb_logger.add_scalar('Memory', memory, it)
                    for idx, p in enumerate(optimizer.param_groups):
                        tb_logger.add_scalar('lr_param_group_{}'.format(idx + 1), p['lr'], it)

                    if it % optimizer_cfg.lb_freq == 0:
                        run_time = -time.time()
                        self.eval()
                        test_lbs.append(self.lower_bound_batch())
                        if self.emp_dataloader:
                            test_kls.append(self.kl_div()[0])
                        current_state_dict = deepcopy(self.state_dict())
                        self.load_state_dict(ema.ema_model.state_dict())
                        ema_test_lbs.append(self.lower_bound_batch())
                        if self.emp_dataloader:
                            ema_test_kls.append(self.kl_div()[0])
                        self.load_state_dict(current_state_dict)
                        gc.collect()
                        
                        self.train()
                        run_time += time.time()
                        if self.emp_dataloader:
                            logger.info('>>> Iter {}:({:.1f}s) Lower Bound: {:.4f} | KL Div: {:.4f} | EMA Lower Bound: {:.4f} | EMA KL Div: {:.4f}'.format(it, run_time, test_lbs[-1], test_kls[-1], ema_test_lbs[-1], ema_test_kls[-1]))
                            tb_logger.add_scalar('MLB (K=1000)', test_lbs[-1], it)
                            tb_logger.add_scalar('KL', test_kls[-1], it)
                            tb_logger.add_scalar('EMA MLB (K=1000)', ema_test_lbs[-1], it)
                            tb_logger.add_scalar('EMA KL', ema_test_kls[-1], it)
                        else:
                            logger.info('>>> Iter {}:({:.1f}s) Lower Bound: {:.4f} | EMA Lower Bound: {:.4f}'.format(it, run_time, test_lbs[-1], ema_test_lbs[-1]))
                            tb_logger.add_scalar('MLB (K=1000)', test_lbs[-1], it)
                            tb_logger.add_scalar('EMA MLB (K=1000)', ema_test_lbs[-1], it)

                    run_time = -time.time()
                    lbs, gradnorms = [], []

                if it % optimizer_cfg.save_freq == 0:
                    torch.save({'model': self.state_dict(), 'ema': ema.ema_model.state_dict()}, cfg.base.save_to_path.replace('final', str(it)))
            
        torch.save({'model': self.state_dict(), 'ema': ema.ema_model.state_dict()}, cfg.base.save_to_path)
                
        return test_lbs, test_kls, ema_test_lbs, ema_test_kls, times
        
        
 
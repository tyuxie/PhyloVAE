import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

import numpy as np
import logging
import os
import random
from omegaconf import OmegaConf

from src.tb_logging import TensorboardLogger as TBLogger
from src.latent_tree_model import VAETree
from src.datasets import EmbedData
import tqdm

def get_cfg():
    cfg_file = OmegaConf.load('config.yaml')
    cfg_cmd = OmegaConf.from_cli()
    cfg = OmegaConf.merge(cfg_file, cfg_cmd)
    return cfg

def main(cfg):
    base_cfg = cfg.base
    base_cfg.result_folder = os.path.join(
        cfg.base.folder,
        'tde',
        cfg.data.dataset,
        f'rep_{cfg.data.rep_id}',
        '_'.join(['latent_dim', str(cfg.decoder.latent_dim),'batch_size', str(cfg.objective.batch_size), 'n_particles', str(cfg.objective.n_particles), cfg.encoder.gnn_type, cfg.encoder.aggr, cfg.base.datetime])
    )
    base_cfg.save_to_path = base_cfg.result_folder + '/final.pt'
    os.makedirs(base_cfg.result_folder, exist_ok=False if base_cfg.mode=='train' else True)

    torch.set_num_threads(1)    
    torch.random.manual_seed(base_cfg.seed)
    np.random.seed(base_cfg.seed)
    random.seed(base_cfg.seed)
    base_cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tb_logger = TBLogger(base_cfg.result_folder)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    filehandler = logging.FileHandler(base_cfg.result_folder + '/final.log')
    filehandler.setLevel(logging.INFO)
    logger.addHandler(filehandler)

    if base_cfg.mode == 'train':
        logger.info('Training with the following settings:')
        logger.info(OmegaConf.to_yaml(cfg))
    elif base_cfg.mode == 'test':
        logger.info('Testing with the following settings:')
        logger.info(OmegaConf.to_yaml(cfg))
    elif base_cfg.mode == 'rep':
        logger.info('Building representations with the following settings:')
        logger.info(OmegaConf.to_yaml(cfg))
    else: 
        raise NotImplementedError(base_cfg.mode)

    taxa = np.load(os.path.join('embed_data', cfg.data.dataset, f'repo{cfg.data.rep_id}', 'taxa.npy'))
    dataset = EmbedData(cfg.data.dataset, repo=cfg.data.rep_id)
    sampler = WeightedRandomSampler(weights=dataset.wts, num_samples=cfg.objective.batch_size*cfg.optimizer.maxIter, replacement=True)
    dataloader = DataLoader(dataset, batch_size=cfg.objective.batch_size, sampler=sampler, num_workers=2)
    if cfg.data.empFreq:
        emp_dataset = EmbedData(cfg.data.dataset, repo='emp')
        emp_dataloader = DataLoader(emp_dataset, batch_size=1, num_workers=2)
    else:
        emp_dataloader = None
    model = VAETree(taxa, dataloader, emp_dataloader, cfg=cfg).to(device=cfg.base.device)

    if base_cfg.mode == 'train':
        logger.info('\nPhyloVAE running, results will be saved to: {}\n'.format(base_cfg.save_to_path))
        logger.info('Entropy of training data: {:.4f}\n'.format(np.sum(dataloader.dataset.wts*np.log(dataloader.dataset.wts))))
        
        test_lbs, test_kls, ema_test_lbs, ema_test_kls, times = model.learn(cfg=cfg, logger=logger, tb_logger=tb_logger)
        
        np.save(base_cfg.save_to_path.replace('.pt', '_test_lbs.npy'), test_lbs)
        np.save(base_cfg.save_to_path.replace('.pt', '_test_kls.npy'), test_kls)
        np.save(base_cfg.save_to_path.replace('.pt', '_ema_test_lbs.npy'), ema_test_lbs)
        np.save(base_cfg.save_to_path.replace('.pt', '_ema_test_kls.npy'), ema_test_kls)
        np.save(base_cfg.save_to_path.replace('.pt', '_times.npy'), times)
    elif base_cfg.mode == 'test':
        logger.info('\nPhyloVAE testing, results will be saved to: {}\n'.format(base_cfg.save_to_path))
        
        for key in ['ema', 'model']:
            model.load_state_dict(torch.load(base_cfg.save_to_path)[key])
            model.eval()
            with torch.no_grad():
                if cfg.data.empFreq:
                    kldiv, pred_probs = model.kl_div()
                    np.save(base_cfg.save_to_path.replace('final.pt', key+'_kldiv.npy'), [kldiv])
                    np.save(base_cfg.save_to_path.replace('final.pt', key+'_pred_probs.npy'), pred_probs)
                    logger.info('\nThe {} final KL Divergence is {:.4f}\n'.format(key, kldiv))
                else:
                    mlls = np.array([model.lower_bound_batch() for _ in range(50)])
                    np.save(base_cfg.save_to_path.replace('final.pt', key+'_mlls.npy'), mlls)
                    logger.info('\nThe {} marginal likelihood evaluation is finished. Mean: {:.4f} Std: {:.4f}'.format(key, np.mean(mlls), np.std(mlls)))
    elif base_cfg.mode == 'rep':
        # for key in ['ema', 'model']:
        model.load_state_dict(torch.load(base_cfg.save_to_path)['ema'])
        model.eval()
        representations = model.compute_rep()
        np.savetxt(base_cfg.save_to_path.replace('.pt', '_representations.txt'), representations)



if __name__ == "__main__":
    cfg = get_cfg()
    main(cfg)
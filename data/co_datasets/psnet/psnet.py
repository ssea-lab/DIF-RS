import os

import torch
import torch.nn as nn
import torch.optim as optim
from co_datasets.psnet.actor import ActorNet
from co_datasets.psnet.critic import CriticNet
import logging
import numpy as np


def model_init(cfg, device, dataset, type='train'):
    model = None
    model_path = cfg.model_path.format(dataset)
    if os.path.exists(model_path):
        model = torch.load(model_path)
        if type == 'train':
            logging.info('load model finished, continue training from basic checkpoint')
    else:
        if type == 'train':
            logging.info('no model found, start training new model')
        else:
            raise Exception('no model found')

    return PSNet(cfg, model, device)


def score_handler(pred_seq_1, pred_seq_2):
    """
    :param
    pred_seq_1: sequence predicted by actor net 1
    pred_seq_2: sequence predicted by actor net 2
    :returns
    final_seq: final sequence after score
    """
    final_seq = []
    for seq_1, seq_2 in zip(pred_seq_1, pred_seq_2):
        score = [0] * len(seq_1)
        for i in range(len(seq_1)):
            score[seq_1[i]] += i
            score[seq_2[i]] += i
        final_seq.append(np.argsort(score).tolist())
    return final_seq


class PSNet:
    def __init__(self, cfg, model, device):
        # actor net 1
        act_net_1 = ActorNet(cfg)
        if model:
            act_net_1.load_state_dict(model['actor'])
        self.act_optim_1 = optim.Adam(act_net_1.parameters(), lr=cfg.lr if not model else model['lr'])
        self.act_lr_scheduler_1 = optim.lr_scheduler.StepLR(self.act_optim_1,
                                                            step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        self.act_net_1 = act_net_1.to(device)

        # critic net l
        cri_net_1 = CriticNet(cfg)
        if model:
            cri_net_1.load_state_dict(model['critic'])
        self.cri_optim_1 = optim.Adam(cri_net_1.parameters(), lr=cfg.lr if not model else model['lr'])
        self.cri_lr_scheduler_1 = optim.lr_scheduler.StepLR(self.cri_optim_1,
                                                            step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        self.cri_net_1 = cri_net_1.to(device)

        # actor net 2
        act_net_2 = ActorNet(cfg)
        if model:
            act_net_2.load_state_dict(model['actor2'])
        self.act_optim_2 = optim.Adam(act_net_2.parameters(), lr=cfg.lr if not model else model['lr'])
        self.act_lr_scheduler_2 = optim.lr_scheduler.StepLR(self.act_optim_2,
                                                            step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        self.act_net_2 = act_net_2.to(device)

        # critic net 2
        cri_net_2 = CriticNet(cfg)
        if model:
            cri_net_2.load_state_dict(model['critic2'])
        self.cri_optim_2 = optim.Adam(cri_net_2.parameters(), lr=cfg.lr if not model else model['lr'])
        self.cri_lr_scheduler_2 = optim.lr_scheduler.StepLR(self.cri_optim_2,
                                                            step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        self.cri_net_2 = cri_net_2.to(device)

        self.mse_loss = nn.MSELoss()

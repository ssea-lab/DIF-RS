import os

import torch
import torch.nn as nn
import torch.optim as optim
from algorithms.rlpnet.actor import ActorNet
from algorithms.rlpnet.critic import CriticNet
import logging


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

    return RLPNet(cfg, model, device)


class RLPNet:
    def __init__(self, cfg, model, device):
        # actor net
        act_net = ActorNet(cfg)
        if model:
            act_net.load_state_dict(model['actor'])
        self.act_optim = optim.Adam(act_net.parameters(), lr=cfg.lr if not model else model['lr'])
        self.act_lr_scheduler = optim.lr_scheduler.StepLR(self.act_optim,
                                                          step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        self.act_net = act_net.to(device)

        # critic net l
        cri_net = CriticNet(cfg)
        if model:
            cri_net.load_state_dict(model['critic'])
        self.cri_optim = optim.Adam(cri_net.parameters(), lr=cfg.lr if not model else model['lr'])
        self.cri_lr_scheduler = optim.lr_scheduler.StepLR(self.cri_optim,
                                                          step_size=cfg.lr_decay_step, gamma=cfg.lr_decay)
        self.cri_net = cri_net.to(device)

        self.mse_loss = nn.MSELoss()

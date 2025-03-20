import torch

from algorithms.rlpnet.config import Config, load_pkl, pkl_parser

from algorithms.rlpnet.rlpnet import model_init


def sampling(task_list, cfg, dataset):
    torch.cuda.empty_cache()
    device = torch.device('cuda:{}'.format(cfg.cuda_dv) if cfg.type == 'gpu' and torch.cuda.is_available() else 'cpu')

    rlpnet = model_init(cfg, device, dataset, 'test')
    task_list = torch.tensor(task_list, dtype=torch.float32, device=device).repeat(1, 1, 1)

    pred_seqs, _ = rlpnet.act_net(task_list, device)

    return pred_seqs[0]


def get_idx_list(task_list, dataset):
    cfg = load_pkl(pkl_parser().path)
    return sampling(task_list, cfg, dataset)

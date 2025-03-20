import torch

from algorithms.psnet.config import Config, load_pkl, pkl_parser

from algorithms.psnet.psnet import model_init, score_handler


def sampling(task_list, cfg, dataset):
    torch.cuda.empty_cache()
    device = torch.device('cuda:{}'.format(cfg.cuda_dv) if cfg.type == 'gpu' and torch.cuda.is_available() else 'cpu')

    psnet = model_init(cfg, device, dataset, 'test')
    task_list = torch.tensor(task_list, dtype=torch.float32, device=device).repeat(1, 1, 1)

    pred_seq_1, _ = psnet.act_net_1(task_list, device)
    pred_seq_2, _ = psnet.act_net_2(task_list, device)
    actor_seq = score_handler(pred_seq_1, pred_seq_2)
    return actor_seq[0]


def get_idx_list(task_list, dataset):
    cfg = load_pkl(pkl_parser().path)
    return sampling(task_list, cfg, dataset)

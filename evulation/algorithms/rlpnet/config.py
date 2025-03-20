import pickle
import os
import argparse


def argparser():
    parser = argparse.ArgumentParser()
    # main parts
    parser.add_argument('-tp', '--type', metavar='TP', type=str, default='gpu', choices=['gpu', 'cpu'],
                        help='gpu or cpu for train and test')
    parser.add_argument('-b', '--batch', metavar='B', type=int, default=128, help='batch size, default: 128')
    parser.add_argument('-t', '--task_n', metavar='T', type=int, default=100,
                        help='number of tasks, default: 100')
    parser.add_argument('-s', '--steps', metavar='S', type=int, default=10000,
                        help='training steps(epochs), default: 10000')

    # details
    parser.add_argument('-e', '--embed', metavar='EM', type=int, default=128, help='embedding size')
    parser.add_argument('-hi', '--hidden', metavar='HI', type=int, default=128, help='hidden size')
    parser.add_argument('-c', '--clip_logits', metavar='C', type=int, default=10,
                        help='improve exploration; clipping logits')
    parser.add_argument('-st', '--softmax_T', metavar='ST', type=float, default=1.0,
                        help='might improve exploration; softmax temperature default 1.0 but 2.0, 2.2 and 1.5 might yield better results')
    parser.add_argument('-o', '--optim', metavar='O', type=str, default='Adam', help='torch optimizer')
    parser.add_argument('-minv', '--init_min', metavar='MINV', type=float, default=-0.08,
                        help='initialize weight minimun value -0.08~')
    parser.add_argument('-maxv', '--init_max', metavar='MAXV', type=float, default=0.08,
                        help='initialize weight ~0.08 maximum value')
    parser.add_argument('-ng', '--n_glimpse', metavar='NG', type=int, default=1, help='how many glimpse function')
    parser.add_argument('-np', '--n_process', metavar='NP', type=int, default=3,
                        help='how many process step in critic; at each process step, use glimpse')
    parser.add_argument('-dt', '--decode_type', metavar='DT', type=str, default='sampling',
                        choices=['greedy', 'sampling'], help='how to choose next task in actor model')

    # train, learning rate
    parser.add_argument('--lr', metavar='LR', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--lr_decay', metavar='LRD', type=float, default=0.96,
                        help='learning rate scheduler, decay by a factor of 0.96 ')
    parser.add_argument('--lr_decay_step', metavar='LRDS', type=int, default=500,
                        help='learning rate scheduler, decay every 500 steps')

    # path
    parser.add_argument('-mp', '--model_path', metavar='MP', default='/ssd_data/lzw/DIF-RS/evulation/algorithms/rlpnet/model/{}/rlpnet.ckpt',
                        type=str, help='model save path')
    parser.add_argument('-pp', '--pkl_path', metavar='PP', type=str, default='/ssd_data/lzw/DIF-RS/evulation/algorithms/rlpnet/config/config.pkl',
                        help='pkl save path')

    # GPU
    parser.add_argument('-cd', '--cuda_dv', metavar='CD', type=str, default='0',
                        help='os CUDA_VISIBLE_DEVICE, default single GPU')
    args = parser.parse_known_args()[0]
    return args


class Config:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v
        self.samples = self.batch * self.steps


def print_cfg(cfg):
    print(''.join('%s: %s\n' % item for item in vars(cfg).items()))


def dump_pkl(args):
    cfg = Config(**vars(args))
    with open(cfg.pkl_path, 'wb') as f:
        pickle.dump(cfg, f)
        print('--- save pickle file in %s ---\n' % cfg.pkl_path)
        print_cfg(cfg)


def load_pkl(pkl_path):
    if not os.path.isfile(pkl_path):
        raise FileNotFoundError('pkl_path')
    with open(pkl_path, 'rb') as f:
        cfg = pickle.load(f)
        os.environ['CUDA_VISIBLE_DEVICE'] = cfg.cuda_dv
    return cfg


def pkl_parser():
    parser = argparse.ArgumentParser()
    file_path = '/ssd_data/lzw/DIF-RS/evulation/algorithms/rlpnet/config/config.pkl'
    parser.add_argument('-p', '--path', metavar='P', type=str,
                        default=file_path, help='pkl file path')
    args = parser.parse_known_args()[0]
    return args


if __name__ == '__main__':
    args = argparser()
    dump_pkl(args)

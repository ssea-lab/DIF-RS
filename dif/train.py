"""The handler for training and evaluation."""

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from argparse import ArgumentParser

import torch
import wandb
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.utilities import rank_zero_info
from pl_edge_model import EDGEModel
import sys
from algorithms.rlpnet.config import Config
import random
import numpy as np
import pandas as pd

def arg_parser():
  parser = ArgumentParser(description='Train a Pytorch-Lightning diffusion model on a EDGE dataset. Use the SA dataset and categorical ')
  parser.add_argument('--task', type=str,default="edge")
  parser.add_argument('--storage_path', type=str, default="/ssd_data/lzw/DIF-RS/")
  parser.add_argument('--training_split', type=str, default='data/train/SA/google_cluster_trace_50.txt')
  parser.add_argument('--training_split_label_dir', type=str, default=None,
                      help="Directory containing labels for training split.")
  parser.add_argument('--validation_split', type=str, default='data/val/SA/google_cluster_trace_50.txt')
  parser.add_argument('--test_split', type=str, default='data/test/SA/google_cluster_trace_50.txt')
  parser.add_argument('--validation_examples', type=int, default=2) # max = 2500
  parser.add_argument('--dataset_size', type=float, default=1) # 设置读取数据的比例
  parser.add_argument('--dataset', type=str, default="google_cluster_trace")
  parser.add_argument('--task_n', type=int, default=50)

  parser.add_argument('--batch_size', type=int, default=32)
  parser.add_argument('--num_epochs', type=int, default=100)

  parser.add_argument('--learning_rate', type=float, default=0.0001) # 0.0002
  parser.add_argument('--weight_decay', type=float, default=0.0001) # 0.0001
  parser.add_argument('--lr_scheduler', type=str, default='cosine-decay') #cosine-decay

  parser.add_argument('--num_workers', type=int, default=32)
  parser.add_argument('--fp16', action='store_false')
  parser.add_argument('--use_activation_checkpoint', action='store_true')

  parser.add_argument('--diffusion_type', type=str, default='categorical')
  parser.add_argument('--diffusion_schedule', type=str, default='linear') # 换其它的？
  parser.add_argument('--diffusion_steps', type=int, default=1000)
  parser.add_argument('--sequential_sampling', type=int, default=1)

  parser.add_argument('--parallel_sampling', type=int, default=1)
  parser.add_argument('--inference_diffusion_steps', type=int, default=1)

  parser.add_argument('--inference_schedule', type=str, default='cosine') # 换其它的？
  parser.add_argument('--inference_trick', type=str, default="ddim")


  parser.add_argument('--n_layers', type=int, default=12)
  parser.add_argument('--hidden_dim', type=int, default=256)
  parser.add_argument('--sparse_factor', type=int, default=-1)
  parser.add_argument('--aggregation', type=str, default='sum')
  parser.add_argument('--two_opt_iterations', type=int, default=1000)
  parser.add_argument('--save_numpy_heatmap', action='store_true')

  parser.add_argument('--project_name', type=str, default='edge_diffusion')
  parser.add_argument('--wandb_entity', type=str, default=None)
  parser.add_argument('--wandb_logger_name', type=str, default="edge_diffusion_graph_categorical_edge50_SA_Google_samples4")
  parser.add_argument("--resume_id", type=str, default=None, help="Resume training on wandb.")
  parser.add_argument('--ckpt_path', type=str, default="/ssd_data/lzw/DIF-RS/models/edge_diffusion_graph_categorical_edge50_SA_Google_samples4/sfxn51lr/checkpoints/last.ckpt")
  parser.add_argument('--resume_weight_only', action='store_true')
  # 为true时，不写--do_train为false
  parser.add_argument('--do_train', action='store_true')
  parser.add_argument('--do_test', action='store_true')
  parser.add_argument('--do_test_only', action='store_true')
  parser.add_argument('--robustness_test', action='store_true')
  parser.add_argument('--robust_del_num', type=int, default=16)
  parser.add_argument('--algorithms_file_path', type=str, default='/ssd_data/lzw/DIF-RS/dif/algs')
  args = parser.parse_args()
  return args


def main(args):
  # 设置随机种子
  seed = 42
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.cuda.manual_seed_all(seed)  # 如果使用多块 GPU
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True  # 确保结果的确定性
  torch.backends.cudnn.benchmark = False  # 确保结果的稳定性

  epochs = args.num_epochs
  project_name = args.project_name

  if args.task == 'tsp':
    model_class = TSPModel
    saving_mode = 'min'
  elif args.task == 'mis':
    model_class = MISModel
    saving_mode = 'max'
  elif args.task == 'edge':
    model_class = EDGEModel
    saving_mode = 'min'
  else:
    raise NotImplementedError

  model = model_class(param_args=args)
  os.environ["WANDB_API_KEY"] ="***********************"
  # os.environ["WANDB_MODE"] = "offline"

  wandb_id = os.getenv("WANDB_RUN_ID") or wandb.util.generate_id()
  # export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
  # echo "WANDB_ID is $WANDB_RUN_ID"
  wandb_logger = WandbLogger(
      name=args.wandb_logger_name,
      project=project_name,
      entity=args.wandb_entity,
      save_dir=os.path.join(args.storage_path, f'models'),
      id=args.resume_id or wandb_id,
  )
  rank_zero_info(f"Arguments: {args}")
  rank_zero_info(f"Logging to {wandb_logger.save_dir}/{wandb_logger.name}/{wandb_logger.version}")
  print(os.path.join(wandb_logger.save_dir,
                           args.wandb_logger_name,
                           wandb_logger._id,
                           'checkpoints'))
  checkpoint_callback = ModelCheckpoint(
      monitor='val/gap', mode=saving_mode,
      save_top_k=1, save_last=True,
      dirpath=os.path.join(wandb_logger.save_dir,
                           args.wandb_logger_name,
                           wandb_logger._id,
                           'checkpoints'),
                           filename="last"
  )
  lr_callback = LearningRateMonitor(logging_interval='step')

  trainer = Trainer(
      # gpus=1,
      accelerator="auto",
      devices=torch.cuda.device_count() if torch.cuda.is_available() else None,
      max_epochs=epochs,
      callbacks=[TQDMProgressBar(refresh_rate=5), checkpoint_callback, lr_callback],
      logger=wandb_logger,
      check_val_every_n_epoch=1,
      # val_check_interval=9999,
      strategy=DDPStrategy(static_graph=True),
      precision=16 if args.fp16 else 32,
      log_every_n_steps = 1,
      num_sanity_val_steps=0
  )

  # rank_zero_info(
  #     f"{'-' * 100}\n"
  #     f"{str(model.model)}\n"
  #     f"{'-' * 100}\n"
  # )
  test_results = None
  ckpt_path = args.ckpt_path
  if args.do_train:
    if args.resume_weight_only:
      model = model_class.load_from_checkpoint(ckpt_path, param_args=args)
      trainer.fit(model)
    else:
      trainer.fit(model, ckpt_path=None)
    if args.do_test:
      test_results = trainer.test(ckpt_path=checkpoint_callback.best_model_path)
  elif args.do_test_only:
    test_results = trainer.test(model, ckpt_path=ckpt_path)
  if test_results:
    df = pd.DataFrame(test_results)
    df.to_csv(os.path.join(wandb_logger.save_dir,args.wandb_logger_name,f'samples_{args.parallel_sampling}--infer_step_{args.inference_diffusion_steps}.csv'), index=False)
    print(f"Test results saved to {os.path.join(wandb_logger.save_dir,args.wandb_logger_name,f'samples_{args.parallel_sampling}--infer_step_{args.inference_diffusion_steps}.csv')}")
  
  trainer.logger.finalize("success")


if __name__ == '__main__':
  args = arg_parser()
  main(args)

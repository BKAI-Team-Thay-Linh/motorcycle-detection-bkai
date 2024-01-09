from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, random_split

import mortobike_project as mp
import torch
import wandb
import pytorch_lightning as pl
import argparse

torch.multiprocessing.set_sharing_strategy('file_system')
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='swinv2',
                    help='model name')
parser.add_argument('--name', type=str, default=None,
                    help='name of the experiment')
parser.add_argument('--folder', type=str, default='default',
                    help='folder name')
parser.add_argument('--max_epochs', '-me', type=int, default=20,
                    help='max epoch')
parser.add_argument('--batch_size', '-bs', type=int, default=64,
                    help='batch size')
parser.add_argument('--lr', '-l', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--num_workers', '-nw', type=int, default=0,
                    help='number of workers')
parser.add_argument('--seed', '-s', type=int, default=42,
                    help='seed')
parser.add_argument('--wandb', '-w', default=False, action='store_true',
                    help='use wandb or not')
parser.add_argument('--wandb_key', '-wk', type=str,
                    help='wandb API key')
args = parser.parse_args()


def train(args, model_name):
    pl.seed_everything(args.seed, workers=True)

    # Wandb (Optional)
    if args.wandb:
        wandb.login(key=args.wandb_key)
        name = f"{model_name}-{args.max_epochs}-{args.batch_size}-{args.lr}"
        logger = WandbLogger(
            project='BKAI-mortobike-detector',
            name=name,
            log_model='all'  # Log model checkpoint at the end of training
        )

    else:
        logger = None

    # Dataloader

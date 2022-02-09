"""Experiment-running framework."""
import argparse
import importlib
from pathlib import Path
import os
import math
import sys

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
# wandb.init(project='lit')

from data import *
from models import *

import warnings
warnings.simplefilter('ignore')


# In order to ensure reproducible experiments, we must set random seeds.
np.random.seed(42)
torch.manual_seed(42)

PROJECT='ubm'
MODEL_PATH = './models/'


# def _import_class(module_and_class_name: str) -> type:
#     """Import class from a module, e.g. 'models.MLP'"""
#     print(module_and_class_name)
#     module_name, class_name = module_and_class_name.rsplit(".", 1)
#     print(module_name, class_name)
#     module = importlib.import_module(module_name)
#     # print(dir(module))
#     class_ = getattr(module, class_name)
#     return class_


def _setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)

    # Add Trainer specific arguments, such as --max_epochs, --gpus, --precision
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = "Trainer Args"  # pylint: disable=protected-access
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])
    
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--project", type=str, default=PROJECT)
    parser.add_argument("--model_path" ,type=str, default=MODEL_PATH)

    # Basic arguments
    parser.add_argument("--mixup", type=float, default=0)
    parser.add_argument("--load_checkpoint", type=str, default=None)
    parser.add_argument("--seed", type=int, default=1234)

    #model chechpoints 
    # parser.add_argument("--min_val_loss", action="store_true", default=False)
    # parser.add_argument("--k", type=int, default=1)
    # parser.add_argument("--max_top5", action="store_true", default=False)
    parser.add_argument("--last_epoch", action="store_true", default=False)

    # Get the data and model classes, so that we can add their specific arguments
    temp_args, _ = parser.parse_known_args()
    print(temp_args)
    # print(f'model_call {temp_args.model_class}')
    # data_class = _import_class(f"data.{temp_args.data_class}")
    data_class = WinDM
    # model_class = _import_class(f"models.{temp_args.model_class}")
    # model_class = _import_class(f"tsai.models.all.{temp_args.model_class}")
    model_class = LitModel

    # Get data, model, and LitModel specific arguments
    data_group = parser.add_argument_group("Data Args")
    data_class.add_to_argparse(data_group)

    model_group = parser.add_argument_group("Model Args")
    model_class.add_to_argparse(model_group)
    # add_to_argparse(temp_args.model_class, model_group)

    # lit_model_group = parser.add_argument_group("LitModel Args")
    # lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument("--help", "-h", action="help")
    return parser


def main():
    """
    Run an experiment.

    Sample command:
    ```
    python training/run_experiment.py --one_cycle_max_lr=3e-5 --wandb --loss=double_loss_squared --config_id=bets_ts_basic_10c_2y_ah_opp --gpus=1 --num_workers=4 --magnitude  0.2 0.2 0.2 0.2 --batch_size=512 --accumulate_grad_batches=4 --one_cycle_total_steps=100 --weight_decay=3e-2 --n_transforms 0 0 0 3 --augments all --augments all --augments all --augments all --data_class=TSBasic --model_class=TSTMult
    ```
    """
    parser = _setup_parser()
    args = parser.parse_args()
    print('args:')
    print(args)
    print('args.data_class:')
    print(args.win_len)
    print(args.time_win_len)
    seed=args.seed
    pl.utilities.seed.seed_everything(seed)

    dm = WinDM(args)
    dm.setup()
    model = MLP_Time(args.num_features, 1, 0, n_hidden=args.n_hidden,
            dropout=args.dropout, layer_bn=args.layer_bn)
    print(model)
    # sys.exit()
    lm = LitModel(model, args)

    logger=pl.loggers.WandbLogger(project=args.project)
    logger.log_hyperparams(vars(args))
    model_checkpoint_callback_last = pl.callbacks.ModelCheckpoint(dirpath=args.model_path,
        filename="last-{epoch:03d}-{val_loss:.3f}-{pearson:.3f}", monitor=None
        )
    callbacks = [model_checkpoint_callback_last]

    trainer = pl.Trainer.from_argparse_args(args, callbacks=callbacks, logger=logger,
         weights_save_path='./wandb/models')
    trainer.fit(lm, dm)






if __name__ == "__main__":
    main()
    

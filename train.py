import hydra
import torch
import pytorch_lightning as pl
import numpy as np
import logging
import random

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data import REDataModule
from model import T5FineTuneModel


def set_seed(seed):
    logging.info(f'Setting random seed: {seed}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.info('Program start')
    set_seed(cfg.train.random_seed)
    model = T5FineTuneModel(
        model_name=cfg.train.t5.model,
        lr_rate=cfg.train.optimizer.lr_rate,
        eps=cfg.train.optimizer.eps,
        num_training_step=cfg.train.min_steps
    )
    data_module = REDataModule(
        model_name=cfg.train.t5.model,
        train_path=cfg.train.dataset.train_path,
        valid_path=cfg.train.dataset.valid_path,
        batch_size=cfg.train.dataset.batch_size,
        max_token=cfg.train.dataset.max_token,
        num_workers=cfg.train.dataset.num_workers,
        weighted=cfg.train.dataset.weighted_data,
        alpha=cfg.train.dataset.weight_alpha
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    num_train_ex = len(open(cfg.train.dataset.train_path).readlines())
    min_steps = num_train_ex // cfg.train.dataset.batch_size // cfg.train.accumulate_grad_batches * cfg.train.min_epochs
    logging.info(f'Min training steps: {min_steps}')
    if cfg.train.fp_16:
        precision = 16
    else:
        precision = 32
    trainer = Trainer(
                      logger=logger,
                      min_steps=min_steps,
                      num_sanity_val_steps=2,
                      gpus=cfg.train.gpus,
                      accumulate_grad_batches=cfg.train.accumulate_grad_batches,
                      gradient_clip_val=cfg.train.max_grad_norm,
                      precision=precision
    )
    trainer.fit(model=model, datamodule=data_module)

if __name__ == '__main__':
    main()

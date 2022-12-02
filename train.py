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
    model = T5FineTuneModel(cfg.train.t5.model)
    data_module = REDataModule(
        model_name=cfg.train.t5.model,
        train_path=cfg.train.dataset.train_path,
        valid_path=cfg.train.dataset.valid_path,
        batch_size=cfg.train.batch_size
    )
    logger = TensorBoardLogger("tb_logs", name="my_model")
    trainer = Trainer(logger=logger,
                      max_epochs=cfg.train.total_steps,
                      num_sanity_val_steps=2,
                      gpus=cfg.train.gpus,
                      accumulate_grad_batches=4,
                      gradient_clip_val=1.0
    )
    trainer.fit(model=model, datamodule=data_module)

if __name__ == '__main__':
    main()

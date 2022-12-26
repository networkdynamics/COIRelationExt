import json
import logging

import hydra
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from data import Example, REDataset, collate_fn
from model import T5FineTuneModel


@hydra.main(version_base=None, config_path='config', config_name='config')
def main(cfg: DictConfig) -> None:
    logging.info('Prediction start')
    model = T5FineTuneModel(
        model_name=cfg.predict.t5.model,
        max_length=cfg.predict.max_length,
        beam=cfg.predict.beam
    )
    trainer = Trainer(gpus=cfg.predict.gpus)
    predict_ex = []
    with open(cfg.predict.dataset.predict_path) as f:
        for line in f:
            jsonline = json.loads(line)
            ex = Example(question=jsonline['question'], answer=jsonline['answer'])
            predict_ex.append(ex)
    predict_data = REDataset(max_token=cfg.predict.dataset.max_token, model_name=cfg.predict.t5.model)
    if cfg.predict.dataset.debug:
        predict_ex = predict_ex[:8]
    predict_data.init_data(predict_ex)
    dataloader = DataLoader(predict_data, shuffle=False, pin_memory=True,
                            num_workers=cfg.predict.dataset.num_workers, persistent_workers=True,
                            collate_fn=collate_fn, batch_size=cfg.predict.dataset.batch_size)
    if cfg.predict.ckpt_path.strip() == '':
        predictions = trainer.predict(model=model,
                                      dataloaders=dataloader)
    else:
        predictions = trainer.predict(model=model,
                                      dataloaders=dataloader,
                                      ckpt_path=cfg.predict.ckpt_path)
    open(cfg.predict.dataset.prediction_path, 'w') \
        .write('\n'.join([output for batch in predictions for output in batch]))


if __name__ == '__main__':
    main()

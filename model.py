import gc
from collections import Counter
from typing import Union, Any, Callable, Optional

import pytorch_lightning as pl
import torch.optim
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup


class T5FineTuneModel(pl.LightningModule):

    def __init__(self, model_name, lr_rate, eps, num_training_step):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.lr_rate = lr_rate
        self.eps = eps
        self.num_training_step = num_training_step

    def get_dataset(tokenizer, type_path, args):
        pass

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, lm_labels=None):
        return self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                          labels=lm_labels)

    def get_obj(self):
        objs = []
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    objs.append((type(obj), obj.size()))
            except:
                pass
        return Counter(objs)

    def training_step(self, batch, batch_idx):
        model_output = self(
            input_ids=batch['src_tensor'],
            attention_mask=batch['attention_mask'],
            lm_labels=batch['tgt_tensor'],
            decoder_attention_mask=batch['tgt_attention_mask'])
        return model_output.loss

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean().detach()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self(
            input_ids=batch['src_tensor'],
            attention_mask=batch['attention_mask'],
            lm_labels=batch['tgt_tensor'],
            decoder_attention_mask=batch['tgt_attention_mask'])
        return {"val_loss": loss}

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.lr_rate, eps=self.eps)
        self.lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=self.num_training_step
        )
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch: int,
                       batch_idx: int,
                       optimizer: Union[Optimizer, LightningOptimizer],
                       optimizer_idx: int = 0,
                       optimizer_closure: Optional[Callable[[], Any]] = None,
                       on_tpu: bool = False,
                       using_native_amp: bool = False,
                       using_lbfgs: bool = False):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

        self.lr_scheduler.step()

from typing import Union, Any, Callable, Optional

import pytorch_lightning as pl
from pytorch_lightning.core.optimizer import LightningOptimizer
from torch.optim import Optimizer
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW, get_linear_schedule_with_warmup


class T5FineTuneModel(pl.LightningModule):

    def __init__(self, model_name, lr_rate=None, eps=None, num_training_step=None, max_length=None, beam=None):
        super().__init__()
        self.opt = None
        self.lr_scheduler = None
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name
        )
        # self.model.gradient_checkpointing_enable()
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        if lr_rate:
            self.lr_rate = lr_rate
        if eps:
            self.eps = eps
        if num_training_step:
            self.num_training_step = num_training_step
        if max_length:
            self.max_length = max_length
        if beam:
            self.beam = beam

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, lm_labels=None):
        return self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                          labels=lm_labels, use_cache=False)

    def training_step(self, batch, batch_idx):
        model_output = self(
            input_ids=batch['src_tensor'],
            attention_mask=batch['attention_mask'],
            lm_labels=batch['tgt_tensor'],
            decoder_attention_mask=batch['tgt_attention_mask'])
        return model_output.loss

    def validation_step(self, batch, batch_idx):
        model_output = self(
            input_ids=batch['src_tensor'],
            attention_mask=batch['attention_mask'],
            lm_labels=batch['tgt_tensor'],
            decoder_attention_mask=batch['tgt_attention_mask'])
        self.log("val_loss", model_output.loss)
        return {"val_loss": model_output.loss}

    def predict_step(self, batch, batch_idx, dataloader_idx: int = 0) -> Any:
        outputs = self.model.generate(batch['src_tensor'], num_beams=self.beam, max_new_tokens=self.max_length)
        preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return preds

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
                       optimizer_closure: Optional[Callable[[], Any]] = None, ):
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()
        self.lr_scheduler.step()

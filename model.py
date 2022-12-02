import pytorch_lightning as pl
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from transformers import T5ForConditionalGeneration, T5Tokenizer


class T5FineTuneModel(pl.LightningModule):

    def __init__(self, t5_model_name):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(t5_model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_model_name)

    def get_dataset(tokenizer, type_path, args):
        pass

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, lm_labels=None):
        return self.model(input_ids, attention_mask=attention_mask, decoder_input_ids=decoder_input_ids,
                labels=lm_labels)

    def training_step(self, batch, batch_idx):
        loss = self(
            input_ids=batch['src_tensor'],
            attention_mask=batch['attention_mask'],
            lm_labels=batch['tgt_tensor'],
            decoder_attention_mask=batch['tgt_attention_mask'])
        return loss

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
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
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


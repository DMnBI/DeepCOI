import math
import torch
import torch.nn as nn
import pytorch_lightning as pl

from transformers import (
    EsmModel,
    EsmForMaskedLM,
    EsmConfig,
)
from transformers.modeling_outputs import MaskedLMOutput
from transformers.optimization import AdamW

from .lr_scheduler import InverseSqrtScheduler


def gelu(x):
    """
    This is the gelu implementation from the original ESM repo. Using F.gelu yields subtly wrong results.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class EsmLMHead(nn.Module):
    """ESM Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x) + self.bias
        return x


class EsmAbstractModel(pl.LightningModule):
    def __init__(self,
        model_name_or_path: str, 
        learning_rate: float, 
        warmup_steps: int,
        adam_beta1: float, 
        adam_beta2: float, 
        adam_epsilon: float,
        **kwargs):
        super().__init__()

        self.save_hyperparameters()

        self.config = EsmConfig.from_pretrained(
            model_name_or_path, return_dict=True)

    def forward(self, input_ids, attention_mask=None, labels=None):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(),
                          self.hparams.learning_rate,
                          betas=(self.hparams.adam_beta1,
                                 self.hparams.adam_beta2),
                          eps=self.hparams.adam_epsilon,)
        scheduler = InverseSqrtScheduler(optimizer, self.hparams.warmup_steps)
        sch_config = {
            "scheduler": scheduler,
            "interval": "step",
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": sch_config,
        }


class EsmMLMModel(EsmAbstractModel):
    def __init__(self,
        **kwargs):
        super().__init__(**kwargs)

        self.model = EsmForMaskedLM(self.config)

    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids, 
            attention_mask=attention_mask, 
            labels=labels
        ).logits

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('train_loss', loss, on_step=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log('valid_loss', loss, on_step=True, sync_dist=True) 
        

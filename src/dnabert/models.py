from dbtk._utils import export
from dbtk.data import tokenizers, vocabularies
from dbtk.nn import layers
from dbtk.nn.models import bert

import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional

@export
class DnaBertModel(L.LightningModule):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        kmer: int = 1,
        kmer_stride: int = 1
    ):
        super().__init__()
        self.transformer_encoder = transformer_encoder
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.tokenizer=tokenizers.DnaTokenizer(kmer, kmer_stride)
        self.vocabulary=bert.BertVocabulary(vocabularies.dna(kmer))
        self.token_embeddings = nn.Embedding(
            num_embeddings=len(self.vocabulary),
            embedding_dim=self.embed_dim,
            padding_idx=0)
        self.return_class_embeddings = True
        self.return_item_embeddings = True

    def forward(
        self,
        src: torch.Tensor,
        average_attention_weights: bool = True,
        return_attention_weights: bool = False,
        **kwargs
    ):
        # Construct input
        src = F.pad(src, (1, 0), mode="constant", value=self.vocabulary["[CLS]"])
        mask = (src == self.vocabulary["[PAD]"])
        src = self.token_embeddings(src)

        # Pass through transformer encoder
        output = self.transformer_encoder(
            src,
            src_key_padding_mask=mask,
            average_attention_weights=average_attention_weights,
            return_attention_weights=return_attention_weights,
            **kwargs)

        # Construct outputs
        if isinstance(output, tuple):
            output, *extra = output
        else:
            extra = ()
        # (class_tokens, output_tokens, extra)
        # (class_tokens, (output_tokens_a, output_tokens_b), extra)
        result = ()
        if self.return_class_embeddings:
            result += (output.select(-2, 0),)
        if self.return_item_embeddings:
            result += (output.narrow(-2, 1, src.shape[-2] - 1),)
        result += extra
        if len(result) == 1:
            return result[0]
        return result

    @property
    def embed_dim(self):
        return self.transformer_encoder.embed_dim

    def outputs(self, class_embeddings: Optional[bool] = None, item_embeddings: Optional[bool] = None) -> "BertModel":
        """
        Configure what the model should return.
        """
        if class_embeddings is not None:
            self.return_class_embeddings = class_embeddings
        if item_embeddings is not None:
            self.return_item_embeddings = item_embeddings
        return self


@export
class DnaBertPretrainingModel(L.LightningModule):
    def __init__(self, base: DnaBertModel):
        super().__init__()
        self.base = base
        self.predict_tokens = nn.Linear(
            self.embed_dim,
            len(self.base.vocabulary))

    def forward(
        self,
        src: torch.Tensor
    ):
        return self.base(src)

    def _step(self, mode, batch):
        src, masked_tokens = batch
        class_tokens, output = self(src_a)

        indices = torch.where(src.flatten() == self.base.vocabulary["[MASK]"])
        predicted = self.predict_tokens(output.flatten(0, -2)[indices])
        loss = F.cross_entropy(predicted, masked_tokens)
        num_correct = torch.sum(torch.argmax(predicted, dim=-1) == masked_tokens)

        n = masked_tokens.shape[-1]
        self.log(f"{mode}/loss", loss, prog_bar=True)
        self.log(f"{mode}/reconstruction_accuracy", num_correct.float() / n, prog_bar=True)
        return loss

    def training_step(self, batch):
        return self._step("train", batch)

    def validation_step(self, batch):
        return self._step("val", batch)

    def test_step(self, batch):
        return self._step("test", batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4) # type: ignore

    @property
    def embed_dim(self):
        return self.base.embed_dim

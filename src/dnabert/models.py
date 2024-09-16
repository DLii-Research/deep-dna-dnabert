from dbtk._utils import export
from dbtk.data import tokenizers, vocabularies
from dbtk.nn import layers
from dbtk.nn.models import bert

@export
class DnaBertModel(bert.BertModel):
    def __init__(
        self,
        transformer_encoder: layers.TransformerEncoder,
        kmer: int = 1,
        kmer_stride: int = 1
    ):
        super().__init__(
            transformer_encoder,
            tokenizer=tokenizers.DnaTokenizer(kmer, kmer_stride),
            vocabulary=bert.BertVocabulary(vocabularies.dna(kmer))
        )
        self.kmer = kmer
        self.kmer_stride = kmer_stride

@export
class DnaBertPretrainingModel(bert.BertPretrainingModel):
    ...

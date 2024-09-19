from lightning.pytorch.cli import LightningArgumentParser, LightningCLI
from dnabert import DnaBertPretrainingModel
from dnabert.datamodules import DnaBertPretrainingDataModule

class ExtendedLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser):
        parser.link_arguments("model.base.tokenizer", "data.tokenizer", apply_on="instantiate")
        parser.link_arguments("model.base.vocabulary", "data.vocabulary", apply_on="instantiate")
        parser.link_arguments("model.base.kmer", "data.kmer", apply_on="instantiate")
        parser.link_arguments("model.base.kmer_stride", "data.kmer_stride", apply_on="instantiate")

def main():
    cli = ExtendedLightningCLI(
        DnaBertPretrainingModel,
        DnaBertPretrainingDataModule,
        save_config_kwargs={"overwrite": True},
        parser_kwargs={
            "parser_mode": "omegaconf"
        }
    )

if __name__ == "__main__":
    main()

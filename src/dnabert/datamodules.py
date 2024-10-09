from dbtk.data.datasets import SequenceDataset, SequenceTaxonomyDataset
from dbtk.data.vocabularies import Vocabulary
from dnadb import fasta, taxonomy
import lightning as L
from pathlib import Path
import torch
import torch.nn.functional as F
from typing import Callable, Optional, Union


class DnaBertPretrainingDataModule(L.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer: Callable,
        train_sequences_path: Union[str, Path],
        test_sequences_path: Optional[Union[str, Path]] = None,
        val_split: float = 0.0,
        min_length: int = 65,
        max_length: int = 250,
        kmer: int = 1,
        kmer_stride: int = 1,
        mask_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.train_sequences_path = train_sequences_path
        self.test_sequences_path = test_sequences_path
        self.val_split = val_split
        self.min_length = min_length
        self.max_length = max_length
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _transform(self, fasta_entry):
        # Sequence
        sequence = fasta_entry.sequence

        # Randomly trim the sequence such that the resulting kmer-sequence length
        # is within the minimum/maximum sequence length range.
        min_length = min(self.min_length*self.kmer_stride - 1 + self.kmer, len(sequence))
        max_length= min(self.max_length*self.kmer_stride - 1 + self.kmer, len(sequence))
        length = torch.randint(min_length, max_length, size=(1,)).item()
        offset = torch.randint(0, len(sequence) - length + 1, size=(1,)).item()
        sequence = torch.tensor(list(self.vocabulary(self.tokenizer(sequence[offset:offset+length]))))

        # Masking
        mask_length = torch.randint(1, int(len(sequence)*self.mask_ratio) + 1, size=(1,)).item()
        mask_offset = torch.randint(0, len(sequence) - mask_length + 1, size=(1,)).item()
        masked_tokens = sequence[mask_offset:mask_offset+mask_length].clone()
        sequence[mask_offset:mask_offset+mask_length] = self.vocabulary["[MASK]"]

        # Padding
        sequence = F.pad(sequence, (0, self.max_length - len(sequence)), value=self.vocabulary["[PAD]"])

        return sequence, masked_tokens

    def collate(self, batch):
        sequences, masked_tokens = zip(*batch)
        # (src_a, mask_a), (src_b, mask_b), same_sequence
        return (torch.stack(sequences), torch.cat(masked_tokens)), None, None

    def setup(self, stage: str):
        if stage == "fit":
            self.train_sequences = SequenceDataset(
                self.train_sequences_path,
                transform=self._transform
            )
            num_val_sequences = int(len(self.train_sequences) * self.val_split)
            self.train_sequences, self.val_sequences = torch.utils.data.random_split(
                self.train_sequences,
                [len(self.train_sequences) - num_val_sequences, num_val_sequences],
                generator=torch.Generator()
            )

        elif stage == "test":
            self.test_sequences = SequenceDataset(
                self.test_sequences_path,
                transform=self._transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_sequences,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=True,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_sequences,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=False,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_sequences,
            batch_size=self.batch_size,
            collate_fn=self.collate,
            shuffle=False,
            num_workers=self.num_workers
        )


class DnaBertTaxonomyDataModule(L.LightningDataModule):
    def __init__(
        self,
        vocabulary: Vocabulary,
        tokenizer: Callable,
        train_sequences_path: Union[str, Path],
        train_taxonomies_path: Union[str, Path],
        test_sequences_path: Optional[Union[str, Path]] = None,
        test_taxonomies_path: Optional[Union[str, Path]] = None,
        val_split: float = 0.0,
        min_length: int = 65,
        max_length: int = 250,
        kmer: int = 1,
        kmer_stride: int = 1,
        mask_ratio: float = 0.15,
        batch_size: int = 32,
        num_workers: int = 0
    ):
        super().__init__()
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.train_sequences_path = train_sequences_path
        self.train_taxonomies_path = train_taxonomies_path
        self.test_sequences_path = test_sequences_path
        self.test_taxonomies_path = test_taxonomies_path
        self.val_split = val_split
        self.min_length = min_length
        self.max_length = max_length
        self.kmer = kmer
        self.kmer_stride = kmer_stride
        self.mask_ratio = mask_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

    def _sequence_transform(self, fasta_entry: fasta.FastaEntry):
        # Sequence
        sequence = fasta_entry.sequence

        # Randomly trim the sequence such that the resulting kmer-sequence length
        # is within the minimum/maximum sequence length range.
        min_length = min(self.min_length*self.kmer_stride - 1 + self.kmer, len(sequence))
        max_length= min(self.max_length*self.kmer_stride - 1 + self.kmer, len(sequence))
        length = torch.randint(min_length, max_length, size=(1,)).item()
        offset = torch.randint(0, len(sequence) - length + 1, size=(1,)).item()
        sequence = torch.tensor(list(self.vocabulary(self.tokenizer(sequence[offset:offset+length]))))

        # Padding
        sequence = F.pad(sequence, (0, self.max_length - len(sequence)), value=self.vocabulary["[PAD]"])

        return sequence

    def _taxonomy_transform(self, taxonomy_entry: taxonomy.TaxonomyDbEntry):
        return torch.tensor(taxonomy_entry.taxonomy.taxonomy_ids)

    def _collate(self, batch):
        sequences, taxonomies = zip(*batch)
        sequences = torch.stack(sequences)
        taxonomies = torch.stack(taxonomies)
        taxonomies = taxonomies.permute(-1, *torch.arange(taxonomies.ndim - 1))
        return sequences, taxonomies

    def setup(self, stage: str):
        if stage == "fit":
            self.train_data = SequenceTaxonomyDataset(
                self.train_sequences_path,
                self.train_taxonomies_path,
                sequence_transform=self._sequence_transform,
                taxonomy_transform=self._taxonomy_transform
            )
            num_val_data = int(len(self.train_data) * self.val_split)
            self.train_data, self.val_data = torch.utils.data.random_split(
                self.train_data,
                [len(self.train_data) - num_val_data, num_val_data],
                generator=torch.Generator()
            )

        elif stage == "test":
            self.test_data = SequenceTaxonomyDataset(
                self.test_sequences_path,
                self.test_taxonomies_path,
                sequence_transform=self._sequence_transform,
                taxonomy_transform=self._taxonomy_transform
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate,
            num_workers=self.num_workers
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate,
            num_workers=self.num_workers
        )
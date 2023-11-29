"""Source Data Mock."""
from typing import TypedDict

import torch

from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TokenizedPrompts


TEST_CONTEXT_SIZE: int = 4


class MockHuggingFaceDatasetItem(TypedDict):
    """Mock Hugging Face dataset item typed dict."""

    text: str
    meta: dict


class MockSourceDataset(SourceDataset[MockHuggingFaceDatasetItem]):
    """Mock source dataset for testing the inherited abstract dataset."""

    def preprocess(
        self,
        source_batch: MockHuggingFaceDatasetItem,  # noqa: ARG002
        *,
        context_size: int,  # noqa: ARG002
    ) -> TokenizedPrompts:
        """Preprocess a batch of prompts."""
        preprocess_batch = 100
        tokenized_texts = torch.randint(
            low=0, high=50000, size=(preprocess_batch, TEST_CONTEXT_SIZE)
        ).tolist()
        return {"input_ids": tokenized_texts}

    def __init__(
        self,
        dataset_path: str = "mock_dataset_path",
        dataset_split: str = "test",
        context_size: int = TEST_CONTEXT_SIZE,
        buffer_size: int = 1000,
        preprocess_batch_size: int = 1000,
    ):
        """Initialise the dataset."""
        super().__init__(
            dataset_path,
            dataset_split,
            context_size,
            buffer_size,
            preprocess_batch_size,
        )

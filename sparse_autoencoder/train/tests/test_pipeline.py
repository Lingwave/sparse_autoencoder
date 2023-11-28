"""Tests for Pipeline."""
from pathlib import Path
from typing import Any, TypedDict

from datasets import IterableDataset, load_dataset
import pytest
import torch
from transformer_lens import HookedTransformer

from sparse_autoencoder.activation_resampler.abstract_activation_resampler import (
    AbstractActivationResampler,
)
from sparse_autoencoder.activation_store.tensor_store import TensorActivationStore
from sparse_autoencoder.autoencoder.model import SparseAutoencoder
from sparse_autoencoder.loss.tests.test_abstract_loss import DummyLoss
from sparse_autoencoder.metrics.metrics_container import MetricsContainer, default_metrics
from sparse_autoencoder.optimizer.adam_with_reset import AdamWithReset
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset, TokenizedPrompts
from sparse_autoencoder.train.pipeline import Pipeline


TEST_CONTEXT_SIZE: int = 4
src_model = HookedTransformer.from_pretrained("gelu-2l")


def create_pipeline(
    activation_resampler: AbstractActivationResampler | None = None,
    layer: int = 1,
    checkpoint_directory: Path | None = None,
    log_frequency: int = 100,
    metrics: MetricsContainer = default_metrics,
    source_data_batch_size: int = 12,
) -> Pipeline:
    geometric_median = torch.tensor([1.0, 2.0, 3.0])
    model = SparseAutoencoder(3, 6, geometric_median)

    optimizer = AdamWithReset(
        model.parameters(),
        named_parameters=model.named_parameters(),
    )

    return Pipeline(
        activation_resampler=activation_resampler,
        autoencoder=model,
        cache_name="blocks.0.hook_mlp_out",
        layer=1,
        loss=DummyLoss(),
        optimizer=optimizer,
        source_dataset=MockSourceDataset(),
        source_model=src_model,
        checkpoint_directory=checkpoint_directory,
        log_frequency=log_frequency,
        metrics=metrics,
        source_data_batch_size=12,
    )


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


@pytest.fixture()
def mock_hugging_face_load_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the `load_dataset` function from Hugging Face.

    Instead load the text data from mocks/text_dataset.txt, using a restored `load_dataset` method.
    """

    def mock_load_dataset(*args: Any, **kwargs: Any) -> IterableDataset:  # noqa:   ANN401
        """Mock load dataset function."""
        mock_path = Path(__file__).parent / "mocks" / "text_dataset.txt"
        return load_dataset(
            "text", data_files={"train": [str(mock_path)]}, streaming=True, split="train"
        )  # type: ignore

    monkeypatch.setattr(
        "sparse_autoencoder.source_data.abstract_dataset.load_dataset", mock_load_dataset
    )


@pytest.fixture()
def minimal_pipeline(mock_hugging_face_load_dataset: pytest.Function) -> Pipeline:
    return create_pipeline()


@pytest.mark.parametrize("store_size", [1, 50, 0, -5])
def test_generate_activations(minimal_pipeline: Pipeline, store_size: int) -> None:
    result = minimal_pipeline.generate_activations(store_size)
    print(result)


def test_train_autoencoder(minimal_pipeline: Pipeline) -> None:
    activation_store = TensorActivationStore(
        max_items=10,
        num_neurons=3,
    )
    result = minimal_pipeline.train_autoencoder(
        activation_store=activation_store, train_batch_size=40
    )
    print(result)

"""Test the abstract dataset."""
from pathlib import Path
from typing import Any

from datasets import IterableDataset, load_dataset
import pytest

from sparse_autoencoder.mocks.mock_source_data import TEST_CONTEXT_SIZE, MockSourceDataset
from sparse_autoencoder.source_data.abstract_dataset import SourceDataset


@pytest.fixture()
def mock_hugging_face_load_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mock the `load_dataset` function from Hugging Face.

    Instead load the text data from mocks/text_dataset.txt, using a restored `load_dataset` method.
    """

    def mock_load_dataset(*args: Any, **kwargs: Any) -> IterableDataset:  # noqa:   ANN401
        """Mock load dataset function."""
        mock_path = Path(__file__).parent.parent.parent / "mocks" / "data" / "text_dataset.txt"
        return load_dataset(
            "text", data_files={"train": [str(mock_path)]}, streaming=True, split="train"
        )  # type: ignore

    monkeypatch.setattr(
        "sparse_autoencoder.source_data.abstract_dataset.load_dataset", mock_load_dataset
    )


def test_extended_dataset_initialization(mock_hugging_face_load_dataset: pytest.Function) -> None:
    """Test the initialization of the extended dataset."""
    data = MockSourceDataset()
    assert data is not None
    assert isinstance(data, SourceDataset)


def test_extended_dataset_iterator(mock_hugging_face_load_dataset: pytest.Function) -> None:
    """Test the iterator of the extended dataset."""
    data = MockSourceDataset()
    iterator = iter(data)
    assert iterator is not None

    first_item = next(iterator)
    assert len(first_item["input_ids"]) == TEST_CONTEXT_SIZE


def test_get_dataloader(mock_hugging_face_load_dataset: pytest.Function) -> None:
    """Test the get_dataloader method of the extended dataset."""
    data = MockSourceDataset()
    batch_size = 3
    dataloader = data.get_dataloader(batch_size=batch_size)
    first_item = next(iter(dataloader))["input_ids"]
    assert first_item.shape[0] == batch_size
    assert first_item.shape[-1] == TEST_CONTEXT_SIZE

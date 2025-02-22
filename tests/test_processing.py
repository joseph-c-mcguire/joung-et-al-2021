from io import StringIO

import pytest
import pandas as pd
import numpy as np

from src.data.processing import load_and_clean_data, smiles_to_adjacency


@pytest.fixture
def sample_data():
    return """col1\tcol2\tcol3
1\t2\t3
4\t5\t6
7\t8\t9"""


@pytest.fixture
def filepath(tmp_path, sample_data):
    file = tmp_path / "test_data.tsv"
    file.write_text(sample_data)
    return file


def test_load_and_clean_data(filepath, sample_data):
    # Use StringIO to simulate file reading
    test_data = StringIO(sample_data)
    expected_df = pd.read_csv(test_data, sep='\t')

    # Load and clean data using the function
    result_df = load_and_clean_data(filepath)

    # Check if the loaded DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_valid_smiles():
    smiles = "CCO"
    result = smiles_to_adjacency(smiles)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3
    assert result.shape[1] == 3


def test_invalid_smiles():
    smiles = "invalid_smiles"
    result = smiles_to_adjacency(smiles)
    assert result is None


def test_empty_smiles():
    smiles = ""
    result = smiles_to_adjacency(smiles)
    assert result is None

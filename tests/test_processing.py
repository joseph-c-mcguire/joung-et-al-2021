from io import StringIO

import pytest
import pandas as pd
import numpy as np
from rdkit import Chem

from src.data.processing import *


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
    result_df = fetch_and_prepare_data(filepath)

    # Check if the loaded DataFrame matches the expected DataFrame
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_valid_smiles():
    smiles = "CCO"
    result = convert_smiles_to_adjacency_matrix(smiles)
    assert isinstance(result, np.ndarray)
    assert result.shape[0] == 3
    assert result.shape[1] == 3


def test_invalid_smiles():
    smiles = "invalid_smiles"
    result = convert_smiles_to_adjacency_matrix(smiles)
    assert result is None


def test_empty_smiles():
    smiles = ""
    result = convert_smiles_to_adjacency_matrix(smiles)
    # Assert they're equal to the empty np array
    assert np.all(result == np.array([]))


def test_get_atom_features():
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles("CCO")
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get features for the first atom
    features = get_atom_features(atom)

    # Check the length of the features list
    # 118 for atomic number + 2 Neighbors Features + 1 Aromaticity + 3 Hybridization + 1 Ring + 1 Formal Charge
    assert len(features) == 126

    # Check one-hot encoding for atomic number
    # Carbon has atomic number 6, so index 5 should be 1
    assert features[5] == 1
    assert sum(features[:118]) == 1  # Only one position should be 1

    # Check total number of hydrogen atoms attached
    # Carbon in "CCO" has 3 hydrogen atoms attached
    assert features[118] == 3

    # Check number of heavy atom neighbors
    # Carbon in "CCO" has 1 heavy atom neighbor (another Carbon)
    assert features[119] == 1

    # Check aromaticity
    assert features[120] == 0  # Carbon in "CCO" is not aromatic

    # Check hybridization
    assert features[121] == 0  # SP hybridization
    assert features[122] == 0  # SP2 hybridization
    assert features[123] == 1  # SP3 hybridization

    # Check if atom is in a ring
    assert features[124] == 0  # Carbon in "CCO" is not in a ring

    # Check formal charge
    assert features[125] == 0  # Carbon in "CCO" has no formal charge


def test_get_atomic_number_features():
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles("CCO")
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get atomic number features for the first atom
    features = get_atomic_number_features(atom)

    # Check the length of the features list
    assert len(features) == 118

    # Check one-hot encoding for atomic number
    # Carbon has atomic number 6, so index 5 should be 1
    assert features[5] == 1
    assert sum(features) == 1  # Only one position should be 1

    # Test for another atom (Oxygen)
    atom = mol.GetAtomWithIdx(2)  # Get the third atom (Oxygen)
    features = get_atomic_number_features(atom)

    # Oxygen has atomic number 8, so index 7 should be 1
    assert features[7] == 1
    assert sum(features) == 1  # Only one position should be 1

    # Test for Hydrogen (implicit in the molecule)
    mol = Chem.AddHs(mol)
    atom = mol.GetAtomWithIdx(3)  # Get the fourth atom (Hydrogen)
    features = get_atomic_number_features(atom)

    # Hydrogen has atomic number 1, so index 0 should be 1
    assert features[0] == 1
    assert sum(features) == 1  # Only one position should be 1


def test_get_hydrogen_and_neighbors_features():
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles("CCO")
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get hydrogen and neighbors features for the first atom
    features = get_hydrogen_and_neighbors_features(atom)

    # Check the length of the features list
    assert len(features) == 2

    # Check total number of hydrogen atoms attached
    # Carbon in "CCO" has 3 hydrogen atoms attached
    assert features[0] == 3

    # Check number of heavy atom neighbors
    # Carbon in "CCO" has 1 heavy atom neighbor (another Carbon)
    assert features[1] == 1

    # Test for another atom (Oxygen)
    atom = mol.GetAtomWithIdx(2)  # Get the third atom (Oxygen)
    features = get_hydrogen_and_neighbors_features(atom)

    # Oxygen in "CCO" has 1 hydrogen atom attached
    assert features[0] == 1

    # Oxygen in "CCO" has 1 heavy atom neighbor (Carbon)
    assert features[1] == 1

    # Test for Hydrogen (implicit in the molecule)
    mol = Chem.AddHs(mol)
    atom = mol.GetAtomWithIdx(3)  # Get the fourth atom (Hydrogen)
    features = get_hydrogen_and_neighbors_features(atom)

    # Hydrogen in "CCO" has 0 hydrogen atoms attached
    assert features[0] == 0

    # Hydrogen in "CCO" has 1 heavy atom neighbor (Carbon)
    assert features[1] == 1


def test_get_aromaticity_feature():
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene, which is aromatic
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get aromaticity feature for the first atom
    features = get_aromaticity_feature(atom)

    # Check the length of the features list
    assert len(features) == 1

    # Check aromaticity
    assert features[0] == 1  # Carbon in benzene is aromatic

    # Test for a non-aromatic molecule
    mol = Chem.MolFromSmiles("CCO")  # Ethanol, which is not aromatic
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get aromaticity feature for the first atom
    features = get_aromaticity_feature(atom)

    # Check the length of the features list
    assert len(features) == 1

    # Check aromaticity
    assert features[0] == 0  # Carbon in ethanol is not aromatic

    # Test for another non-aromatic molecule
    mol = Chem.MolFromSmiles("C1CCCCC1")  # Cyclohexane, which is not aromatic
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get aromaticity feature for the first atom
    features = get_aromaticity_feature(atom)

    # Check the length of the features list
    assert len(features) == 1

    # Check aromaticity
    assert features[0] == 0  # Carbon in cyclohexane is not aromatic


def test_get_hybridization_features():
    # Create a molecule from SMILES
    mol = Chem.MolFromSmiles("CCO")
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get hybridization features for the first atom
    features = get_hybridization_features(atom)

    # Check the length of the features list
    assert len(features) == 3

    # Check hybridization
    assert features[0] == 0  # SP hybridization
    assert features[1] == 0  # SP2 hybridization
    assert features[2] == 1  # SP3 hybridization

    # Test for another atom (Oxygen)
    atom = mol.GetAtomWithIdx(2)  # Get the third atom (Oxygen)
    features = get_hybridization_features(atom)

    # Oxygen in "CCO" is SP3 hybridized
    assert features[0] == 0  # SP hybridization
    assert features[1] == 0  # SP2 hybridization
    assert features[2] == 1  # SP3 hybridization

    # Test for an SP2 hybridized atom
    mol = Chem.MolFromSmiles("C=C")  # Ethene, which has SP2 hybridized carbons
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)
    features = get_hybridization_features(atom)

    assert features[0] == 0  # SP hybridization
    assert features[1] == 1  # SP2 hybridization
    assert features[2] == 0  # SP3 hybridization

    # Test for an SP hybridized atom
    mol = Chem.MolFromSmiles("C#C")  # Ethyne, which has SP hybridized carbons
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)
    features = get_hybridization_features(atom)

    assert features[0] == 1  # SP hybridization
    assert features[1] == 0  # SP2 hybridization
    assert features[2] == 0  # SP3 hybridization


def test_get_ring_and_charge_features():
    # Test for a molecule with a ring (Benzene)
    mol = Chem.MolFromSmiles("c1ccccc1")  # Benzene
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get ring and charge features for the first atom
    features = get_ring_and_charge_features(atom)

    # Check the length of the features list
    assert len(features) == 2

    # Check if the atom is in a ring
    assert features[0] == 1  # Carbon in benzene is in a ring

    # Check formal charge
    assert features[1] == 0  # Carbon in benzene has no formal charge

    # Test for a molecule without a ring (Ethanol)
    mol = Chem.MolFromSmiles("CCO")  # Ethanol
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Carbon)

    # Get ring and charge features for the first atom
    features = get_ring_and_charge_features(atom)

    # Check the length of the features list
    assert len(features) == 2

    # Check if the atom is in a ring
    assert features[0] == 0  # Carbon in ethanol is not in a ring

    # Check formal charge
    assert features[1] == 0  # Carbon in ethanol has no formal charge

    # Test for a molecule with a charged atom (Ammonium ion)
    mol = Chem.MolFromSmiles("[NH4+]")  # Ammonium ion
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Nitrogen)

    # Get ring and charge features for the first atom
    features = get_ring_and_charge_features(atom)

    # Check the length of the features list
    assert len(features) == 2

    # Check if the atom is in a ring
    assert features[0] == 0  # Nitrogen in ammonium is not in a ring

    # Check formal charge
    assert features[1] == 1  # Nitrogen in ammonium has a formal charge of +1

    # Test for a molecule with a negatively charged atom (Hydroxide ion)
    mol = Chem.MolFromSmiles("[OH-]")  # Hydroxide ion
    atom = mol.GetAtomWithIdx(0)  # Get the first atom (Oxygen)

    # Get ring and charge features for the first atom
    features = get_ring_and_charge_features(atom)

    # Check the length of the features list
    assert len(features) == 2

    # Check if the atom is in a ring
    assert features[0] == 0  # Oxygen in hydroxide is not in a ring

    # Check formal charge
    assert features[1] == -1  # Oxygen in hydroxide has a formal charge of -1


def test_convert_smiles_to_matrices_valid_smiles():
    smiles = "CCO"
    adj_matrix, feature_matrix = convert_smiles_to_matrices(smiles)

    assert isinstance(adj_matrix, np.ndarray)
    assert isinstance(feature_matrix, np.ndarray)
    assert adj_matrix.shape == (150, 150)
    assert feature_matrix.shape == (150, feature_matrix.shape[1])

    # Check if the original adjacency matrix is correctly placed in the padded matrix
    mol = Chem.MolFromSmiles(smiles)
    original_adj_matrix = create_adjacency_matrix(mol)
    assert np.array_equal(adj_matrix[:original_adj_matrix.shape[0],
                          :original_adj_matrix.shape[1]], original_adj_matrix)

    # Check if the original feature matrix is correctly placed in the padded matrix
    original_feature_matrix = extract_atom_feature_matrix(mol)
    assert np.array_equal(feature_matrix[:original_feature_matrix.shape[0],
                          :original_feature_matrix.shape[1]], original_feature_matrix)


def test_convert_smiles_to_matrices_invalid_smiles():
    smiles = "invalid_smiles"
    adj_matrix, feature_matrix = convert_smiles_to_matrices(smiles)

    assert adj_matrix is None
    assert feature_matrix is None


def test_convert_smiles_to_matrices_empty_smiles():
    smiles = ""
    adj_matrix, feature_matrix = convert_smiles_to_matrices(smiles)

    assert adj_matrix is None
    assert feature_matrix is None


def test_convert_smiles_to_matrices_large_molecule():
    smiles = "C" * 200  # A large molecule with 200 carbon atoms
    adj_matrix, feature_matrix = convert_smiles_to_matrices(smiles)

    assert isinstance(adj_matrix, np.ndarray)
    assert isinstance(feature_matrix, np.ndarray)
    assert adj_matrix.shape == (150, 150)
    assert feature_matrix.shape == (150, feature_matrix.shape[1])

    # Check if the original adjacency matrix is correctly placed in the padded matrix
    mol = Chem.MolFromSmiles(smiles)
    original_adj_matrix = create_adjacency_matrix(mol)
    assert np.array_equal(adj_matrix[:original_adj_matrix.shape[0],
                          :original_adj_matrix.shape[1]], original_adj_matrix[:150, :150])

    # Check if the original feature matrix is correctly placed in the padded matrix
    original_feature_matrix = extract_atom_feature_matrix(mol)
    assert np.array_equal(feature_matrix[:original_feature_matrix.shape[0],
                          :original_feature_matrix.shape[1]], original_feature_matrix[:150, :])

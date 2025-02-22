"""
This module provides functions for processing molecular data, including feature extraction from atoms,
conversion of SMILES strings to adjacency and feature matrices, and data cleaning and imputation.
Functions:
    get_atom_features(atom: Chem.rdchem.Atom) -> list:
    convert_smiles_to_matrices(smiles: str, max_size: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    create_adjacency_matrix(mol: Chem.rdchem.Mol) -> np.ndarray:
    extract_atom_feature_matrix(mol: Chem.rdchem.Mol) -> np.ndarray:
    resize_and_pad_matrix(matrix: np.ndarray, max_size: int, feature_size: Optional[int] = None) -> np.ndarray:
    fetch_and_prepare_data(filepath: str) -> pd.DataFrame:
    convert_smiles_to_adjacency_matrix(smiles: str) -> np.ndarray:
    fill_missing_numerics(df: pd.DataFrame) -> pd.DataFrame:
    convert_smiles_to_adjacency_matrices(df: pd.DataFrame) -> pd.DataFrame:
    load_and_process_dataset(filepath: str) -> pd.DataFrame:
"""

from typing import Tuple, Optional
import pandas as pd
import numpy as np
from rdkit import Chem


def get_atom_features(atom: Chem.rdchem.Atom) -> list:
    """
    Generate a list of features for a given atom.

    Args:
        atom (rdkit.Chem.rdchem.Atom): The atom for which features are to be generated.

    Returns:
    list: A list of features representing the atom, including:
        - One-hot encoding of the atomic number (length 118).
        - Total number of hydrogen atoms attached to the atom.
        - Number of heavy atom neighbors (atomic number > 1).
        - Aromaticity (1 if aromatic, 0 otherwise).
        - Hybridization type (SP, SP2, SP3) as a list of binary values.
        - Whether the atom is in a ring (1 if in a ring, 0 otherwise).
        - Formal charge of the atom.
    """

    features = []

    # Atom identity (one-hot encoding of atomic number)
    atomic_num = atom.GetAtomicNum()
    atom_features = np.zeros(118)  # Max atomic number
    atom_features[atomic_num-1] = 1
    features.extend(atom_features.tolist())

    features.extend(
        (
            atom.GetTotalNumHs(),
            len([n for n in atom.GetNeighbors() if n.GetAtomicNum() > 1]),
            int(atom.GetIsAromatic()),
        )
    )
    # Hybridization
    hyb_types = [Chem.rdchem.HybridizationType.SP,
                 Chem.rdchem.HybridizationType.SP2,
                 Chem.rdchem.HybridizationType.SP3]
    features.extend([int(atom.GetHybridization() == hyb) for hyb in hyb_types])

    features.extend((int(atom.IsInRing()), atom.GetFormalCharge()))
    return features


def convert_smiles_to_matrices(smiles: str, max_size: int = 150) -> Tuple[np.ndarray, np.ndarray]:
    """
    Converts a SMILES string to its corresponding adjacency and feature matrices,
    with optional padding to a specified maximum size.

    Args:
    smiles (str): The SMILES string representing the molecular structure.
    max_size (int, optional): The maximum size for padding the matrices. Default is 150.

    Returns:
    tuple: A tuple containing the padded adjacency matrix and the padded feature matrix.
           If the SMILES string is invalid or an error occurs, returns (None, None).
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None

        adj_matrix = create_adjacency_matrix(mol)
        feature_matrix = extract_atom_feature_matrix(mol)

        padded_adj = resize_and_pad_matrix(adj_matrix, max_size)
        padded_features = resize_and_pad_matrix(
            feature_matrix, max_size, feature_matrix.shape[1])

        return padded_adj, padded_features
    except Exception:
        return None, None


def create_adjacency_matrix(mol: Chem.rdchem.Mol) -> np.ndarray:
    """
    Generate the adjacency matrix for a given molecule.

    Args:
        mol (rdkit.Chem.rdchem.Mol): The molecule for which to generate the adjacency matrix.

    Returns:
        numpy.ndarray: The adjacency matrix of the molecule.
    """
    return Chem.GetAdjacencyMatrix(mol)


def extract_atom_feature_matrix(mol: Chem.rdchem.Mol) -> np.ndarray:
    """
    Get feature matrix from molecule.

    Parameters:
        mol (rdkit.Chem.Mol): The molecule from which to extract the feature matrix.

    Returns:
        numpy.ndarray: A 2D array where each row represents the features of an atom in the molecule.
    """
    return np.array([get_atom_features(atom) for atom in mol.GetAtoms()])


def resize_and_pad_matrix(matrix: np.ndarray, max_size: int, feature_size: Optional[int] = None) -> np.ndarray:
    """
    Pads a given matrix to a specified maximum size.

    Args:
        matrix (numpy.ndarray): The input matrix to be padded.
        max_size (int): The desired size to pad the matrix to. If the matrix is larger than this size, it will be truncated.
        feature_size (int, optional): The number of features for each row. If provided, the matrix will be padded/truncated accordingly.

    Returns:
        numpy.ndarray: The padded (or truncated) matrix with dimensions (max_size, max_size) or (max_size, feature_size).
    """
    n_atoms = matrix.shape[0]
    if n_atoms >= max_size:
        return matrix[:max_size, :max_size] if feature_size is None else matrix[:max_size, :]

    padded_matrix = np.zeros(
        (max_size, max_size if feature_size is None else feature_size))
    padded_matrix[:n_atoms,
                  :n_atoms] = matrix if feature_size is None else matrix[:n_atoms, :]

    return padded_matrix


def fetch_and_prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and perform initial cleaning of the dataset.

    Args:
    filepath (str): The path to the dataset file.

    Returns:
    DataFrame: A pandas DataFrame containing the cleaned dataset.
    """
    return pd.read_csv(filepath, sep='\t')


def convert_smiles_to_adjacency_matrix(smiles: str) -> np.ndarray:
    """
    Converts a SMILES string to an adjacency matrix.

    Args:
    smiles (str): A string representing the SMILES notation of a molecule.

    Returns:
    np.ndarray: A 2D numpy array representing the adjacency matrix of the molecule.
                Returns None if the SMILES string is invalid or an error occurs.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else Chem.GetAdjacencyMatrix(mol)
    except Exception:
        return None


def fill_missing_numerics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in specified numeric columns of a DataFrame using the median.

    Args:
    df (pandas.DataFrame): The input DataFrame containing the data to be imputed.

    Returns:
    pandas.DataFrame: The DataFrame with missing values imputed in the specified numeric columns.

    Notes:
    The following columns are imputed using their respective median values:
    - 'Lifetime (ns)'
    - 'Quantum yield'
    - 'log(e/mol-1 dm3 cm-1)'
    - 'abs FWHM (cm-1)'
    - 'emi FWHM (cm-1)'
    - 'abs FWHM (nm)'
    - 'emi FWHM (nm)'
    """
    # Numeric columns that need imputation
    numeric_cols = ['Lifetime (ns)', 'Quantum yield', 'log(e/mol-1 dm3 cm-1)',
                    'abs FWHM (cm-1)', 'emi FWHM (cm-1)', 'abs FWHM (nm)',
                    'emi FWHM (nm)']

    # Use median for imputation as it's more robust to outliers
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def convert_smiles_to_adjacency_matrices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes molecular data by converting SMILES strings to adjacency matrices.

    This function takes a DataFrame containing molecular data and applies the 
    `smiles_to_adjacency` function to the 'Chromophore' and 'Solvent' columns 
    to generate their respective adjacency matrices. The resulting matrices 
    are stored in new columns 'Chromophore_Matrix' and 'Solvent_Matrix'.

    Args:
    df (pandas.DataFrame): A DataFrame containing at least 'Chromophore' and 
                           'Solvent' columns with SMILES strings.

    Returns:
    pandas.DataFrame: The input DataFrame with additional columns 
                      'Chromophore_Matrix' and 'Solvent_Matrix' containing 
                      the adjacency matrices.
    """
    df['Chromophore_Adj'] = df['Chromophore'].apply(
        lambda x: convert_smiles_to_matrices(x)[0])
    df['Chromophore_Features'] = df['Chromophore'].apply(
        lambda x: convert_smiles_to_matrices(x)[1])
    df['Solvent_Adj'] = df['Solvent'].apply(
        lambda x: convert_smiles_to_matrices(x)[0])
    df['Solvent_Features'] = df['Solvent'].apply(
        lambda x: convert_smiles_to_matrices(x)[1])
    return df


def load_and_process_dataset(filepath: str) -> pd.DataFrame:
    """
    Prepares the dataset by loading, cleaning, imputing missing values, and processing molecular data.

    Args:
        filepath (str): The path to the dataset file.

    Returns:
        pandas.DataFrame: The processed dataset.
    """
    df = fetch_and_prepare_data(filepath)
    df = fill_missing_numerics(df)
    df = convert_smiles_to_adjacency_matrices(df)
    return df

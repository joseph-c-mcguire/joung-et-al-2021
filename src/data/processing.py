""" This module provides functions for loading, cleaning, and processing molecular data from a dataset file.
Functions:
- load_and_clean_data(filepath: str) -> pd.DataFrame:
- smiles_to_adjacency(smiles: str) -> np.ndarray:
- impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
- process_molecular_data(df: pd.DataFrame) -> pd.DataFrame:
- prepare_dataset(filepath: str) -> pd.DataFrame:
"""


import pandas as pd
import numpy as np
from rdkit import Chem


def load_and_clean_data(filepath):
    """
    Load and perform initial cleaning of the dataset.

    Parameters:
    filepath (str): The path to the dataset file.

    Returns:
    DataFrame: A pandas DataFrame containing the cleaned dataset.
    """
    return pd.read_csv(filepath, sep='\t')


def smiles_to_adjacency(smiles: str) -> np.ndarray:
    """
    Converts a SMILES string to an adjacency matrix.

    Parameters:
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


def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
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


def process_molecular_data(df: pd.DataFrame) -> pd.DataFrame:
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
    df['Chromophore_Matrix'] = df['Chromophore'].apply(smiles_to_adjacency)
    df['Solvent_Matrix'] = df['Solvent'].apply(smiles_to_adjacency)
    return df


def prepare_dataset(filepath: str) -> pd.DataFrame:
    """
    Prepares the dataset by loading, cleaning, imputing missing values, and processing molecular data.

    Args:
        filepath (str): The path to the dataset file.

    Returns:
        pandas.DataFrame: The processed dataset.
    """
    df = load_and_clean_data(filepath)
    df = impute_missing_values(df)
    df = process_molecular_data(df)
    return df

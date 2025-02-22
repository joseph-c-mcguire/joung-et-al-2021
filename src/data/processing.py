import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def load_and_clean_data(filepath):
    """Load and perform initial cleaning of the dataset."""
    return pd.read_csv(filepath, sep='\t')


def smiles_to_adjacency(smiles):
    """Convert SMILES string to adjacency matrix."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        return None if mol is None else Chem.GetAdjacencyMatrix(mol)
    except Exception:
        return None


def impute_missing_values(df):
    """Impute missing values in the dataset."""
    # Numeric columns that need imputation
    numeric_cols = ['Lifetime (ns)', 'Quantum yield', 'log(e/mol-1 dm3 cm-1)',
                    'abs FWHM (cm-1)', 'emi FWHM (cm-1)', 'abs FWHM (nm)',
                    'emi FWHM (nm)']

    # Use median for imputation as it's more robust to outliers
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())

    return df


def process_molecular_data(df):
    """Process molecular structures and create feature matrices."""
    df['Chromophore_Matrix'] = df['Chromophore'].apply(smiles_to_adjacency)
    df['Solvent_Matrix'] = df['Solvent'].apply(smiles_to_adjacency)
    return df


def prepare_dataset(filepath):
    """Main function to prepare the dataset."""
    df = load_and_clean_data(filepath)
    df = impute_missing_values(df)
    df = process_molecular_data(df)
    return df

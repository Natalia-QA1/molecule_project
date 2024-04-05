import re
from joblib import Parallel, delayed  # Added import for joblib
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
import math


class MoleculeProcessingException(Exception):
    pass


class MolecularPropertiesProcessor:

    def __init__(
            self,
            input_file_path: str,
            output_file_name: str,
    ):
        self.mols_df = pd.read_csv(input_file_path, encoding='windows-1252')
        self.output_file_name = output_file_name

        self.smiles_col = self._column_finder("^SMILES$")
        self.mol_name_col = self._column_finder("^Molecule Name$")

    def _column_finder(self, match_str):
        matcher = re.compile(match_str, re.IGNORECASE)
        column_to_find = next(filter(matcher.match, self.mols_df.columns))
        if not column_to_find:
            raise MoleculeProcessingException(f"No {match_str} column found in a dataframe")
        return column_to_find

    def _prepare_data(self):
        self.mols_df = self.mols_df[
            [self.smiles_col, self.mol_name_col]
            + list(self.mols_df.columns.difference([self.smiles_col, self.mol_name_col]))
        ]
        self.mols_df.drop_duplicates(subset=self.mol_name_col, inplace=True)

    def _compute_molecule_properties(self, smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return [None] * 7  # Return None for all properties if parsing failed
        return [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.MolLogP(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.RingCount(mol),
            all([
                Descriptors.MolWt(mol) < 500,
                Descriptors.MolLogP(mol) < 5,
                Descriptors.NumHDonors(mol) < 5,
                Descriptors.NumHAcceptors(mol) < 10
            ])
        ]

    def _compute_molecule_properties_chunk(
            self,
            chunk_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """ Compute molecule properties for chunk dataframe """
        properties = Parallel(n_jobs=-1)(  # Changed to use joblib for parallelization
            delayed(self._compute_molecule_properties)(smiles)
            for smiles in chunk_df[self.smiles_col]
        )

        mol_props_to_compute = [
            "Molecular weight", "TPSA", "logP", "H Acceptors", "H Donors", "Ring Count", "Lipinski pass"
        ]
        properties_df = pd.DataFrame(properties, columns=mol_props_to_compute)

        return pd.concat([chunk_df.reset_index(drop=True), properties_df], axis=1)

    def process_data(self):
        self._prepare_data()

        result = self._compute_molecule_properties_chunk(self.mols_df)
        return result


if __name__ == '__main__':
    mpp = MolecularPropertiesProcessor(
        input_file_path="../initial_datasets/molecules.csv",
        output_file_name="../output.csv",
    )

    mpp.process_data()

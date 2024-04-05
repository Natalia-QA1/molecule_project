import re
import multiprocessing
# from multiprocessing import Pool  # No longer needed
from joblib import Parallel, delayed  # Import Parallel and delayed from joblib
import pandas as pd
import numpy as np
from rdkit.Chem import AllChem, Descriptors
import math

class MoleculeProcessingException(Exception):
    pass

class MolecularPropertiesProcessor:

    def __init__(
            self,
            input_file_path: str,
            # output_file_name: str,
    ):
        self.mols_df = pd.read_csv(input_file_path, encoding='windows-1252')
        # self.output_file_name = output_file_name

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

    def _compute_molecule_properties_chunk(
            self,
            chunk_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """ Compute molecule properties for chunk dataframe """
        chunk_df["mol"] = chunk_df[self.smiles_col].apply(lambda s: AllChem.MolFromSmiles(s))
        mol_props_funcs = {
            "Molecular weight": lambda mol: Descriptors.MolWt(mol),
            "TPSA": lambda mol: Descriptors.TPSA(mol),
            "logP": lambda mol: Descriptors.MolLogP(mol),
            "H Acceptors": lambda mol: Descriptors.NumHAcceptors(mol),
            "H Donors": lambda mol: Descriptors.NumHDonors(mol),
            "Ring Count": lambda mol: Descriptors.RingCount(mol),
            "Lipinski pass": lambda mol: all([
                Descriptors.MolWt(mol) < 500,
                Descriptors.MolLogP(mol) < 5,
                Descriptors.NumHDonors(mol) < 5,
                Descriptors.NumHAcceptors(mol) < 10
            ])
        }

        mol_props_to_compute = list(mol_props_funcs.keys())
        chunk_df[mol_props_to_compute] = chunk_df.apply(
            lambda row: [mol_props_funcs[prop](row["mol"]) for prop in mol_props_to_compute],
            axis=1,
            result_type="expand"
        )

        chunk_df.drop(columns=["mol"], inplace=True)
        chunk_df.set_index(self.mol_name_col, inplace=True)

        return chunk_df

    def _compute_molecule_properties(self) -> pd.DataFrame:
        """
        Compute molecule properties and fingerprints using RDKit
        in chunks
        """
        # Define the number of parallel jobs
        num_jobs = multiprocessing.cpu_count()

        # Split the dataframe into chunks
        chunk_size = 3  # Define the chunk size
        chunks = [self.mols_df[i:i + chunk_size] for i in range(0, len(self.mols_df), chunk_size)]

        # Use joblib to parallelize the computation
        result = Parallel(n_jobs=num_jobs)(delayed(self._compute_molecule_properties_chunk)(chunk) for chunk in chunks)

        return pd.concat(result)

    def process_data(self):
        self._prepare_data()
        mol_properties_df = self._compute_molecule_properties()
        # return mol_properties_df

        # Write the DataFrame to a CSV file
        mol_properties_df.to_csv(self.output_file_name, index=True)  # Adjust index parameter as needed
        print(f"Saved DataFrame to {self.output_file_name}")

        return mol_properties_df


if __name__ == '__main__':
    mpp = MolecularPropertiesProcessor(
        input_file_path="../initial_datasets/molecules.csv")

    mpp.process_data()

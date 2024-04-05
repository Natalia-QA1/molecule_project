import re
import multiprocessing
from joblib import Parallel, delayed  # simple parallel computing
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import math
from datetime import datetime


class MoleculeProcessingException(Exception):
    pass

class MoleculeDecodingException(Exception):
    """Return special message when UnicodeDecodeError occurred

    Condition: during reading file in the  _read_input_data() method
    in case if error_status was not specified and a decoding error occurred.
    """
    def __init__(self, message, input_file_path, encoding):
        self.message = message
        self.input_file_path = input_file_path
        self.encoding = encoding

    def __str__(self):
        return f'UnicodeDecodeError occurred during reading file {self.input_file_path}\n' \
               f' with {self.encoding} encoding.\nError: {self.message}.\nTry to provide error_status parameter.'

class MolecularPropertiesProcessor:

    """Class contains methods to perform some transformations with given dataset in csv format.

    General Steps:
    1. Read csv file and create dataframe.
    2. Prepare data:
        - find corrupted rows and drop them
        - drop duplicates
        - add necessary columns
    3. Return transformed dataframe which will be written in new file by using File_writer module.
    """

    def __init__(
            self,
            input_file_path: str,
            encoding: str,  # encoding for pd.read_csv() to decode special symbols
            process_amount: int,
            chunk_size: int,
            error_file_path: str,  # Path to the file where problematic rows will be written
            error_status: str  # status for encoding_errors in pd.read_csv()
    ):
        self.input_file_path = input_file_path
        self.encoding = encoding
        self.process_amount = process_amount
        self.chunk_size = chunk_size
        self.error_file_path = error_file_path
        self.error_status = error_status

        self.smiles_col = None  # mail column to get necessary attributes (new columns) from rdkit
        self.mols_df = self._read_input_data()

    def _write_error_to_file(self, error_message):
        """Write the given error message to the error file.

        File with errors helps to determine corrupted rows."""
        error_time = datetime.now()
        with open(self.error_file_path, 'a') as f:
            f.write(f'Error time:\n{error_time}\n')
            f.write(error_message + '\n')

    def _read_input_data(self):
        """Read provided csv file with given data"""
        if self.error_status is None:
            try:
                self.mols_df = pd.read_csv(self.input_file_path, encoding=self.encoding)
                return self.mols_df
            except UnicodeDecodeError as e:
                error_message = f"UnicodeDecodeError: {e}\nProblematic Row:\n{e.object}\n\n"
                # Write the error message to the error file
                self._write_error_to_file(error_message)
                # and raise exception with contain help message to determine the problem
                raise MoleculeDecodingException(e, self.input_file_path, self.encoding)
        else:
            self.mols_df = pd.read_csv(self.input_file_path, encoding=self.encoding, encoding_errors=self.error_status)
            return self.mols_df

        if self.mols_df.empty:
            raise MoleculeProcessingException('Something went wrong.\nDataframe is empty.')


    def _column_finder(self, match_str, df):
        """'Smiles' column is the main column to find  formula in rdkit library.

         This column may have different name formats in dataframe.
         But it contains word 'smiles'.
         Method allows to find this column to perform future transformations.
         """
        matcher = re.compile(match_str, re.IGNORECASE)
        column_to_find = next(filter(matcher.match, df.columns))
        if not column_to_find:
            raise MoleculeProcessingException(f"No {match_str} column found in a dataframe")
        return column_to_find

    def _prepare_data(self):
        # TODO: think about other columns which may contain corrupted data according to requirements.
        """Prepare dataframe to adding new properties.

        Drop duplicates.
        Check corrupted data and drop it.
        """
        try:
            # find 'smiles' column in dataframe
            self.smiles_col = self._column_finder(r'smiles', self.mols_df)

            row_count_b = len(self.mols_df)
            print(f'Total row count before deleting duplicates{row_count_b}')

            self.mols_df.drop_duplicates(subset=self.smiles_col, inplace=True)

            row_count_a = len(self.mols_df)
            print(f'Total row count after deleting duplicates{row_count_a}')
        except MoleculeProcessingException as e:
            error_message = f"An error occurred during data preparation: {e}\n"
            # Write the error message to the error file
            self._write_error_to_file(error_message)

        # Handle parse errors and drop corresponding rows
        invalid_smiles_indices = []  # list to write indexes of corrupted rows
        for idx, smiles in self.mols_df[self.smiles_col].items():  # use 'smiles' col to find bad rows
            try:
                # find given molecule in rdkit
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:  # means that this molecule does not exist -> corrupted row
                    invalid_smiles_indices.append(idx)
                    error_message = f'Parsing error in row with index: {idx}\n'
                    self._write_error_to_file(error_message)
            except Exception as e:
                invalid_smiles_indices.append(idx)
                error_message = f"Error: {e}\nInvalid SMILES in row {idx}\n"
                self._write_error_to_file(error_message)
        # drop bad rows
        if invalid_smiles_indices:
            self.mols_df.drop(index=invalid_smiles_indices, inplace=True)
            print(f"Dropped {len(invalid_smiles_indices)} rows with parse errors.")
            row_count_ap = len(self.mols_df)
            print(f'Total row count after deleting currupted rows {row_count_ap}')

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
        chunk_df.set_index(self.smiles_col, inplace=True)

        return chunk_df

    def _compute_molecule_properties(self) -> pd.DataFrame:
        """
        Compute molecule properties and fingerprints using RDKit
        in chunks.
        """
        # Solution using Joblib
        # #Define the number of parallel jobs
        # num_jobs = multiprocessing.cpu_count()
        # #
        # # # Split the dataframe into chunks
        # chunk_size = self.chunk_size
        # chunks = [self.mols_df[i:i + chunk_size] for i in range(0, len(self.mols_df), chunk_size)]
        # #
        # # # Use joblib to parallelize the computation
        # prev_result = Parallel(n_jobs=num_jobs)(delayed(self._compute_molecule_properties_chunk)(chunk) for chunk in chunks)
        # # Concatenate the list of dataframes into a single dataframe
        # result = pd.concat(prev_result)
        # return result


        const_size_of_chunks = self.chunk_size
        max_amount_of_p = self.process_amount

        # calculate amount of chunks
        amount_of_chunk_df = math.ceil(len(self.mols_df) / const_size_of_chunks)

        if amount_of_chunk_df > max_amount_of_p:
            amount_of_chunk_df = max_amount_of_p - 1

        list_of_chunks = np.array_split(self.mols_df, amount_of_chunk_df)

        pool = multiprocessing.Pool(processes=amount_of_chunk_df)
        p_df = pool.map(self._compute_molecule_properties_chunk, list_of_chunks)

        list_of_p = [p for p in p_df]
        # TODO: think how to implement threading in File_Writer and Load_To_DB modules
        # ? leave chunks list for threading
        result = pd.concat(list_of_p)
        return result

    def process_data(self):
        self._prepare_data()
        try:
            if self.process_amount != 0 and self.chunk_size != 0:
                mol_properties_df = self._compute_molecule_properties()
            else:
                mol_properties_df = self._compute_molecule_properties_chunk(self.mols_df)

        except MoleculeProcessingException as e:
            print(e)
        return mol_properties_df



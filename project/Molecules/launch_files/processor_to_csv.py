import pandas as pd
import time
from processor_package.Molecular_Properties_Processor import MolecularPropertiesProcessor
from file_writer_package.File_Writer import WriteToCsv


if __name__ == '__main__':
    mpp = MolecularPropertiesProcessor(input_file_path="../initial_datasets/SMILES_Data_Set.csv",
                                       encoding='windows-1252',
                                       process_amount=4,
                                       chunk_size=5000,
                                       error_file_path='../log_files/molecule_error.txt',
                                       error_status='ignore')

    # process data and measure processing time
    start_t = time.time()
    mol_properties_df = mpp.process_data()  # Process the data
    end_t = time.time()

    # Write the processed data to a CSV file
    writer = WriteToCsv(output_file_name="../transformed_datasets/smiles_transformed.csv")
    writer.write_to_file(mol_properties_df)

    df = pd.read_csv('../transformed_datasets/smiles_transformed.csv')
    row_count_o = len(df)
    unique_smiles = df['SMILES'].nunique()
    print(f'Total row count after deleting duplicates: {row_count_o} and unique smiles: {unique_smiles}')
    # to compare result with different quantity of processes
    print('Process takes', end_t - start_t, ' seconds')
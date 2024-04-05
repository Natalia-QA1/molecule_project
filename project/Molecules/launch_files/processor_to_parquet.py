from processor_package.Molecular_Properties_Processor import MolecularPropertiesProcessor
from file_writer_package.File_Writer import WriteToParquet


if __name__ == '__main__':
    # read raw dataset
    mpp = MolecularPropertiesProcessor(input_file_path="../initial_datasets/SMILES_Data_Set.csv",
                                       encoding='windows-1252',
                                       process_amount=3,
                                       chunk_size=5000,
                                       error_file_path='../log_files/molecule_error.txt',
                                       error_status='ignore')

    # transform data and return a dadaframe
    mol_properties_df = mpp.process_data()

    # Write the processed data to a CSV file
    writer = WriteToParquet(output_file_name="../transformed_datasets/smiles_dataset_transformed.parquet")
    writer.write_to_file(mol_properties_df)
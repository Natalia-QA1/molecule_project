from load_to_database_package.Load_Data_To_DB import LoadDataToPostgres
from config.credentials_variables_postgres import username, password, host, database, port
from read_transformed_data_package.Read_transformed_Data import ReadCsv

if __name__ == '__main__':
    data_to_load = ReadCsv(file_name="../transformed_datasets/smiles_transformed.csv")

    credentials = f"postgresql://{username}:{password}@{host}:{port}/{database}"

    # Create an instance of LoadDataToPostgres
    data_loader = LoadDataToPostgres(file_name="../transformed_datasets/smiles_transformed.csv",
                                     credentials=credentials)

    # Map table columns and types
    columns_info = data_loader.map_table(data_to_load.read_file())

    # Create the table in the database
    data_loader.create_table('molecule_characteristic_3', dataframe=data_to_load.read_file(), columns=columns_info)

    # Insert data into the database
    data_loader.insert_data(table_name='molecule_characteristic_3', dataframe=data_to_load.read_file())

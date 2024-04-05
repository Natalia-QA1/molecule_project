import pandas as pd
from io import BytesIO

# with open("../transformed_datasets/molecules_transformed.parquet", 'rb') as f:
#     parquet_df = f.read()
#     parquet_bytes_io = BytesIO(parquet_df)
#     parquet_data = pd.read_parquet(parquet_bytes_io)
#
# print(parquet_data)
#
# print(parquet_data.info())

df = pd.read_csv("../initial_datasets/SMILES_Data_Set.csv")
print(df.info)
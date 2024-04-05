from abc import ABC, abstractmethod
import pandas as pd
from io import BytesIO


class ReadFile(ABC):

    def __init__(
            self, file_name: str):
        self.file_name = file_name
    """Class allows to read transformed data from different file formats.
    
    Then load them to database."""
    @abstractmethod
    def read_file(self):
        pass

class ReadCsv(ReadFile):

    def read_file(self):
        csv_data = pd.read_csv(self.file_name)
        return csv_data


class ReadParquet(ReadFile):

    def read_file(self):
        with open(self.file_name, 'rb') as f:
            parquet_df = f.read()
        parquet_bytes_io = BytesIO(parquet_df)
        parquet_data = pd.read_parquet(parquet_bytes_io)

        return parquet_data

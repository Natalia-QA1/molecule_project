# TODO: since writing data to a file is I/O task --> think about threading

from abc import ABC, abstractmethod


class FileWriter(ABC):

    def __init__( self, output_file_name: str):
        self.output_file_name = output_file_name

    @abstractmethod
    def write_to_file(self, dataframe):
        pass


class WriteToCsv(FileWriter):
    """Write output result from MolecularPropertiesProcessor class to a CSV file"""

    def write_to_file(self, dataframe):
        dataframe.to_csv(self.output_file_name, index=True)
        print(f"Saved DataFrame to {self.output_file_name}")


class WriteToParquet(FileWriter):
    """Write output result from MolecularPropertiesProcessor class to a parquet file"""

    def write_to_file(self, dataframe):
        dataframe.to_parquet(self.output_file_name, index=True)
        print(f"Saved DataFrame to {self.output_file_name}")
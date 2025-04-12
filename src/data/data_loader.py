from src.utils.public_imports import *


class DataLoader:
    """
    DataLoader class to load and preprocess data.
    """

    def __init__(self, file_path: str):
        """
        Initialize the DataLoader with a path.

        :param file_path: Path to the CSV file.
        """
        self.file_path = file_path
        self.data = None

    def load_data(self) -> pd.DataFrame:
        """
        Load data from the CSV file.

        :return: Loaded DataFrame.
        """
        try:
            self.data = pd.read_csv(self.file_path)
            return self.data
        except FileNotFoundError as e:
            print(f"File not found: {e}")
            raise

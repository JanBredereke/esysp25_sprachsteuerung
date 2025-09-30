from abc import ABC, abstractmethod
import pathlib
from typing import Tuple

import numpy as np


class AudioLoader(ABC):
    """
    Abstract class for loading the audio inputs for data synthesis.

    For each type of input (e.g. mp3, wav etc.) there is a loader which is selected based on the extension.
    """

    def __init__(self):
        pass

    @classmethod
    @abstractmethod
    def get_loader_extension(cls):
        """
        Classmethod used to identify the file extension of the files this loader loads.
        :return: The extension.
        """
        return None

    @abstractmethod
    def load_raw_data(self, file_path: pathlib.Path) -> Tuple[np.ndarray, int]:
        """
        Loads the raw data from the file with the given path.
        :param file_path: The path of the file to load relative to the project root. Has to exist.
        :return: The raw audio data as a numpy array and the sample rate..
        """
        # Check if the path exists
        if not file_path.exists():
            raise FileNotFoundError(f"Failed to load input file {file_path}")
        # This is abstract, return None
        return None

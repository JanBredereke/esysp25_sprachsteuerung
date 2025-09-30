import pathlib
from typing import Tuple

import numpy as np
from pydub import AudioSegment
from trainingData.inputLoading.audioLoaders.AudioLoader import AudioLoader


class Mp3Loader(AudioLoader):
    """
    Loads an MP3 using Pydub.
    """

    @classmethod
    def get_loader_extension(cls):
        """
        This loads mp3s.
        :return:
        """
        return "mp3"

    def load_raw_data(self, file_path: pathlib.Path) -> Tuple[np.ndarray, int]:
        """
        Loads the file from the given path and parses it as an ndarray.
        :param file_path: The file to load.
        :return: The data as an ndarray.
        """
        # Load the file
        sound = AudioSegment.from_mp3(file_path).set_channels(1)
        # Create the array and return
        return np.array(sound.get_array_of_samples()), sound.frame_rate

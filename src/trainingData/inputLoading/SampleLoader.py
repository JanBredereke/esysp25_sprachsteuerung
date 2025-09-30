import pathlib
from typing import Tuple

import numpy as np

from trainingData.inputLoading.audioLoaders.AudioLoaderFactory import (
    AudioLoaderFactory,
)


class SampleLoader:
    """
    Class used to load the inputs for the data synthesis through an AudioLoader.
    """

    def __init__(self):
        """
        Creates the sample loader.
        """
        # Init the dict of the loaders as empty
        self._audio_loaders = {}

    def load_sample_from_path(self, path: pathlib.Path) -> Tuple[np.ndarray, int]:
        """
        Helper to load a sample at the given path.
        :return: The loaded sample together with its sample rate.
        """
        # Get the extension of the sample
        extension = path.suffix[1:].lower()
        # Check if the extension is known
        if extension not in self._audio_loaders:
            # Did not use a loader for this yet, get a loader from the factory
            self._audio_loaders[extension] = AudioLoaderFactory.create(extension)
        # Use the loader to load the sample
        return self._audio_loaders[extension].load_raw_data(path)

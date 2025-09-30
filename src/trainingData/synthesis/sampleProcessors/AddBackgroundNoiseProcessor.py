import logging
import pathlib
import random
from typing import List

import numpy as np
from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.inputLoading.SampleLoader import SampleLoader
from trainingData.synthesis import SampleProcessor


class AddBackgroundNoiseProcessor(SampleProcessor):
    """
    Adds background noise to the samples.
    """

    _BACKGROUNDS_PATH_KEY = "backgrounds_path"

    def __init__(self, synthesis_config: dict):
        """
        Initialises the background processor.
        :param synthesis_config: The config for the synthesis.
        """
        # Mark that we require config to get the path of our backgrounds
        super().__init__(synthesis_config, does_require_config=True)
        # Check that the path for the backgrounds is set and exists
        self._background_samples_path = pathlib.Path(
            self.config[AddBackgroundNoiseProcessor._BACKGROUNDS_PATH_KEY]
        )
        if not self._background_samples_path.exists():
            # If the path does not exist, raise an error
            raise FileNotFoundError(
                f"The path for the background samples does not exist: {self._background_samples_path}"
            )
        # Load the background audio samples
        self._background_samples = self._load_background_samples()

    def _load_background_samples(self) -> List[np.ndarray]:
        """
        Loads the backgrounds as configured in the config.
        :return: The samples.
        """
        # The loaded backgrounds
        backgrounds = []
        # Create an Input Loader
        input_loader = SampleLoader()
        # Iterate over the files in the background samples path
        for file in self._background_samples_path.iterdir():
            # Load the sample and append to the return value
            try:
                audio_data, _ = input_loader.load_sample_from_path(file)
                backgrounds.append(audio_data)
            except Exception as e:
                # If an error occurs, log it and continue
                logging.warning(f"Failed to load background sample: {file}, Reason: {e}")
        logging.info(f"Loaded {len(backgrounds)} background samples from {self._background_samples_path}.")
        # Return
        return backgrounds

    @property
    def is_applicable_to_empty_sample(self):
        """
        Returns True if this processor can be applied to a sample only containing silence.

        This is True for all processors which add noise and do not want to change noise.
        :return: False if this is not applicable to silence, True otherwise.
        """
        return True

    @staticmethod
    def _choose_random_part_of_sample(sample, length):
        """
        Chooses a random part of a given sample with the given length.
        :param sample: The sample to choose from.
        :param length: The length.
        :return: The chosen part.
        """
        # Get the start index
        start_index = random.randint(0, len(sample) - length)
        # Return the part of the sample
        return sample[start_index : start_index + length]

    def process_sample(
        self, sample: AudioSample, do_in_place: bool = True
    ) -> AudioSample:
        """
        Adds background noise to the sample.

        Does work directly on the given sample (in-place).

        :param sample: The sample to change.
        :param do_in_place: If this is True, the sample will be modified in-place.
        :return: The changed sample.
        """
        # Call super to handle in-place or not
        sample = super().process_sample(sample, do_in_place)
        # Get a random background sample
        background_sample = random.choice(self._background_samples)
        # Check if the background is shorter than the sample, if so repeat it until it is longer
        if len(background_sample) < len(sample.data):
            # If the background is shorter, repeat it
            background_sample = np.repeat(
                background_sample, (len(sample.data) // len(background_sample) + 1)
            )
        # Make the data the same length by choosing a random part of the background
        if len(background_sample) > len(sample.data):
            # If the background is longer, choose a random region that is the same length
            background_sample = self._choose_random_part_of_sample(
                background_sample, len(sample.data)
            )
        # Add the background sample to the sample and take the average
        sample.data = np.array(
            sample.data * 0.5 + background_sample * 0.5, dtype=np.int16
        )
        # Return the sample
        return sample

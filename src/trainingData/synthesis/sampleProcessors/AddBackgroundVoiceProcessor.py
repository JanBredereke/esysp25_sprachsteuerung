import pathlib
import random

import numpy as np
from pydub import AudioSegment

from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.synthesis.sampleProcessors.AddBackgroundNoiseProcessor import (
    AddBackgroundNoiseProcessor,
)

_BACKGROUNDS_PATH_KEY = "backgrounds_path"
_MAX_BACKGROUND_SAMPLE_NUMBER = "max_background_sample_number"
_SAMPLE_WIDTH = "sample_width"
_CHANNELS = "channels"
_SAMPLE_RATE = "sample_rate_hz"


class AddBackgroundVoiceProcessor(AddBackgroundNoiseProcessor):
    """
    Adds background noise to the samples.
    """

    def __init__(self, synthesis_config: dict):
        """
        Initialises the background processor.
        :param synthesis_config: The config for the synthesis.
        """
        # Mark that we require config to get the path of our backgrounds
        super().__init__(synthesis_config)
        self._background_samples_path = self.backgrounds_path
        # Load the background audio samples
        self._background_samples = self._load_background_samples()

    @property
    def backgrounds_path(self):
        """
        Loads the backgrounds as configured in the config.
        :return: The samples.
        """
        backgrounds_path = pathlib.Path(self.config[_BACKGROUNDS_PATH_KEY])
        if not backgrounds_path.exists():
            # If the path does not exist, raise an error
            raise FileNotFoundError(
                f"The path for the background samples does not exist: {backgrounds_path}"
            )
        return backgrounds_path

    @property
    def max_background_sample_number(self):
        """
        Returns the maximum number of background samples.
        :return: The maximum number of background samples.
        """
        return self.config[_MAX_BACKGROUND_SAMPLE_NUMBER]

    @property
    def is_applicable_to_empty_sample(self):
        """
        Returns True if this processor can be applied to a sample only containing silence.

        This is True for all processors which add noise and do not want to change noise.
        :return: False if this is not applicable to silence, True otherwise.
        """
        return True

    def _load_background_samples(self):
        """
        Loads the backgrounds as configured in the config.
        :return: The samples. A list of numpy arrays.
        """
        return super()._load_background_samples()

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

        # Get the background samples to apply on sample
        number_of_samples = random.randint(0, self.max_background_sample_number)
        background_samples_to_apply = random.choices(
            self._background_samples, k=number_of_samples
        )
        # Create empty sample to be filled with background samples and overlay with
        # the original sample afterwards
        full_background_sample = np.zeros_like(sample.data, dtype=np.int16)
        # Add the background sample to the sample and take the average
        for background_sample in background_samples_to_apply:
            # Generate random start index to add background sample in
            start_index = random.randint(
                0, len(full_background_sample) - len(background_sample)
            )
            # Add background to sample at given index
            full_background_sample[
                start_index : start_index + len(background_sample)
            ] = background_sample
        # Add background sample to original sample
        sample.data = np.array(
            (0.5 * sample.data + 0.5 * full_background_sample), dtype=np.int16
        )
        # Return the sample
        return sample

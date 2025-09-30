import random
from math import sqrt

import numpy as np

from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.synthesis import SampleProcessor

_MAX_VOLUME_INCREASE_PERCENT = "max_volume_increase_percent"
_MAX_VOLUME_DECREASE_PERCENT = "max_volume_decrease_percent"


class AdjustVolumeProcessor(SampleProcessor):
    """
    Sample processor that adjusts the volume of the sample.
    """

    def __init__(self, synthesis_config: dict):
        """
        Initialises the background processor.
        :param synthesis_config: The config for the synthesis.
        """
        # Mark that we require config to get the path of our backgrounds
        super().__init__(synthesis_config, does_require_config=True)

    @property
    def max_volume_increase_percent(self):
        """
        Returns the maximum percentage of volume increase.
        :return: The maximum percentage of volume increase.
        """
        return self.config[_MAX_VOLUME_INCREASE_PERCENT]

    @property
    def max_volume_decrease_percent(self):
        """
        Returns the maximum percentage of volume decrease.
        :return: The maximum percentage of volume decrease.
        """
        return self.config[_MAX_VOLUME_DECREASE_PERCENT]

    def _adjust_volume(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        Reduces or increases the volume percentual of the given audio segment.
        The level of increase/decrease is randomly chosen in the range of the
        max_volume_increase_percent and max_volume_decrease_percent parameter
        described in the config file.

        :param audio_segment: The audio segment, that should be increased/decreased in volume
        :return: The audio segment with increased/decreased volume
        """
        volume_adjustment_level_percent = random.randint(
            -self.max_volume_decrease_percent, self.max_volume_increase_percent
        )
        # Transform volume adjustment level from absolut upper and lower percentage
        # range to decimal factor. When the resulting level is >1 the volume
        # increases,when the level is >0 and <1, the volume decreases.
        multiplier_linear = (volume_adjustment_level_percent / 100) + 1

        # convert the linear volume to a logarithmic scale
        multiplier_log = pow(2, (sqrt(sqrt(sqrt(multiplier_linear))) * 192 - 192) / 6)
        # Apply the volume adjustment to the audio_segment
        audio_segment = audio_segment * multiplier_log
        return audio_segment

    def process_sample(
        self, sample: AudioSample, do_in_place: bool = True
    ) -> AudioSample:
        """
        Reduces or increases the volume percentual of the given audio segment.
        The level of increase/decrease is randomly chosen in the range of the
        max_volume_increase_percent and max_volume_decrease_percent parameter
        described in the config file.

        Does work directly on the given sample (in-place).

        :param sample: The sample to change.
        :param do_in_place: If this is True, the sample will be modified in-place.
        :return: The changed sample.
        """
        # Call super to handle in-place or not
        sample = super().process_sample(sample, do_in_place)
        # Adjust the volume of the given sample
        volume_adjusted_audio_sample = self._adjust_volume(sample.data)
        # Transform pydub audio file to numpy array
        sample.data = np.array(volume_adjusted_audio_sample, dtype=np.int16)
        # Return the sample
        return sample

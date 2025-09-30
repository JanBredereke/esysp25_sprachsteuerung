import random
from math import sqrt

import numpy as np

from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.synthesis import SampleProcessor


_MAX_SPEED_INCREASE_PERCENT = "max_speed_increase_percent"
_MAX_SPEED_DECREASE_PERCENT = "max_speed_decrease_percent"


class AdjustPlaybackSpeedProcessor(SampleProcessor):
    """
    SampleProcessor which changes the playback speed of the audio sample.
    """

    def __init__(self, synthesis_config: dict):
        """
        Initialises the background processor.
        :param synthesis_config: The config for the synthesis.
        """
        # Mark that we require config to get the path of our backgrounds
        super().__init__(synthesis_config, does_require_config=True)

    @property
    def max_speed_increase_percent(self):
        """
        Returns the maximum percentage to increase the speed by.
        :return: The percentage.
        """
        # Get from the config
        return self.config[_MAX_SPEED_INCREASE_PERCENT]

    @property
    def max_speed_decrease_percent(self):
        """
        Returns the maximum percentage to decrease the speed by.
        :return: The percentage.
        """
        # Get from the config
        return self.config[_MAX_SPEED_DECREASE_PERCENT]

    def _adjust_speed(self, sound_array: np.ndarray) -> np.ndarray:
        """
        Reduces or increases the speed percentual of the given audio segment.
        The level of increase/decrease is randomly chosen in the range of the
        max_speed_increase_percent and max_speed_decrease_percent parameter
        described in the config file.

        :param sound_array: The audio segment, that should be increased/decreased in speed
        :return: The audio segment with increased/decreased speed
        """
        speed_adjustment_level_percent = random.randint(
            -self.max_speed_decrease_percent, self.max_speed_increase_percent
        )
        # Transform speed adjustment level from absolut upper and lower percentage
        # range to decimal factor. When the resulting level is >1 the speed
        # increases,when the level is >0 and <1, the speed decreases.
        multiplier = (speed_adjustment_level_percent / 100) + 1

        indices = np.round(np.arange(0, len(sound_array), multiplier))
        indices = indices[indices < len(sound_array)].astype(int)
        return sound_array[indices.astype(int)]

    def process_sample(
        self, sample: AudioSample, do_in_place: bool = True
    ) -> AudioSample:
        """
        Reduces or increases the speed percentual of the given audio segment.
        The level of increase/decrease is randomly chosen in the range of the
        max_speed_increase_percent and max_speed_decrease_percent parameter
        described in the config file.

        Does work directly on the given sample (in-place).

        :param sample: The sample to change.
        :param do_in_place: If this is True, the sample will be modified in-place.
        :return: The changed sample.
        """
        # Call super to handle in-place or not
        sample = super().process_sample(sample, do_in_place)
        # Adjust the speed of the given sample
        speed_adjusted_audio_sample = self._adjust_speed(sample.data)
        # Transform pydub audio file to numpy array
        sample.data = np.array(speed_adjusted_audio_sample, dtype=np.int16)
        # Return the sample
        return sample

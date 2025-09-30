import numpy as np

from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.synthesis.sampleProcessors.AdjustVolumeProcessor import (
    AdjustVolumeProcessor,
)

_MAX_VOLUME_INCREASE_PERCENT = "max_volume_increase_percent"
_MAX_VOLUME_DECREASE_PERCENT = "max_volume_decrease_percent"


class DistortionProcessor(AdjustVolumeProcessor):
    """
    Sample processor that adjusts the volume of the sample.
    """

    def __init__(self, synthesis_config: dict):
        """
        Initialises the background processor.
        :param synthesis_config: The config for the synthesis.
        """
        # Mark that we require config to get the path of our backgrounds
        super().__init__(synthesis_config)

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
        return super()._adjust_volume(audio_segment)

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
        return sample

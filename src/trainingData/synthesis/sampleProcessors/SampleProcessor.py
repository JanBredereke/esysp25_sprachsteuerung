import copy
from abc import ABC, abstractmethod

from trainingData.inputLoading.AudioSample import AudioSample


class SampleProcessor(ABC):
    """
    Class used to process a sample by applying some sort of change to it in order to diversify training data.

    This can be making it louder, changing the speed, overlaying some other audio etc.
    """

    # The default probability of applying this processor to a sample
    _DEFAULT_APPLY_PROBABILITY = 0.5
    # The config key for the apply probability
    _APPLY_PROBABILITY_KEY = "apply_probability"

    def __init__(self, synthesis_config: dict, does_require_config: bool = False):
        """
        Initialises the processor by loading its config from the given config.
        :param synthesis_config: The config to use for creation of the processor.
        :param does_require_config: True if this processor requires a part of the config to be loaded
        from the synthesis config, False otherwise.
        """
        # Init the config with None
        self._config = None
        # Check if we shall load a config
        if does_require_config:
            # Load the global config first
            self._config = copy.deepcopy(synthesis_config)
            # Update it to contain our config if it exists
            self._config.update(synthesis_config.get(self.config_key, {}))

    @property
    def config_key(self):
        """
        Returns the config key of this processor. This is just the classname.
        :return: The config key of this processor.
        """
        return self.__class__.__name__

    def update_config_from_dict(self, config: dict):
        """
        Updates the internal configuration with the part of the given configuration that is relevant for this
        processor (i.e. is returned with the config key of self).
        :param config: The config to update the internal config with.
        """
        # Check if we have a config
        if self._config is not None:
            # Update the config
            self._config.update(config.get(self.config_key, {}))

    @property
    def config(self):
        """
        Property that returns the config of this processor.
        :return: The config.
        """
        return self._config

    @property
    def apply_probability(self):
        """
        A value between 0 and 1 that specifies the probability of this processor being applied.

        These are normalized after they were gotten from all processors.
        :return: The probability.
        """
        # Return the probability
        return self.config.get(
            SampleProcessor._APPLY_PROBABILITY_KEY,
            SampleProcessor._DEFAULT_APPLY_PROBABILITY,
        )

    @property
    def is_applicable_to_empty_sample(self):
        """
        Returns True if this processor can be applied to a sample only containing silence.

        This is True for all processors which add noise and do not want to change noise.
        :return: False if this is not applicable to silence, True otherwise.
        """
        return False

    @abstractmethod
    def process_sample(
        self, sample: AudioSample, do_in_place: bool = True
    ) -> AudioSample:
        """
        Processes this audio sample.

        Does work directly on the given sample (in-place) if not specified otherwise.
        :param sample: The sample to process.
        :param do_in_place: If this is True, the sample will be modified in-place.
        :return: The processed sample.
        """
        # Check if we shall do in-place
        if not do_in_place:
            # Copy the sample
            sample = copy.deepcopy(sample)
        # Return the sample
        return sample

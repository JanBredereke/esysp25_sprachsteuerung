import copy
from abc import ABC, abstractmethod
from typing import List

from trainingData.inputLoading.AudioSample import AudioSample


class SynthesisStep(ABC):
    """
    Abstract class for defining steps during synthesis.
    """

    def __init__(self, synthesis_config: dict):
        """
        Initialises the synthesis step with the given config.
        :param synthesis_config: The config to use.
        """
        self._synthesis_config = synthesis_config
        # Get the step specific config
        self._step_config = copy.copy(synthesis_config.get(self.config_key, {}))

    @property
    def config_key(self):
        """
        Returns the config key that is used for this synthesis step.
        :return: The config key. This is the name of the enum constant but converted to UpperCamelCase.
        """
        # Get the components by splitting at "_" and converting the first letter to upper case.
        return self.__class__.__name__

    @abstractmethod
    def perform(self, samples: List[AudioSample]) -> List[AudioSample]:
        """
        Performs the step on the given samples.
        :param samples: The samples to perform the step on.
        :return: The result of the step.
        """
        pass

from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from pipelineConfig.PipelineConfigLoader import PipelineConfigLoader
from trainingData.inputLoading.AudioSample import AudioSample


class InferenceDataProvider(ABC):
    """
    Provides the inference with data.

    This can either be from a microphone, or from a file.
    """

    def __init__(self, audio_length):
        """
        Creates a data provider that returns samples with the given length in seconds.
        :param audio_length: The audio length to use in seconds.
        """
        self._audio_length = audio_length

    @abstractmethod
    def get_next_audio_input(self) -> Union[AudioSample, np.ndarray]:
        """
        Returns the next audio input.

        :return: The next audio input. Can be an audio sample (then it will be converted to mfccs)
        or directly mfccs.
        """
        pass

    @property
    def does_require_preprocessing(self) -> bool:
        """
        Returns whether the data provider returns a mel frequency spectrum or a raw audio sample.

        If this is True, the returned data is an audio sample. Else it will be a numpy
        array holding the mfccs.

        :return: Whether the data provider returns a raw audio samples for MFCCs.
        """
        return True

    @property
    def has_data(self) -> bool:
        """
        Returns whether there is any more data to provide.

        :return: Whether there is any more data to provide.
        """
        return True

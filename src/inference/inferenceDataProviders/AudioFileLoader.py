import logging
import pathlib
import random

import numpy as np

from inference.inferenceDataProviders.InferenceDataProvider import InferenceDataProvider
from trainingData.inputLoading.AudioSample import AudioSample
from trainingData.inputLoading.SampleLoader import SampleLoader


class AudioFileLoader(InferenceDataProvider):
    """
    Used to load an audio file during inference.
    """

    def __init__(self, audio_length, audio_file_path: pathlib.Path):
        """
        Initializes the AudioFileLoader.

        :param audio_length: The length of the audio file in seconds.
        """
        # Init super
        super().__init__(audio_length)
        # Get a sample loader
        sample_loader = SampleLoader()
        # Load the audio file
        audio_sample, loaded_sample_rate = sample_loader.load_sample_from_path(
            audio_file_path
        )
        # Create the audio sample, no category
        self._audio_sample = AudioSample(audio_sample, loaded_sample_rate, [])
        # Save the sample rate
        self._sample_rate = loaded_sample_rate
        # Get the iterator
        self._chunk_iterator = self._loop_over_sample()
        # Set to True when we are done with the sample
        self._is_done = False

    def _loop_over_sample(self):
        """
        Returns an iterator to loop over the sample. This will always return chunks of the wanted length.

        If the sample does not contain enough data, or we are at the end, this will pad zeros.
        :return: A chunk of the sample of self._audio_length length.
        """
        # Get the length of the sample
        sample_length = len(self._audio_sample.data)
        # Get the length of one chunk in samples
        chunk_length = int(self._audio_length * self._sample_rate)
        # Get the start index
        start_index = 0
        while True:
            # Get the end index
            end_index = start_index + chunk_length
            # Check if we are at the end
            if end_index > sample_length:
                # We are at the end, pad zeros
                end_index = sample_length
            # Get the chunk
            chunk = self._audio_sample.data[start_index:end_index]
            # randomize padding
            pad_length = chunk_length - len(chunk)
            pad_right = random.randint(0, pad_length)
            pad_left = pad_length - pad_right
            # Pad the chunk with zeros
            chunk = np.pad(chunk, (pad_left, pad_right), "constant")
            # Make the end index the start index for next time
            start_index = end_index
            # If we are at the end, we need to reset the start index
            if start_index >= sample_length:
                logging.info("Looped inference sample")
                self._is_done = True
            # Yield the chunk
            yield chunk

    @property
    def has_data(self) -> bool:
        """
        Returns whether the sample has data.

        :return: True if there is data, False if not.
        """
        return not self._is_done

    def get_next_audio_input(self) -> AudioSample:
        """
        Returns the next audio sample from the microphone.

        This always returns the same sample.
        :return: The next audio input.
        """
        if self._is_done:
            # We are done, return None
            return None
        # Get the next chunk
        chunk = next(self._chunk_iterator)
        # -19446 22101 -0.38893990929705213 217.19680839002268
        # -22413 24308 -0.11880385487528344 376.31740929705217
        # -21878 20988 -0.9341269841269841 204.783820861678
        # print('FILE-DATA', chunk, min(chunk), max(chunk), np.mean(chunk), np.mean(abs(chunk)))
        # Return the audio sample
        return AudioSample(chunk, self._sample_rate, [])

import logging
import time
import numpy as np
from pydub import AudioSegment

from inference.inferenceDataProviders.InferenceDataProvider import InferenceDataProvider
from trainingData.inputLoading.AudioSample import AudioSample


_FRAMES_PER_BUFFER = 1024
_SIZE_OF_ONE_SAMPLE_BYTES = 2
_MICROPHONE_SAMPLE_RATE = 44100


class PyAudioMicrophone(InferenceDataProvider):
    """
    Provides data for inference as recorded from a microphone using pyaudio.

    This cannot be used on the board but is useful during development.
    """

    def __init__(self, recording_length):
        """
        Constructor.

        :param recording_length: The audio length to use in seconds.
        """
        # Init super
        super().__init__(recording_length)
        # Local import since it has external dependencies
        import pyaudio

        # Init pyaudio
        self._device = pyaudio.PyAudio()

    def __del__(self):
        """
        Destructor to terminate the device.
        """
        self._device.terminate()

    def get_next_audio_input(self) -> AudioSample:
        """
        Returns the next audio sample from the microphone.

        :return: The next audio input.
        """
        # Local import since it has external dependencies
        import pyaudio

        # Start the stream
        stream = self._device.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=_MICROPHONE_SAMPLE_RATE,
            input=True,
           # input_device_index=1,
            frames_per_buffer=_FRAMES_PER_BUFFER,
        )
        # Calculate the number of samples to record
        data = b""
        # Calculate the number of samples to record
        number_of_samples = int(self._audio_length * _MICROPHONE_SAMPLE_RATE)
        # Start recording
        logging.info("Started recording")
        # Record the data until we have enough, note that one sample is an int16
        number_bytes_in_recording = number_of_samples * _SIZE_OF_ONE_SAMPLE_BYTES
        while len(data) < number_bytes_in_recording:
            # Read the data
            read_data = stream.read(_FRAMES_PER_BUFFER)
            # Limit to the max size when we are at the end
            data += read_data[
                : min(len(read_data), number_bytes_in_recording - len(data))
            ]
        # Stop recording
        logging.info("Stopped recording")
        stream.stop_stream()
        stream.close()
        # Create the numpy array
        data = np.frombuffer(data, dtype=np.int16)
        # -32768 32728 -0.26287414965986394 487.83157029478457
        # -4880 4255 1.0375907029478457 114.20062925170068
        # print('MIC-DATA', data, min(data), max(data), np.mean(data), np.mean(abs(data)))
        # Create the audio sample and return (no category)

        # save live recording as mp3 for later testing, only experimental (now only saving filtered audio in inference\init.py)
        """raw_audio = AudioSegment(
            data.tobytes(),
            frame_rate=_MICROPHONE_SAMPLE_RATE,
            sample_width=2,
            channels=1
        )
        timestamp = int(time.time())
        filename = f"recorded_{timestamp}.mp3"
        raw_audio.export(filename, format="mp3")
        logging.info(f"Saved recorded audio as {filename}")"""


        return AudioSample(
            data, _MICROPHONE_SAMPLE_RATE, []
        )
